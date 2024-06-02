import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
from prettytable import PrettyTable
from tqdm import tqdm
import cv2

from utils.geometry import rotation_angle, angle
from utils.geometry import get_camera_dicts, undistort, distort, recover_pose_from_fundamental, bougnoux_original, \
    get_K, pose_from_F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--first', type=int, default=None)
    parser.add_argument('-nw', '--num_workers', type=int, default=1)
    parser.add_argument('-l', '--load', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()

def get_pairs(file):
    return [tuple(x.split('-')) for x in file.keys() if 'feat' not in x and 'desc' not in x]


def normalize(kp, width, height):
    new_kp = np.copy(kp)

    scale = max(width, height) / 2
    new_kp -= np.array([[width / 2, height / 2]])
    new_kp /= scale

    T = np.array([[scale, 0.0, width / 2], [0.0, scale, height / 2], [0, 0, 1]])

    return new_kp, T


def get_result_dict(info, kp1_distorted, kp2_distorted, F_est, k1_est, k2_est, k1_gt, k2_gt, R_gt, t_gt, K1, K2, T1, T2):
    kp1_undistorted = undistort(kp1_distorted[info['inliers']], k1_est)
    kp2_undistorted = undistort(kp2_distorted[info['inliers']], k2_est)

    kp1_undistorted *= T1[0, 0]
    kp2_undistorted *= T2[0, 0]

    pp1 = T1[:2, 2]
    pp2 = T2[:2, 2]

    kp1_undistorted += pp1[np.newaxis]
    kp2_undistorted += pp2[np.newaxis]

    F_est = np.linalg.inv(T2).T @ F_est @ np.linalg.inv(T1)

    out = {}

    f_1_est, f_2_est = bougnoux_original(F_est, pp1, pp2)
    if np.isnan(f_1_est) or np.isnan(f_2_est):
        out['R_err'] = 180
        out['t_err'] = 180
    else:
        R_est, t_est = pose_from_F(F_est, get_K(f_1_est, pp1), get_K(f_2_est, pp2), kp1_undistorted, kp2_undistorted)
        out['R_err'] = rotation_angle(R_est.T @ R_gt)
        out['t_err'] = angle(t_est, t_gt)

    out['P_err'] = max(out['R_err'], out['t_err'])
    out['k1_err'] = np.abs(k1_est - k1_gt) / np.abs(k1_gt)
    out['k2_err'] = np.abs(k2_est - k2_gt) / np.abs(k2_gt)
    out['info'] = info

    return out



def eval_experiment(x):
    experiment, kp1_distorted, kp2_distorted, k, R_gt, t_gt, T1, T2, K1, K2 = x

    solver = experiment.split('_')[0]

    use_undistorted = False

    ransac_dict = {'max_iterations': 10000, 'max_epipolar_error': 2.0 / (T1[0, 0] + T2[0,0]), 'progressive_sampling': False,
                   'min_iterations': 100}

    if solver == 'F':
        rd_vals = []
        if 's3' in experiment:
            rd_vals = [0.0, -0.5, -1.0]

        start = perf_counter()
        F_cam, info = poselib.estimate_kFk(kp1_distorted, kp2_distorted, rd_vals, use_undistorted, False , ransac_dict,
                                           {'verbose': False, 'max_iterations': 100})

        F, info = poselib.estimate_fundamental(kp1_distorted, kp2_distorted, ransac_dict, {})
        info['runtime'] = 1000 * (perf_counter() - start)
        F_est = F_cam.F
        k1_est = F_cam.camera.params[-1]
        k2_est = k1_est
    elif solver == 'kFk':
        use_9pt = '9pt' in experiment
        start = perf_counter()
        F_cam, info = poselib.estimate_kFk(kp1_distorted, kp2_distorted, [], use_undistorted, use_9pt, ransac_dict,
                                          {'verbose': False, 'max_iterations': 100})
        info['runtime'] = 1000 * (perf_counter() - start)
        F_est = F_cam.F
        k1_est = F_cam.camera.params[-1]
        k2_est = k1_est
    else: # solver == 'k2Fk1'
        use_10pt = '10pt' in experiment
        start = perf_counter()
        F_cam, info = poselib.estimate_k2Fk1(kp1_distorted, kp2_distorted, [], use_undistorted, use_10pt, ransac_dict,
                                            {'verbose': False, 'max_iterations': 100})
        info['runtime'] = 1000 * (perf_counter() - start)
        F_est = F_cam.F
        k1_est = F_cam.camera1.params[-1]
        k2_est = F_cam.camera2.params[-1]

    # if solver == 'kFk'

    result_dict = get_result_dict(info, kp1_distorted, kp2_distorted,
                                  F_est, k1_est, k2_est,
                                  k, k, R_gt, t_gt, K1, K2, T1, T2)
    result_dict['experiment'] = experiment

    return result_dict


def print_results(experiments, results):
    tab = PrettyTable(['solver', 'median R_err', 'mean R_err', 'median t_err', 'mean t_err', 'R_AUC@5', 'R_AUC@10', 'R_AUC@20', 't_AUC@5', 't_AUC@10', 't_AUC@20', 'median time', 'mean time'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        R_errs = np.array([r['R_err'] for r in exp_results])
        t_errs = np.array([r['t_err'] for r in exp_results])
        R_errs[np.isnan(R_errs)] = 180
        t_errs[np.isnan(t_errs)] = 180
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 21)])
        t_res = np.array([np.sum(t_errs < t) / len(t_errs) for t in range(1, 21)])

        times = np.array([x['info']['runtime'] for x in exp_results])

        tab.add_row([exp, np.median(R_errs), np.mean(R_errs),
                     np.median(t_errs), np.mean(t_errs),
                     np.mean(R_res[:5]), np.mean(R_res[:10]), np.mean(R_res),
                     np.mean(t_res[:5]), np.mean(t_res[:10]), np.mean(t_res),
                     np.median(times), np.mean(times)])

        # print(f'{err_name}: \t median: {np.median(errs):0.2f} \t mean: {np.mean(errs):0.2f} \t '
        #       f'auc5: {np.mean(res[:5]):0.2f} \t auc10: {np.mean(res[:10]):0.2f} \t auc20: {np.mean(res):0.2f}')

    # for field in ['inlier_ratio', 'iterations', 'runtime', 'refinements']:
    #     xs = [r['info'][field] for r in results]
    #     tab.add_row([field, np.median(xs), np.mean(xs), '-', '-', '-'])
    #     # print(f'{field}: \t median: {np.median(xs):0.02f} \t mean: {np.mean(xs):0.02f}')

    print(tab)


def eval(args):
    experiments = ['F_7pt', 'F_7pt_s3',
                   'kFk_8pt', 'kFk_9pt',
                   'k2k1_9pt', 'k2Fk1_10pt']

    dataset_path = args.dataset_path
    basename = os.path.basename(dataset_path)
    if args.load:
        results = json.load(os.path.join('results', f'{basename}.json'))
        print_results(experiments, results)

    R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
    T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
    P_file = h5py.File(os.path.join(dataset_path, 'parameters_rd.h5'))
    C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))

    R_dict = {k: np.array(v) for k, v in R_file.items()}
    t_dict = {k: np.array(v) for k, v in T_file.items()}
    w_dict = {k.split('-')[0]: v[0, 0] for k, v in P_file.items()}
    h_dict = {k.split('-')[0]: v[1, 1] for k, v in P_file.items()}
    camera_dicts = get_camera_dicts(os.path.join(dataset_path, 'K.h5'))

    pairs = get_pairs(C_file)

    if args.first is not None:
        pairs = pairs[:args.first]

    def gen_data():
        for img_name_1, img_name_2 in pairs:
            R1 = R_dict[img_name_1]
            t1 = t_dict[img_name_1]
            R2 = R_dict[img_name_2]
            t2 = t_dict[img_name_2]
            K1 = camera_dicts[img_name_1]
            K2 = camera_dicts[img_name_2]

            R_gt = R1.T @ R2
            t_gt = t2 - R_gt @ t1


            matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])

            kp1 = matches[:, :2]
            kp2 = matches[:, 2:4]

            if len(kp1) < 10:
                continue

            # kp1 = kp1[matches[:, 4] <= 0.8]
            # kp2 = kp2[matches[:, 4] <= 0.8]

            kp1_normalized, T1 = normalize(kp1, w_dict[img_name_1], h_dict[img_name_1])
            kp2_normalized, T2 = normalize(kp2, w_dict[img_name_2], h_dict[img_name_2])

            k = -np.random.rand()
            k = 0

            kp1_distorted = distort(kp1_normalized, k)
            kp2_distorted = distort(kp2_normalized, k)

            for experiment in experiments:
                yield experiment, np.copy(kp1_distorted), np.copy(kp2_distorted), k, R_gt, t_gt, T1, T2, K1, K2


    total_length = len(experiments) * len(pairs)

    print(f"Total runs: {total_length} for {len(pairs)} samples")

    if args.num_workers == 1:
        results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
    else:
        pool = Pool(args.num_workers)
        results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

    os.makedirs('results', exist_ok=True)

    with open(os.path.join('results', f'{basename}.json'), 'w') as f:
        json.dump(results, f)

    print("Done")

    print_results(experiments, results)

if __name__ == '__main__':
    args = parse_args()
    eval(args)