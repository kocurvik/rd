import argparse
import json
import os
from multiprocessing import Pool
from time import perf_counter

import h5py
import numpy as np
import poselib
from matplotlib import pyplot as plt
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
    parser.add_argument('-g', '--graph', action='store_true', default=False)
    parser.add_argument('-s', '--synth', action='store_true', default=False)
    parser.add_argument('-e', '--syntheq', action='store_true', default=False)
    parser.add_argument('feature_file')
    parser.add_argument('dataset_path')

    return parser.parse_args()

def get_pairs(file):
    return [tuple(x.split('-')) for x in file.keys() if 'feat' not in x and 'desc' not in x]


def normalize(kp, width, height):
    new_kp = np.copy(kp)

    scale = max(width, height)
    new_kp -= np.array([[width / 2, height / 2]])
    new_kp /= scale

    T = np.array([[scale, 0.0, width / 2], [0.0, scale, height / 2], [0, 0, 1]])

    return new_kp, T


def k_err(k_gt, k_est):
    return abs((1 / (1 + k_gt)) - (1 /(1 + k_est))) / abs(( 1 / (1 + k_gt)))



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

    R_est, t_est = pose_from_F(F_est, K1, K2, kp1_undistorted, kp2_undistorted)

    out['K1_gt'] = K1.tolist()
    out['K2_gt'] = K2.tolist()

    out['R_err'] = rotation_angle(R_est.T @ R_gt)
    out['t_err'] = angle(t_est, t_gt)
    out['R'] = R_est.tolist()
    out['R_gt'] = R_gt.tolist()
    out['t'] = R_est.tolist()
    out['t_gt'] = R_gt.tolist()

    out['P_err'] = max(out['R_err'], out['t_err'])
    out['k1_err'] = k_err(k1_gt, k1_est)
    out['k1'] = k1_est
    out['k1_gt'] = k1_gt

    out['k2_err'] = k_err(k2_gt, k2_est)
    out['k2'] = k2_est
    out['k2_gt'] = k2_gt

    out['info'] = info

    return out



def eval_experiment(x):
    iters, experiment, kp1_distorted, kp2_distorted, k1, k2, R_gt, t_gt, T1, T2, K1, K2 = x

    solver = experiment.split('_')[0]

    use_undistorted = False

    mean_scale = (T1[0, 0] + T2[0,0]) / 2
    # mean_scale = 1.0

    if iters is None:
        ransac_dict = {'max_iterations': 10000, 'max_epipolar_error': 3.0 / mean_scale, 'progressive_sampling': False,
                       'min_iterations': 100}
    else:
        ransac_dict = {'max_iterations': iters, 'max_epipolar_error': 3.0 / mean_scale, 'progressive_sampling': False,
                       'min_iterations': iters}

    if solver == 'Feq':
        rd_vals = [0.0]
        if 's3' in experiment:
            rd_vals = [0.0, -0.4, -0.8]

        start = perf_counter()
        F_cam, info = poselib.estimate_kFk(kp1_distorted, kp2_distorted, rd_vals, use_undistorted, False, ransac_dict,
                                           {'verbose': False, 'max_iterations': 100})

        # F, info = poselib.estimate_fundamental(kp1_distorted, kp2_distorted, ransac_dict, {})

        info['runtime'] = 1000 * (perf_counter() - start)
        F_est = F_cam.F
        k1_est = F_cam.camera.params[-1]
        # k1_est = 0.0
        k2_est = k1_est
    elif solver == 'F':
        rd_vals = [0.0]
        if 's3' in experiment:
            rd_vals = [0.0, -0.4, -0.8]

        start = perf_counter()
        F_cam, info = poselib.estimate_k2Fk1(kp1_distorted, kp2_distorted, rd_vals, use_undistorted, False, ransac_dict,
                                           {'verbose': False, 'max_iterations': 100})

        # F, info = poselib.estimate_fundamental(kp1_distorted, kp2_distorted, ransac_dict, {})

        info['runtime'] = 1000 * (perf_counter() - start)
        F_est = F_cam.F
        k1_est = F_cam.camera1.params[-1]
        k2_est = F_cam.camera2.params[-1]
        # k1_est = 0.0
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
                                  k1, k2, R_gt, t_gt, K1, K2, T1, T2)
    result_dict['experiment'] = experiment

    return result_dict


def print_results(experiments, results, eq_only=False):
    tab = PrettyTable(['solver', 'LO', 'median pose err', 'mean pose err',
                       'Pose AUC@5', 'Pose AUC@10', 'Pose AUC@20',
                       'median k err', 'mean k err',
                       'k AUC@0.05', 'k AUC@0.1' ,
                       'median time', 'mean time'])
    tab.align["solver"] = "l"
    tab.float_format = '0.2'

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        if eq_only:
            exp_results = [x for x in exp_results if x['k1_gt'] == x['k2_gt']]# and x['K1_gt'] == x['K2_gt']]
        else:
            exp_results = [x for x in exp_results if x['k1_gt'] != x['k2_gt']]  # and x['K1_gt'] == x['K2_gt']]

        p_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        p_errs[np.isnan(p_errs)] = 180
        p_res = np.array([np.sum(p_errs < t) / len(p_errs) for t in range(1, 21)])

        k_errs = [k_err(r['k1_gt'], r['k1']) for r in exp_results]
        k_errs.extend([k_err(r['k2_gt'], r['k2']) for r in exp_results])
        k_errs = np.array(k_errs)

        k_errs[np.isnan(k_errs)] = 1.0
        k_res = np.array([np.sum(k_errs < t / 100) / len(k_errs) for t in range(1, 21)])


        times = np.array([x['info']['runtime'] for x in exp_results])
        inliers = np.array([x['info']['inlier_ratio'] for x in exp_results])

        lo = 'kFk' if 'kFk' in exp or 'eq' in exp else 'k2Fk1'
        exp_name = exp.replace('_', ' ').replace('eq','')


        tab.add_row([exp_name, lo, np.median(p_errs), np.mean(p_errs),
                     np.mean(p_res[:5]), np.mean(p_res[:10]), np.mean(p_res),
                     np.median(k_errs), np.mean(k_errs),
                     np.mean(k_res[:5]), np.mean(k_res[:10]),
                     np.median(times), np.mean(times)])
    print(tab)

    print('latex')

    print(tab.get_formatted_string('latex'))

def draw_cumplots(experiments, results, eq_only=False):
    plt.figure()
    plt.xlabel('Pose error')
    plt.ylabel('Portion of samples')

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        if eq_only:
            exp_results = [x for x in exp_results if x['k1_gt'] == x['k2_gt']]# and x['K1_gt'] == x['K2_gt']]
        else:
            exp_results = [x for x in exp_results if x['k1_gt'] != x['k2_gt']]  # and x['K1_gt'] == x['K2_gt']]

        lo = 'kFk' if 'kFk' in exp or 'eq' in exp else 'k2Fk1'
        exp_name = exp.replace('_', ' ').replace('eq','')
        label = f'{exp_name} + {lo} LO'

        R_errs = np.array([max(r['R_err'], r['t_err']) for r in exp_results])
        R_res = np.array([np.sum(R_errs < t) / len(R_errs) for t in range(1, 180)])
        plt.plot(np.arange(1, 180), R_res, label = label)

    plt.legend()
    plt.show()

    plt.figure()
    plt.xlabel('k error')
    plt.ylabel('Portion of samples')

    for exp in experiments:
        exp_results = [x for x in results if x['experiment'] == exp]

        if eq_only:
            exp_results = [x for x in exp_results if x['k1_gt'] == x['k2_gt']]# and x['K1_gt'] == x['K2_gt']]
        else:
            exp_results = [x for x in exp_results if x['k1_gt'] != x['k2_gt']]  # and x['K1_gt'] == x['K2_gt']]

        lo = 'kFk' if 'kFk' in exp or 'eq' in exp else 'k2Fk1'
        exp_name = exp.replace('_', ' ').replace('eq','')
        label = f'{exp_name} + {lo} LO'

        k_errs = [k_err(r['k1_gt'], r['k1']) for r in exp_results]
        k_errs.extend([k_err(r['k2_gt'], r['k2']) for r in exp_results])
        k_errs = np.array(k_errs)
        k_res = np.array([np.sum(k_errs < t / 100) / len(k_errs) for t in range(1, 201)])
        plt.plot(np.arange(1, 201) / 100, k_res, label = label)

    plt.legend()
    plt.show()


def draw_results(results, experiments, iterations_list):
    plt.figure()

    for experiment in experiments:
        experiment_results = [x for x in results if x['experiment'] == experiment]

        xs = []
        ys = []

        for iterations in iterations_list:
            iter_results = [x for x in experiment_results if x['info']['iterations'] == iterations]
            mean_runtime = np.mean([x['info']['runtime'] for x in iter_results])
            errs = np.array([r['P_err'] for r in iter_results])
            errs[np.isnan(errs)] = 180
            AUC10 = np.mean(np.array([np.sum(errs < t) / len(errs) for t in range(1, 11)]))

            xs.append(mean_runtime)
            ys.append(AUC10)

        plt.semilogx(xs, ys, label=experiment, marker='*')

    plt.xlabel('Mean runtime (ms)')
    plt.ylabel('AUC@10$\\deg$')
    plt.legend()
    plt.show()


def eval(args):
    experiments = ['Feq_7pt', 'Feq_7pt_s3',
                   'kFk_8pt', 'kFk_9pt',
                   'k2k1_9pt', 'k2Fk1_10pt',
                   'F_7pt', 'F_7pt_s3']

    dataset_path = args.dataset_path
    basename = os.path.basename(dataset_path)

    matches_basename = os.path.basename(args.feature_file)

    if args.graph:
        basename = f'{basename}-graph'
        iterations_list = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]
    else:
        iterations_list = [None]

    if args.load:
        s_string = ""
        if args.synth:
            s_string = "-synth"
            if args.syntheq:
                s_string = "-syntheq"

        with open(os.path.join('results', f'{basename}-{matches_basename}{s_string}.json'), 'r') as f:
            results = json.load(f)


    else:



        R_file = h5py.File(os.path.join(dataset_path, 'R.h5'))
        T_file = h5py.File(os.path.join(dataset_path, 'T.h5'))
        P_file = h5py.File(os.path.join(dataset_path, 'parameters_rd.h5'))
        C_file = h5py.File(os.path.join(dataset_path, f'{args.feature_file}.h5'))

        R_dict = {k: np.array(v) for k, v in R_file.items()}
        t_dict = {k: np.array(v) for k, v in T_file.items()}
        w_dict = {k.split('-')[0]: v[0, 0] for k, v in P_file.items()}
        h_dict = {k.split('-')[0]: v[1, 1] for k, v in P_file.items()}
        k_dict = {k.split('-')[0]: v[2, 2] for k, v in P_file.items()}
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

                R_gt = np.dot(R2, R1.T)
                t_gt = t2 - np.dot(R_gt, t1)


                matches = np.array(C_file[f'{img_name_1}-{img_name_2}'])

                kp1 = matches[:, :2]
                kp2 = matches[:, 2:4]

                if len(kp1) < 10:
                    continue

                # F, info = poselib.estimate_fundamental(kp1, kp2)
                # R, t = pose_from_F(F, K1, K2, kp1, kp2)
                #
                # print(angle(t,t_gt), info['inlier_ratio'])

                # kp1 = kp1[matches[:, 4] <= 0.8]
                # kp2 = kp2[matches[:, 4] <= 0.8]

                kp1_normalized, T1 = normalize(kp1, w_dict[img_name_1], h_dict[img_name_1])
                kp2_normalized, T2 = normalize(kp2, w_dict[img_name_2], h_dict[img_name_2])

                if args.synth:
                    k1 = -np.random.rand()
                    if args.syntheq:
                        k2 = k1
                    else:
                        k2 = -np.random.rand()

                    kp1_distorted = distort(kp1_normalized, k1)
                    kp2_distorted = distort(kp2_normalized, k2)
                else:
                    kp1_distorted = kp1_normalized
                    kp2_distorted = kp2_normalized
                    k1 = k_dict[img_name_1]
                    k2 = k_dict[img_name_2]

                for experiment in experiments:
                    for iterations in iterations_list:
                        yield iterations, experiment, np.copy(kp1_distorted), np.copy(kp2_distorted), k1, k2, R_gt, t_gt, T1, T2, K1, K2


        total_length = len(experiments) * len(pairs) * len(iterations_list)

        print(f"Total runs: {total_length} for {len(pairs)} samples")

        if args.num_workers == 1:
            results = [eval_experiment(x) for x in tqdm(gen_data(), total=total_length)]
        else:
            pool = Pool(args.num_workers)
            results = [x for x in pool.imap(eval_experiment, tqdm(gen_data(), total=total_length))]

        os.makedirs('results', exist_ok=True)

        s_string = ""
        if args.synth:
            s_string = "-synth"
            if args.syntheq:
                s_string = "-syntheq"
        with open(os.path.join('results', f'{basename}-{matches_basename}{s_string}.json'), 'w') as f:
            json.dump(results, f)

        print("Done")

    print("Printing results for all combinations")
    print_results(experiments, results)
    # draw_cumplots(experiments, results)

    print("Printing results for pairs with equal intrinsics")
    print_results(experiments, results, eq_only=True)
    # draw_cumplots(experiments, results, eq_only=True)

    if args.graph:
        draw_results(results, experiments, iterations_list)

if __name__ == '__main__':
    args = parse_args()
    eval(args)