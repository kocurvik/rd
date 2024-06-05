import argparse
import itertools
import os
import random
from pathlib import Path
import xml.etree.ElementTree as ET
import ntpath

import cv2
import h5py
import joblib
import numpy as np
import torch
from scipy.spatial.transform import Rotation
from tqdm import tqdm
import lightglue
from lightglue.utils import load_image, rbd

from utils.matching import get_extractor, get_matcher_string, get_area
from utils.read_write_colmap import cam_to_K, read_model


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-n', '--num_samples', type=int, default=None)
    parser.add_argument('-s', '--seed', type=int, default=100)
    parser.add_argument('-f', '--features', type=str, default='superpoint')
    parser.add_argument('-mf', '--max_features', type=int, default=2048)
    parser.add_argument('-r', '--resize', type=int, default=None)
    parser.add_argument('-e', '--equal', action='store_true', default=False)
    parser.add_argument('--recalc', action='store_true', default=False)
    parser.add_argument('out_path')
    parser.add_argument('dataset_path')

    return parser.parse_args()


def create_gt_h5(images, out_dir, args):
    exist = [os.path.exists(os.path.join(out_dir, f'{x}.h5')) for x in ['K', 'R', 'T', 'parameters_rd']]
    if not False in exist and not args.recalc:
        print(f"GT info exists in {out_dir} - not creating it anew")
        return

    print(f"Writing GT info to {out_dir}")
    fK = h5py.File(os.path.join(out_dir, 'K.h5'), 'w')
    fR = h5py.File(os.path.join(out_dir, 'R.h5'), 'w')
    fT = h5py.File(os.path.join(out_dir, 'T.h5'), 'w')
    fH = h5py.File(os.path.join(out_dir, 'parameters_rd.h5'), 'w')

    X = np.array([[1.0, -1.0, -1.0], [-1.0, 1.0, 1.0], [-1.0, 1.0, 1.0]])
    for image in images:
        name = image['name']
        t = image['t']
        # t[2] *= -1
        # R = X * image['R']
        R = image['R']
        w = image['width']
        h = image['height']
        K = image['K']
        k_rd = image['k_rd']

        hwK = np.array([[w, 0, w / 2], [0, h, h / 2], [0, 0, k_rd]])

        fR.create_dataset(name, shape=(3, 3), data=R)
        fT.create_dataset(name, shape=(3, 1), data=t.reshape(3, 1))
        fK.create_dataset(name, shape=(3, 3), data=K)
        fH.create_dataset(f'{name}-hwK', shape=(3, 3), data=hwK)


def extract_features(img_dir_path, images, out_dir, args):
    # extractor = lightglue.SuperPoint(max_num_keypoints=2048).eval().cuda()
    extractor = get_extractor(args)
    out_path = os.path.join(out_dir, f"{get_matcher_string(args)}.pt")

    if os.path.exists(out_path) and not args.recalc:
        print(f"Features already found in {out_path}")
        return

    print("Extracting features")
    feature_dict = {}

    for img_id, img in tqdm(enumerate(images), total=len(images)):
        img_path = os.path.join(img_dir_path, img['filename'].replace('\\', os.sep))
        name = img['name']
        image_tensor = load_image(img_path).cuda()

        if img['width'] != image_tensor.size(-1):
            if img['width'] == image_tensor.size(-2):
                image_tensor = torch.rot90(image_tensor, 1, (-2, -1))
                print(f"Rotated image: {img_path}!")
                # continue
            else:
                print(f"Image dimensions do not comply with camera width and height for: {img_path} - skipping!")
                continue

        kp_tensor = extractor.extract(image_tensor, resize=args.resize)
        feature_dict[name] = kp_tensor

    torch.save(feature_dict, out_path)
    print("Features saved to: ", out_path)


def get_overlap_areas(images, pts, img_ids):
    img_id1, img_id2 = img_ids
    imgs = list(images[x] for x in img_ids)
    img_1, img_2 = imgs

    img_1_point3D_ids = np.array(img_1['points'])
    img_2_point3D_ids = np.array(img_2['points'])

    overlap = set(img_1_point3D_ids).intersection(set(img_2_point3D_ids))

    if len(overlap) < 5:
        return 0.0, 0.0

    pts_img_1 = []
    pts_img_2 = []

    for pt_id in list(overlap):
        pt = pts[pt_id]

        idx1 = np.where(pt['images'] == img_id1)[0][0]
        idx2 = np.where(pt['images'] == img_id2)[0][0]

        pts_img_1.append(pt['coords'][idx1])
        pts_img_2.append(pt['coords'][idx2])

    pts_img_1 = np.array(pts_img_1)
    pts_img_2 = np.array(pts_img_2)

    area_1 = get_area(pts_img_1) / (img_1['width'] * img_1['height'])
    area_2 = get_area(pts_img_2) / (img_2['width'] * img_2['height'])

    return area_1, area_2


def create_pairs(out_dir, images, pts, args):
    np.random.seed(args.seed)
    random.seed(args.seed)
    output = 0

    features = torch.load(os.path.join(out_dir, f"{get_matcher_string(args)}.pt"))

    matcher = lightglue.LightGlue(features=args.features).eval().cuda()

    name_str = f'pairs-{get_matcher_string(args)}-LG'

    if args.num_samples is None or not args.equal:
        h5_path = os.path.join(out_dir, f'{name_str}.h5')
        h5_file = h5py.File(h5_path, 'w')
        pairs = []
        print("Writing matches to: ", h5_path)

    if args.equal:
        h5_path_eq = os.path.join(out_dir, f'{name_str}_eq.h5')
        h5_file_eq = h5py.File(h5_path_eq, 'w')
        pairs_eq = []
        print("Writing eq matches to: ", h5_path_eq)

    id_list = list(range(len(images)))

    if args.num_samples is None:
        img_ids_list = list(itertools.combinations(id_list, 2))
        total = len(img_ids_list)
    else:
        total = args.num_samples

    all_counter = 0

    with tqdm(total=total) as pbar:
        while output < total:
            if args.num_samples is not None:
                img_ids = random.sample(id_list, 2)
            else:
                if all_counter >= len(img_ids_list):
                    break
                img_ids = img_ids_list[all_counter]
                all_counter += 1
                pbar.update(1)

            label = '-'.join([images[x]['name'] for x in img_ids])
            img_1, img_2 = (images[x] for x in img_ids)

            # if 'GO' in img_1['filename'] or 'GO' in img_2['filename']:
            #     continue

            if args.num_samples is None or not args.equal:
                if label in h5_file:
                    continue
            else:
                if label in h5_file_eq:
                    continue

            if args.num_samples is not None and args.equal:
                if img_1['calibration_group'] != img_2['calibration_group'] or img_1['calibration_group'] == -1 or img_2['calibration_group'] == -1:
                    continue



            if not img_1['name'] in features.keys() or not img_2['name'] in features.keys():
                continue

            area_1, area_2 = get_overlap_areas(images, pts, img_ids)
            if area_1 > 0.1 and area_2 > 0.1:
                feats_1 = features[img_1['name']]
                feats_2 = features[img_2['name']]

                out_12 = matcher({'image0': feats_1, 'image1': feats_2})

                scores_12 = out_12['matching_scores0'][0].detach().cpu().numpy()

                matches_12 = out_12['matches0'][0].detach().cpu().numpy()

                idxs = []

                for idx_1, idx_2 in enumerate(matches_12):
                    if idx_2 != -1:
                        idxs.append((idx_1, idx_2))

                if len(idxs) < 20:
                    continue

                out_array = np.empty([len(idxs), 5])

                for i, idx in enumerate(idxs):
                    idx_1, idx_2 = idx
                    point_1 = feats_1['keypoints'][0, idx_1].detach().cpu().numpy()
                    point_2 = feats_2['keypoints'][0, idx_2].detach().cpu().numpy()
                    score_12 = scores_12[idx_1]
                    out_array[i] = np.array([*point_1, *point_2, score_12])

                if args.num_samples is None or not args.equal:
                    h5_file.create_dataset(label, shape=out_array.shape, data=out_array)
                    pairs.append(label.replace('-', ' '))

                if args.equal and img_1['calibration_group'] == img_2['calibration_group'] and img_1[
                    'calibration_group'] != -1:
                    h5_file_eq.create_dataset(label, shape=out_array.shape, data=out_array)
                    pairs_eq.append(label.replace('-', ' '))

                if args.num_samples is not None:
                    pbar.update(1)
                    output += 1

    if args.num_samples is None or not args.equal:
        pairs_txt_path = os.path.join(out_dir, f'{name_str}.txt')
        print("Writing list of pairs to: ", pairs_txt_path)
        with open(pairs_txt_path, 'w') as f:
            f.writelines(line + '\n' for line in pairs)

    if args.equal:
        pairs_txt_path = os.path.join(out_dir, f'{name_str}_eq.txt')
        print("Writing list of eq pairs to: ", pairs_txt_path)
        with open(pairs_txt_path, 'w') as f:
            f.writelines(line + '\n' for line in pairs_eq)


def get_calib_params(x):
    focal = x[0]

    w = x[7]
    h = x[8]

    k_rd = x[1] * (max(w, h)) ** 2 / (focal ** 2)

    cx = x[5] + w / 2
    cy = x[6] + h / 2

    K = np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]])
    return K, k_rd, w, h


def read_bundler(bundler_path, rc_path, img_path, img_list_path=None):
    with open(bundler_path, 'r') as f:
        lines = f.readlines()

    num_images, num_points = (int(x) for x in lines[1].split(' '))
    print(f"Loading {bundler_path} with {num_images} images and {num_points} points")

    images = []

    tree = ET.parse(rc_path)
    root = tree.getroot()
    for idx, element in enumerate(root):
        if element.tag == 'source':
            break

    img_data = {}
    rc_filenames = []

    for element in root[idx]:
        p = ntpath.normpath(element.attrib['fileName'])

        if p.split('\\')[-2] == ntpath.basename(img_path):
            filename = ntpath.basename(p)
        else:
            filename = ntpath.join(p.split('\\')[-2], p.split('\\')[-1])

        rc_filenames.append(filename)
        try:
            img_data[filename] = {'calibration_group': element.attrib['calibrationGroup'],
                                  'distortion_group': element.attrib['distortionGroup']}
        except:
            img_data[filename] = {'calibration_group': -1,
                                  'distortion_group': -1}

    if img_list_path is None:
        # we are assuming all images were used for the given component
        img_list = rc_filenames
    else:
        with open(img_list_path, 'r', encoding='utf-16-le') as f:
            l = f.readlines()
        img_list = []
        for line in l:
            p = ntpath.normpath(line.strip())
            if p.split('\\')[-2] == ntpath.basename(img_path):
                img_list.append(ntpath.basename(line).strip())
            else:
                img_list.append(ntpath.join(p.split('\\')[-2], p.split('\\')[-1]))

    assert len(img_list) == num_images
    print(f"Loaded: {len(img_data)} from {rc_path}")

    for i in range(num_images):
        img = {'points': []}
        img['filename'] = img_list[i]
        img['calibration_group'] = img_data[img['filename']]['calibration_group']
        img['distortion_group'] = img_data[img['filename']]['distortion_group']
        img['name'] = img['filename'].split('.')[0]
        start = 2 + i * 5
        end = 7 + i * 5
        calib_line = np.fromstring(lines[start], sep=' ')
        img['R'] = np.fromstring(' '.join(lines[start + 1: end - 1]), sep=' ').reshape(3, 3)
        img['t'] = np.fromstring(lines[end - 1], sep=' ')

        K, k_rd, width, height = get_calib_params(calib_line)

        img['K'] = K
        img['k_rd'] = k_rd
        img['width'] = width
        img['height'] = height

        images.append(img)

    points_start_point = end
    points = []

    for i in range(num_points):
        point = {}

        start = points_start_point + i * 3

        point['XYZ'] = np.fromstring(lines[start], sep=' ')
        point['color'] = np.fromstring(lines[start + 1], sep=' ').astype(int)
        views = np.fromstring(lines[start + 2], sep=' ')
        views = views[1:].reshape(int(views[0]), 4)

        point['images'] = views[:, 0].astype(int)
        point['coords'] = views[:, 2:]

        for cam_id in point['images']:
            images[cam_id]['points'].append(i)

        points.append(point)

    print("Done!")

    return images, points


def get_paths(dataset_path):
    basename = os.path.basename(dataset_path)

    if basename == 'rotunda_new':
        bundler_path = os.path.join(dataset_path, 'bundler.out')
        rc_path = os.path.join(dataset_path, 'project.rcproj')
        img_path = os.path.join(dataset_path, 'images_all')
        img_list_path = None
    elif basename == 'st_vitus_all':
        bundler_path = os.path.join(dataset_path, 'st_vitus_component_2_viktor.out')
        rc_path = os.path.join(dataset_path, 'rc.rcproj')
        img_path = os.path.join(dataset_path, 'images')
        img_list_path = os.path.join(dataset_path, 'images_component_2.imagelist')
    else:
        raise NotImplementedError

    return bundler_path, rc_path, img_path, img_list_path


def prepare_single(args):
    dataset_path = Path(args.dataset_path)

    bundler_path, rc_path, img_path, img_list_path = get_paths(dataset_path)

    images, points = read_bundler(bundler_path, rc_path, img_path, img_list_path)

    out_dir = args.out_path
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    create_gt_h5(images, out_dir, args)
    extract_features(img_path, images, out_dir, args)
    create_pairs(out_dir, images, points, args)


if __name__ == '__main__':
    args = parse_args()
    prepare_single(args)