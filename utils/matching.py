import numpy as np
import lightglue

def get_area(pts):
    width = np.max(pts[:, 0]) - np.min(pts[:, 0])
    height = np.max(pts[:, 1]) - np.min(pts[:, 1])
    return width * height


def get_matcher_string(args):
    if args.resize is None:
        resize_str = 'noresize'
    else:
        resize_str = str(args.resize)

    return f'features_{args.features}_{resize_str}_{args.max_features}'


def get_extractor(args):
    if args.features == 'superpoint':
        extractor = lightglue.SuperPoint(max_num_keypoints=args.max_features).eval().cuda()
    elif args.features == 'disk':
        extractor = lightglue.DISK(max_num_keypoints=args.max_features).eval().cuda()
    elif args.features == 'sift':
        extractor = lightglue.SIFT(max_num_keypoints=args.max_features).eval().cuda()
    else:
        raise NotImplementedError

    return extractor
