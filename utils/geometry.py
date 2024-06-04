import h5py
import numpy as np
import cv2

def angle(x, y):
    x = x.ravel()
    y = y.ravel()
    return np.rad2deg(np.arccos(np.dot(x, y) / (np.linalg.norm(x) * np.linalg.norm(y))))

def rotation_angle(R):
    return np.rad2deg(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1)))


def get_camera_dicts(K_file_path):
    K_file = h5py.File(K_file_path)

    d = {}

    # Treat data from Charalambos differently since it is in pairs
    if 'K1_K2' in K_file_path:

        for k, v in K_file.items():
            key1, key2 = k.split('-')
            if key1 not in d.keys():
                K1 = np.array(v)[0, 0]
                # d[key1] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K1[0, 2]), 'height': int(2 * K1[1,2]), 'params': [K1[0, 0], K1[0, 2], K1[1, 2]]}
                d[key1] = K1
            if key2 not in d.keys():
                K2 = np.array(v)[0, 1]
                d[key2] = K2
                # d[key2] = {'model': 'SIMPLE_PINHOLE', 'width': int(2 * K2[0, 2]), 'height': int(2 * K2[1,2]), 'params': [K2[0, 0], K2[0, 2], K2[1, 2]]}

        return d

    for key, v in K_file.items():
        K = np.array(v)
        d[key] = K
        # d[key] = {'model': 'PINHOLE', 'width': int(2 * K[0, 2]), 'height': int(2 * K[1,2]), 'params': [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]}

    return d


def undistort(x, k):
    if k == 0.0:
        return x
    return x / (1 + k * (x[:, :1] ** 2 + x[:, 1:] ** 2))


def distort(u, k):
    if k == 0.0:
        return u

    x_u = u[:, :1]
    y_u = u[:, 1:]

    ru2 = x_u ** 2 + y_u ** 2
    ru = np.sqrt(ru2)

    dist_sign = np.sign(k)

    rd = (0.5 / k) / ru - dist_sign * np.sqrt((0.25 / (k * k)) /ru2 - 1 / k)
    rd /= ru

    return rd * u


def recover_pose_from_fundamental(F, K1, K2, pts1, pts2):
    # Compute the essential matrix E from the fundamental matrix F
    E = K2.T @ F @ K1

    # Decompose the essential matrix into rotation and translation
    # This function returns the possible rotation matrices and translation vectors
    R1, R2, t = cv2.decomposeEssentialMat(E)

    # Ensure t is a column vector
    t = t.reshape(3, 1)

    # Four possible solutions: (R1, t), (R1, -t), (R2, t), (R2, -t)
    possible_solutions = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    # To determine the correct solution, we use the cheirality condition
    # We need to triangulate points and check which solution has points in front of both cameras
    correct_solution = None
    max_positive_depths = -1

    for R, t in possible_solutions:
        # Create projection matrices for both cameras
        P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
        P2 = K2 @ np.hstack((R, t))

        points_4d_hom = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)

        # Convert homogeneous coordinates to 3D
        points_3d = points_4d_hom[:3] / points_4d_hom[3]

        # Check the number of points in front of both cameras
        num_positive_depths = np.sum((points_3d[2, :] > 0))

        if num_positive_depths > max_positive_depths:
            max_positive_depths = num_positive_depths
            correct_solution = (R, t)

    R, t = correct_solution
    return R, t


def bougnoux_original(F, p1=np.array([0, 0]), p2=np.array([0, 0])):
    ''' Returns squared focal losses estimated using the Bougnoux formula with given principal points

    :param F: 3 x 3 Fundamental matrix
    :param p1: 2-dimensional coordinates of the principal point of the first camera
    :param p2: 2-dimensional coordinates of the principal point of the first camera
    :return: the estimated squared focal lengths for the two cameras
    '''
    p1 = np.append(p1, 1).reshape(3, 1)
    p2 = np.append(p2, 1).reshape(3, 1)
    try:
        e2, _, e1 = np.linalg.svd(F)
    except Exception:
        return np.nan, np.nan

    e1 = e1[2, :]
    e2 = e2[:, 2]


    s_e2 = np.array([
        [0, -e2[2], e2[1]],
        [e2[2], 0, -e2[0]],
        [-e2[1], e2[0], 0]
    ])
    s_e1 = np.array([
        [0, -e1[2], e1[1]],
        [e1[2], 0, -e1[0]],
        [-e1[1], e1[0], 0]
    ])

    II = np.diag([1, 1, 0])

    f1 = (-p2.T @ s_e2 @ II @ F @ (p1 @ p1.T) @ F.T @ p2) / (p2.T @ s_e2 @ II @ F @ II @ F.T @ p2)
    f2 = (-p1.T @ s_e1 @ II @ F.T @ (p2 @ p2.T) @ F @ p1) / (p1.T @ s_e1 @ II @ F.T @ II @ F @ p1)

    return np.sqrt(f1[0, 0]), np.sqrt(f2[0, 0])


def get_K(f, p=np.array([0, 0])):
    return np.array([[f, 0, p[0]], [0, f, p[1]], [0, 0, 1]])


def pose_from_F(F, K1, K2, kp1, kp2):
    try:
        K1_inv = np.linalg.inv(K1)
        K2_inv = np.linalg.inv(K2)

        E = K2.T @(F @ K1)

        # print(np.linalg.svd(E)[1])

        kp1 = np.column_stack([kp1, np.ones(len(kp1))])
        kp2 = np.column_stack([kp2, np.ones(len(kp1))])

        kp1_unproj = (K1_inv @ kp1.T).T
        kp1_unproj = kp1_unproj[:, :2] / kp1_unproj[:, 2, np.newaxis]
        kp2_unproj = (K2_inv @ kp2.T).T
        kp2_unproj = kp2_unproj[:, :2] / kp2_unproj[:, 2, np.newaxis]

        _, R, t, mask = cv2.recoverPose(E, kp1_unproj, kp2_unproj)
    except:
        print("Pose exception!")
        return np.eye(3), np.ones(3)

    return R, t
