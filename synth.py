import numpy as np
import cv2
import poselib
from matplotlib import pyplot as plt
from scipy.spatial.transform import Rotation
from tqdm import tqdm

from utils.geometry import rotation_angle, distort


def generate_points(num_pts, f, distance, depth, width=640, height=480):
    zs = (1 + distance) * f + depth * np.random.rand(num_pts) * f
    xs = (np.random.rand(num_pts) - 0.5) * width * (1 + distance)
    ys = (np.random.rand(num_pts) - 0.5) * height * (1 + distance)
    return np.column_stack([xs, ys, zs, np.ones_like(xs)])

def get_projection(P, X):
    x = P @ X.T
    x = x[:2, :] / x[2, np.newaxis, :]
    return x.T

def visible_in_view(x, width=640, height=480):
    visible = np.logical_and(np.abs(x[:, 0]) <= width / 2, np.abs(x[:, 1]) <= height / 2)
    return visible


def set_axes_equal(ax):
    """
    Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    """

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5 * max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_scene(points, R, t, f, width=640, height=480, color_1='black', color_2='red', name=""):
    c_x_1 = np.array([0.5 * width, 0.5 * width, -0.5 * width, -0.5 * width, 0])
    c_y_1 = np.array([0.5 * height, -0.5 * height, -0.5 * height, 0.5 * height, 0])
    c_z_1 = np.array([f, f, f, f, 0])
    c_z_2 = np.array([f, f, f, f, 0])

    camera2_X = np.row_stack([c_x_1, c_y_1, c_z_2, np.ones_like(c_x_1)])
    c_x_2, c_y_2, c_z_2 = np.column_stack([R.T, -R.T @ t]) @ camera2_X

    # fig = plt.figure()

    ax = plt.axes(projection="3d")
    ax.set_box_aspect([1.0, 1., 1.0])

    ax.plot3D(c_x_1, c_y_1, c_z_1, color_1)
    ax.plot3D(c_x_2, c_y_2, c_z_2, color_2)

    ax.plot3D([c_x_1[0], c_x_1[3]], [c_y_1[0], c_y_1[3]], [c_z_1[0], c_z_1[3]], color_1)
    ax.plot3D([c_x_2[0], c_x_2[3]], [c_y_2[0], c_y_2[3]], [c_z_2[0], c_z_2[3]], color_2)

    for i in range(4):
        ax.plot3D([c_x_1[i], c_x_1[-1]], [c_y_1[i], c_y_1[-1]], [c_z_1[i], c_z_1[-1]], color_1)
        ax.plot3D([c_x_2[i], c_x_2[-1]], [c_y_2[i], c_y_2[-1]], [c_z_2[i], c_z_2[-1]], color_2)

    ax.scatter3D(points[:, 0], points[:, 1], points[:, 2], c='blue')

    set_axes_equal(ax)

    plt.xlabel('X')
    plt.ylabel('Y')
    ax.set_zlabel('Z')
    plt.title(name)


def get_scene(f, k1, k2, R, t, num_pts, X=None, min_distance=1, depth=1, width=640, height=480, sigma_p=0.0, plot=None, seed=None):
    if seed is not None:
        np.random.seed(seed)
    K = np.diag([f, f, 1])


    P1 = K @ np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]])
    P2 = K @ np.column_stack([R, t])

    if X is None:
        X = generate_points(3 * num_pts, f, min_distance, depth, width=width, height=height)
    x1_u = get_projection(P1, X)
    x2_u = get_projection(P2, X)

    if k1 == 0.0:
        x1 = x1_u
    else:
        x1 = distort(x1_u, k1)
    if k2 == 0.0:
        x2 = x2_u
    else:
        x2 = distort(x2_u, k2)


    # visible = visible_in_view(x2, width=width, height=height)

    # x1, x2, X = x1[visible][:num_pts], x2[visible][:num_pts], X[visible]

    # run(f1, f2, x1, x2, scale=scale, name=name)
    # if plot is not None:
    #     plot_scene(X, R1, t1, f1, f2, name=plot)
    #     # plt.savefig(f'{np.round(t)}.png')
    #     plt.show()

    return x1, x2, X


def run_synth():
    f = 1200
    gt_K = np.diag([f, f, 1.0])
    k1 = 5e-7
    k2 = 5e-8
    rd_vals = [1e-7, 1e-8, 1e-9, 0.0]
    rd_vals = []

    R = Rotation.from_euler('xyz', (5, 60, 0), degrees=True).as_matrix()
    c = np.array([2 * f, 0, f])
    # R = Rotation.from_euler('xyz', (theta, 30, 0), degrees=True).as_matrix()
    # c = np.array([f1, y, 0])
    t = -R @ c

    x1, x2, X = get_scene(f, k1, k2, R, t, 100)

    sigmas = [0.0, 0.1, 0.2, 0.3, 0.5, 1.0, 1.5]
    # sigmas = [0.5, 1.0, 1.5]
    # sigmas = [0.0]

    rot_errs = {sigma: [] for sigma in sigmas}
    k1s = {sigma: [] for sigma in sigmas}
    k2s = {sigma: [] for sigma in sigmas}
    inliers = {sigma: [] for sigma in sigmas}
    use_undistorted = False
    use_9pt = True


    for sigma in sigmas:
        for _ in tqdm(range(10)):
            x1, x2, X = get_scene(f, k1, k2, R, t, 100)

            xx1 = x1 + sigma * np.random.randn(*(x1.shape))
            xx2 = x2 + sigma * np.random.randn(*(x1.shape))

            # idxs1 = np.random.permutation(np.arange(30))
            # xx1[:30] = xx1[idxs1]
            # idxs2 = np.random.permutation(np.arange(30, 60))
            # xx2[30:60] = xx2[idxs2]

            if np.isnan(xx1).any():
                print("xx1 NaN")

            if np.isnan(xx2).any():
                print("xx2 NaN")

            camera_dict =  {'model': 'SIMPLE_PINHOLE', 'width': 640, 'height': 480, 'params': [f, 0, 0]}
            ransac_dict = {'max_iterations': 1000, 'max_epipolar_error': 2.0, 'progressive_sampling': False, 'min_iterations': 10}

            # image_pair, out = poselib.estimate_rd_shared_focal_relative_pose(xx1, xx2,np.array([0.0, 0.0]), rd_vals, ransac_dict, {'verbose': False})
            # image_pair, out = poselib.estimate_rd_shared_focal_relative_pose(xx1, xx2,np.array([0.0, 0.0]), [], ransac_dict, {'verbose': False})
            # rot_errs[sigma].append(rotation_angle(image_pair.pose.R.T @ R))
            # k1s[sigma].append(image_pair.camera1.params[-1])
            # k2s[sigma].append(image_pair.camera1.params[-1])
            # inliers[sigma].append(out['num_inliers'])

            # F_cam, out = poselib.estimate_kFk(xx1, xx2, rd_vals, use_undistorted, use_9pt, ransac_dict,
            #                                          {'verbose': False, 'max_iterations': 100})
            # F = F_cam.F
            # kk1 = F_cam.camera.params[-1]
            # kk2 = kk1


            F_cam_pair, out = poselib.estimate_k2Fk1(xx1, xx2, rd_vals, use_undistorted, use_9pt, ransac_dict, {'verbose': False, 'max_iterations':1000})

            F = F_cam_pair.F
            kk1 = F_cam_pair.camera1.params[3]
            kk2 = F_cam_pair.camera2.params[3]

            E_est = gt_K.T @ F @ gt_K
            R1, R2, _ = cv2.decomposeEssentialMat(E_est)

            rot_err = min(rotation_angle(R1.T @ R), rotation_angle(R2.T @ R))

            rot_errs[sigma].append(rot_err)
            k1s[sigma].append(kk1)
            k2s[sigma].append(kk2)
            inliers[sigma].append(out['num_inliers'])

    rot_errs = [np.array(rot_errs[sigma]) for sigma in sigmas]
    inliers = [np.array(inliers[sigma]) for sigma in sigmas]
    k1s = [np.array(k1s[sigma]) for sigma in sigmas]
    k2s = [np.array(k2s[sigma]) for sigma in sigmas]
    plt.title("R")
    plt.boxplot(rot_errs)
    plt.show()

    plt.title("k1s")
    plt.boxplot(k1s)
    plt.show()

    plt.title("inliers")
    plt.boxplot(inliers)
    plt.show()

    plt.title("k2s")
    plt.boxplot(k2s)
    plt.show()

if __name__ == '__main__':
    # for _ in range(100):
    #     e = -3 + 3 * np.random.rand()
    #     k = 10 ** e
    #
    #     x = 200 * np.random.randn(10, 2)
    #
    #     x_u = undistort(x, -1e-8)
    #     xx = distort(x_u, -1e-8)
    #     diff = x - xx
    #
    #     if np.any(np.abs(diff) > 1e-8):
    #         print("Fail")

    run_synth()
