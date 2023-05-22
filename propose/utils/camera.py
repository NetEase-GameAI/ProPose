import numpy as np


def world2cam(world_coord, R, T):
    cam_coord = np.dot(R, world_coord - T)
    return cam_coord


def weak_cam2pixel(cam_coord, root_z, f, c):
    x = cam_coord[:, 0] / (root_z + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (root_z + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)

    avg_f = (f[0] + f[1]) / 2
    cam_param = np.array([avg_f / (root_z + 1e-8), c[0], c[1]])
    return img_coord, cam_param


def cam2pixel(cam_coord, f, c):
    x = cam_coord[:, 0] / (cam_coord[:, 2] + 1e-8) * f[0] + c[0]
    y = cam_coord[:, 1] / (cam_coord[:, 2] + 1e-8) * f[1] + c[1]
    z = cam_coord[:, 2]
    img_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return img_coord


def cam2pixel_matrix(cam_coord, intrinsic_param):
    cam_coord = cam_coord.transpose(1, 0)
    cam_homogeneous_coord = np.concatenate((cam_coord, np.ones((1, cam_coord.shape[1]), dtype=np.float32)), axis=0)
    img_coord = np.dot(intrinsic_param, cam_homogeneous_coord) / (cam_coord[2, :] + 1e-8)
    img_coord = np.concatenate((img_coord[:2, :], cam_coord[2:3, :]), axis=0)
    return img_coord.transpose(1, 0)


def pixel2cam(pixel_coord, f, c):
    x = (pixel_coord[:, 0] - c[0]) / f[0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - c[1]) / f[1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def pixel2cam_matrix(pixel_coord, intrinsic_param):

    x = (pixel_coord[:, 0] - intrinsic_param[0][2]) / intrinsic_param[0][0] * pixel_coord[:, 2]
    y = (pixel_coord[:, 1] - intrinsic_param[1][2]) / intrinsic_param[1][1] * pixel_coord[:, 2]
    z = pixel_coord[:, 2]
    cam_coord = np.concatenate((x[:, None], y[:, None], z[:, None]), 1)
    return cam_coord


def get_intrinsic_metrix(f, c, inv=False):
    intrinsic_metrix = np.zeros((3, 3)).astype(np.float32)

    if inv:
        intrinsic_metrix[0, 0] = 1.0 / f[0]
        intrinsic_metrix[0, 2] = -c[0] / f[0]
        intrinsic_metrix[1, 1] = 1.0 / f[1]
        intrinsic_metrix[1, 2] = -c[1] / f[1]
        intrinsic_metrix[2, 2] = 1
    else:
        intrinsic_metrix[0, 0] = f[0]
        intrinsic_metrix[0, 2] = c[0]
        intrinsic_metrix[1, 1] = f[1]
        intrinsic_metrix[1, 2] = c[1]
        intrinsic_metrix[2, 2] = 1

    return intrinsic_metrix
