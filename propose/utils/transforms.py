import cv2
import numpy as np
import torch
from torch.nn import functional as F


#----------------------------- Transform format -----------------------------#
def im_to_torch(img):
    """ Transform ndarray image to torch tensor.

    Args:
    - img: numpy.ndarray with shape: `(H, W, 3)`.

    Returns:
    - torch.Tensor with shape: `(3, H, W)`.

    """
    img = np.transpose(img, (2, 0, 1))  # C*H*W
    img = to_torch(img).float()
    if img.max() > 1:
        img /= 255
    return img


def to_numpy(tensor):
    # torch.Tensor => numpy.ndarray
    if torch.is_tensor(tensor):
        return tensor.cpu().numpy()
    elif type(tensor).__module__ != 'numpy':
        raise ValueError("Cannot convert {} to numpy array"
                         .format(type(tensor)))
    return tensor


def to_torch(ndarray):
    # numpy.ndarray => torch.Tensor
    if type(ndarray).__module__ == 'numpy':
        return torch.from_numpy(ndarray)
    elif not torch.is_tensor(ndarray):
        raise ValueError("Cannot convert {} to torch tensor"
                         .format(type(ndarray)))
    return ndarray


def get_affine_transform(center, scale, rot, output_size,
                         shift=np.array([0, 0], dtype=np.float32), inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale])

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5]) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])

    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))

    return trans


def affine_transform(pt, t):
    new_pt = np.array([pt[0], pt[1], 1.]).T
    new_pt = np.dot(t, new_pt)
    return new_pt[:2]


def transform_preds(coords, center, scale, output_size):
    target_coords = np.zeros(coords.shape)
    trans = get_affine_transform(center, scale, 0, output_size, inv=1)
    target_coords[0:2] = affine_transform(coords[0:2], trans)
    return target_coords


def get_3rd_point(a, b):
    """ Return vector c that perpendicular to (a - b). """
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    """ Rotate the point by `rot_rad` degree. """
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result


def batch_rodrigues(rot_vecs, epsilon=1e-8, dtype=torch.float32):
    """ Calculates the rotation matrices for a batch of rotation vectors.
    
    Args:
    - rot_vecs: torch.tensor (N, 3), array of N axis-angle vectors
    
    Returns:
    - R: torch.tensor (N, 3, 3), the rotation matrices for the given axis angle.
    """

    batch_size = rot_vecs.shape[0]
    device = rot_vecs.device

    angle = torch.norm(rot_vecs + epsilon, dim=1, keepdim=True)
    rot_dir = rot_vecs / angle

    cos = torch.unsqueeze(torch.cos(angle), dim=1)
    sin = torch.unsqueeze(torch.sin(angle), dim=1)

    # Bx1 arrays
    rx, ry, rz = torch.split(rot_dir, 1, dim=1)
    K = torch.zeros((batch_size, 3, 3), dtype=dtype, device=device)

    zeros = torch.zeros((batch_size, 1), dtype=dtype, device=device)
    K = torch.cat([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], dim=1) \
        .view((batch_size, 3, 3))

    ident = torch.eye(3, dtype=dtype, device=device).unsqueeze(dim=0)
    rot_mat = ident + sin * K + (1 - cos) * torch.bmm(K, K)
    return rot_mat


def batch_rodrigues_numpy(rot_vecs, epsilon=1e-8):
    """ Calculates the rotation matrices for a batch of rotation vectors.
    
    Args:
    - rot_vecs: np.ndarray (N, 3), array of N axis-angle vectors
    
    Returns:
    - R: np.ndarray (N, 3, 3), the rotation matrices for the given axis angle.
    """

    batch_size = rot_vecs.shape[0]

    angle = np.linalg.norm(rot_vecs + epsilon, axis=1, keepdims=True)
    rot_dir = rot_vecs / angle

    cos = np.cos(angle)[:, None, :]
    sin = np.sin(angle)[:, None, :]

    # Bx1 arrays
    rx, ry, rz = np.split(rot_dir, 3, axis=1)
    K = np.zeros((batch_size, 3, 3))
    zeros = np.zeros((batch_size, 1))

    K = np.concatenate([zeros, -rz, ry, rz, zeros, -rx, -ry, rx, zeros], axis=1) \
        .reshape((batch_size, 3, 3))

    ident = np.eye(3)[None, :, :]
    rot_mat = ident + sin * K + (1 - cos) * np.einsum('bij,bjk->bik', K, K)
    return rot_mat


#----------------------------- Flip -----------------------------#
def flip_last(x):
    assert (x.dim() == 3 or x.dim() == 4)
    dim = x.dim() - 1

    return x.flip(dims=(dim,))


def flip_joints_vis(joints_vis, width, joint_pairs):
    """ Flip joints and visibility.

    Args:
    - joints_vis: numpy.ndarray (J, 3, 2), joints and visibility.
    - width: int, image width.
    - joint_pairs: list, list of joint pairs.

    Returns:
    - numpy.ndarray, (J, 3, 2), flipped 3D joints.

    """
    joints = joints_vis.copy()
    # flip horizontally
    joints[:, 0, 0] = width - joints[:, 0, 0] - 1
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :, 0], joints[pair[1], :, 0] = \
            joints[pair[1], :, 0], joints[pair[0], :, 0].copy()
        joints[pair[0], :, 1], joints[pair[1], :, 1] = \
            joints[pair[1], :, 1], joints[pair[0], :, 1].copy()

    joints[:, :, 0] *= joints[:, :, 1]
    return joints


def flip_xyz_joints_3d(joints_3d, joint_pairs):
    """ Flip 3D xyz joints.

    Args:
    - joints_3d: numpy.ndarray (J, 3), joints.
    - joint_pairs: list of joint pairs.

    Returns:
    - numpy.ndarray (J, 3), flipped 3D joints.
    """
    assert joints_3d.ndim in (2, 3)

    joints = joints_3d.copy()
    # flip horizontally
    joints[:, 0] = -1 * joints[:, 0]
    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints


def flip_cam_xyz_joints_3d(joints_3d, joint_pairs):
    """ Flip translated 3D xyz joints.

    Args:
    - joints_3d: numpy.ndarray (J, 3), joints.
    - joint_pairs: list of joint pairs.

    Returns:
    - numpy.ndarray (J, 3), flipped 3D joints.
    """
    root_jts = joints_3d[:1].copy()
    joints = (joints_3d - root_jts)
    assert joints_3d.ndim in (2, 3)

    # flip horizontally
    joints[:, 0] = -1 * joints[:, 0]

    # change left-right parts
    for pair in joint_pairs:
        joints[pair[0], :], joints[pair[1], :] = joints[pair[1], :], joints[pair[0], :].copy()

    return joints + root_jts


def flip_thetas(thetas, theta_pairs):
    """ Flip thetas.

    Args:
    - thetas: numpy.ndarray (num_thetas, 3), thetas in axis angle.
    - theta_pairs: list of theta pairs.

    Returns:
    - (num_thetas, 3), flipped thetas.
    """
    thetas_flip = thetas.copy()
    # reflect horizontally
    thetas_flip[:, 1] = -1 * thetas_flip[:, 1]
    thetas_flip[:, 2] = -1 * thetas_flip[:, 2]
    # change left-right parts
    for pair in theta_pairs:
        thetas_flip[pair[0], :], thetas_flip[pair[1], :] = \
            thetas_flip[pair[1], :], thetas_flip[pair[0], :].copy()

    return thetas_flip


#----------------------------- Rotate -----------------------------#
def rotate_axis_angle(aa, rot):
    """ Rotate axis angle parameters. """
    R = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                  [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                  [0, 0, 1]])
    # find the rotation of the body in camera frame
    per_rdg, _ = cv2.Rodrigues(aa)
    # apply the global rotation to the global orientation
    resrot, _ = cv2.Rodrigues(np.dot(R, per_rdg))
    aa = (resrot.T)[0]
    return aa


def rotate_xyz_jts(xyz_jts, rot):
    assert xyz_jts.ndim == 2 and xyz_jts.shape[1] == 3
    xyz_jts_new = xyz_jts.copy()

    rot_rad = - np.pi * rot / 180

    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    xyz_jts_new[:, 0] = xyz_jts[:, 0] * cs - xyz_jts[:, 1] * sn
    xyz_jts_new[:, 1] = xyz_jts[:, 0] * sn + xyz_jts[:, 1] * cs
    return xyz_jts_new


#----------------------------- Heatmaps -----------------------------#
def norm_heatmap(norm_name, heatmap):
    # Input tensor shape: (B, C, ...)
    assert isinstance(heatmap, torch.Tensor), 'Heatmap to be normalized must be torch.Tensor!'
    shape = heatmap.shape

    if norm_name == 'softmax':
        heatmap = heatmap.reshape(*shape[:2], -1)
        # global soft max
        heatmap = F.softmax(heatmap, 2)
        return heatmap.reshape(*shape)
    elif norm_name == 'sigmoid':
        return heatmap.sigmoid()
    elif norm_name == 'divide_sum':
        heatmap = heatmap.reshape(*shape[:2], -1)
        heatmap = heatmap / heatmap.sum(dim=2, keepdim=True)
        return heatmap.reshape(*shape)
    else:
        raise NotImplementedError


def drawGaussian(img, pt, sigma):
    """Draw 2d gaussian on input image.

    Args:
    - img: torch.Tensor, a tensor with shape: `(3, H, W)`.
    - pt: list or tuple, a point: (x, y).
    - sigma: int, sigma of gaussian distribution.

    Returns:
    - torch.Tensor, a tensor with shape: `(3, H, W)`.
    """
    img = to_numpy(img)
    tmpSize = 3 * sigma
    # Check that any part of the gaussian is in-bounds
    ul = [int(pt[0] - tmpSize), int(pt[1] - tmpSize)]
    br = [int(pt[0] + tmpSize + 1), int(pt[1] + tmpSize + 1)]

    if (ul[0] >= img.shape[1] or ul[1] >= img.shape[0] or br[0] < 0 or br[1] < 0):
        # If not, just return the image as is
        return to_torch(img)

    # Generate gaussian
    size = 2 * tmpSize + 1
    x = np.arange(0, size, 1, float)
    y = x[:, np.newaxis]
    x0 = y0 = size // 2
    # The gaussian is not normalized, we want the center value to equal 1
    g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

    # Usable gaussian range
    g_x = max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]
    g_y = max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]
    # Image range
    img_x = max(0, ul[0]), min(br[0], img.shape[1])
    img_y = max(0, ul[1]), min(br[1], img.shape[0])

    img[img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    return to_torch(img)


def heatmap_to_coord(pred_jts, pred_scores, hm_shape, bbox, output_3d=False, mean_bbox_scale=None):
    # This cause imbalanced GPU usage, implement cpu version
    hm_width, hm_height = hm_shape

    ndims = pred_jts.dim()
    assert ndims in [2, 3], "Dimensions of input heatmap should be 2 or 3"
    if ndims == 2:
        pred_jts = pred_jts.unsqueeze(0)
        pred_scores = pred_scores.unsqueeze(0)

    coords = pred_jts.cpu().numpy()
    coords = coords.astype(float)
    pred_scores = pred_scores.cpu().numpy()
    pred_scores = pred_scores.astype(float)

    coords[:, :, 0] = (coords[:, :, 0] + 0.5) * hm_width
    coords[:, :, 1] = (coords[:, :, 1] + 0.5) * hm_height

    preds = np.zeros_like(coords)
    # transform bbox to scale
    xmin, ymin, xmax, ymax = bbox
    w = xmax - xmin
    h = ymax - ymin
    center = np.array([xmin + w * 0.5, ymin + h * 0.5])
    scale = np.array([w, h])
    # Transform back
    for i in range(coords.shape[0]):
        for j in range(coords.shape[1]):
            preds[i, j, 0:2] = transform_preds(coords[i, j, 0:2], center, scale,
                                               [hm_width, hm_height])
            if output_3d:
                if mean_bbox_scale is not None:
                    zscale = scale[0] / mean_bbox_scale
                    preds[i, j, 2] = coords[i, j, 2] / zscale
                else:
                    preds[i, j, 2] = coords[i, j, 2]
    # maxvals = np.ones((*preds.shape[:2], 1), dtype=float)
    # score_mul = 1 if norm_name == 'sigmoid' else 5

    return preds, pred_scores

