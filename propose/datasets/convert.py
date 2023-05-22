import numpy as np
import torch

from propose.utils.ext_smpl import constants
from propose.utils.ext_smpl.smpl import SMPL


s_coco_2_smpl_jt = [
    -1, 11, 12,
    -1, 13, 14,
    -1, 15, 16,
    -1, -1, -1,
    -1, -1, -1,
    -1,
    5, 6,
    7, 8,
    9, 10,
    -1, -1
]

s_3dhp_2_smpl_jt = [
    4, -1, -1,
    -1, 19, 24,
    -1, 20, 25,
    -1, -1, -1,
    5, -1, -1,
    -1,
    9, 14,
    10, 15,
    11, 16,
    -1, -1,
    # 7, 
    # -1, -1,
    # 21, 26
]

s_mpii_2_smpl_jt = [
    6, 3, 2,
    -1, 4, 1,
    -1, 5, 0,
    -1, -1, -1,
    8, -1, -1,
    -1,
    13, 12,
    14, 11,
    15, 10,
    -1, -1
]


smpl_model = SMPL('model_files/smpl', batch_size=1, create_transl=False)

# smplx.vertex_ids
VERT_MAP = {'Nose': 332, 'REye': 6260, 'LEye':2800, 'REar':4071, 'LEar':583,
            'LBigToe': 3216, 'LSmallToe': 3226, 'LHeel': 3387, 
            'RBigToe': 6617, 'RSmallToe': 6624, 'RHeel': 6787}
EXTRA_VERT_IDS = [*VERT_MAP.values()]


def generate_extra_jts(beta, theta, root_cam, f, c):
    """ Get extra joints from SMPL vertices. """
    
    beta = torch.from_numpy(beta).float().unsqueeze(0)
    body_pose = torch.from_numpy(theta).float().reshape(1, -1)

    output = smpl_model(betas=beta, body_pose=body_pose[:, 3:],
                        global_orient=body_pose[:, :3], pose2rot=True)

    jts_3d = 1000 * output.vertices[:, EXTRA_VERT_IDS].numpy().squeeze(0)
    root_smpl = 1000 * output.joints[:, [constants.JOINT_IDS['OP MidHip']]].numpy().squeeze(0)

    new_jts_3d = jts_3d - root_smpl + root_cam
    
    K = np.eye(3)
    K[0, 0] = f[0]; K[1, 1] = f[1]
    K[0, 2] = c[0]; K[1, 2] = c[1]
    
    jts_2d = new_jts_3d @ K.T
    jts_2d = jts_2d[:, :2] / jts_2d[:, 2:]

    joint_img = np.zeros((jts_2d.shape[0], 3, 2), dtype=np.float32)
    joint_img[:, :2, 0] = jts_2d
    joint_img[:, :2, 1] = 1

    return joint_img, new_jts_3d