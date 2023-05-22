import copy
import os

import cv2
import joblib
import numpy as np
import torch.utils.data as data

from propose.utils.wrapper import SMPL3DCamWrapper
from .convert import generate_extra_jts


class AGORA(data.Dataset):
    """ AGORA dataset. """

    def __init__(self, cfg, ann_file, root='./data/agora', train=True, lazy_import=False):
        
        self._cfg = cfg
        self._ann_file = os.path.join(root, 'annots', ann_file)
        self._root = root
        self._train = train
        self._lazy_import = lazy_import

        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))

        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._occlusion = cfg.DATASET.OCCLUSION
        self._rot = cfg.DATASET.ROT_FACTOR
        self._flip = cfg.DATASET.FLIP
        self._dpg = cfg.DATASET.DPG

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.upper_body_ids = (6, 9, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23)
        self.lower_body_ids = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11)

        self.root_idx_17 = 0
        self.root_idx_smpl = 0

        self.use_kid = cfg.MODEL.USE_KID

        self.db = joblib.load(self._ann_file, None)
        self._items, self._labels = self._lazy_load_pt(self.db)

        self.transformation = SMPL3DCamWrapper(
            self, input_size=self._input_size, output_size=self._output_size, train=self._train,
            scale_factor=self._scale_factor, color_factor=self._color_factor, occlusion=self._occlusion, 
            add_dpg=self._dpg, rot=self._rot, flip=self._flip, scale_mult=1.25, with_smpl=True)

    def __getitem__(self, idx):
        img_path = self._items[idx]

        label = copy.deepcopy(self._labels[idx])

        joint_img_11, joint_cam_11 = \
            generate_extra_jts(label['beta'], label['theta'], label['joint_cam_29'][0:1], label['f'], label['c'])
        label['joint_img_11'] = joint_img_11
        label['joint_cam_11'] = joint_cam_11

        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        target = self.transformation(orig_img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        return img, target, img_path, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_pt(self, db):
        print('Lazy load AOGRA ...')

        items, labels = [], []
        db_len = len(db['ann_path'])

        for k, v in db.items():
            assert len(v) == db_len, k
        
        img_cnt = 0
        for idx in range(db_len):
            if self._train:
                is_valid = db['is_valid'][idx]
                if not is_valid: continue
                occlusion = db['occlusion'][idx]
                if occlusion > 50: continue  # remove severely occluded person

            beta_kid = np.array(db['shape_kid'][idx])
            if not self.use_kid and beta_kid.item() != 0:
                continue

            img_name = db['img_path'][idx]
            ann_path = db['ann_path'][idx]
            ann_file = ann_path.split('/')[-1]
            ann_file = ann_file.split('_')
            if 'train' in img_name:
                img_parent_path = os.path.join(self._root, 'images', f'{ann_file[0]}_{ann_file[1]}')
            else:
                img_parent_path = os.path.join(self._root, 'images', 'validation')

            img_path = os.path.join(img_parent_path, img_name)

            beta = np.array(db['shape'][idx]).reshape(10)
            theta = np.array(db['pose'][idx]).reshape(24, 3)

            joint_rel_17 = db['xyz_17'][idx].reshape(17, 3) * 1000.
            joint_vis_17 = np.ones((17, 3))

            joint_rel_17 = joint_rel_17 - joint_rel_17[0, :]

            joint_cam_24 = db['gt_joints_3d'][idx].reshape(-1, 3)[:24] * 1000.
            joint_cam_29 = db['xyz_29'][idx].reshape(29, 3) * 1000.
            joint_cam_29 += joint_cam_24[0:1]

            joint_2d_full = db['uv_24'][idx].reshape(-1, 2)
            joint_2d = joint_2d_full[:24]
            joint_img_29 = np.zeros_like(joint_cam_29)
            joint_img_29[:, 2] = joint_cam_29[:, 2]
            joint_img_29[:24, :2] = joint_2d

            joint_vis_24 = np.ones((24, 3))
            joint_vis_29 = np.zeros((29, 3))
            joint_vis_29[:24, :] = joint_vis_24

            root_cam = joint_cam_24[0]
            
            left, right, upper, lower = \
                joint_2d_full[:, 0].min(), joint_2d_full[:, 0].max(), joint_2d_full[:, 1].min(), joint_2d_full[:, 1].max()

            center = np.array([(left+right)*0.5, (upper+lower)*0.5], dtype=np.float32)
            scale = [right-left, lower-upper]

            scale = float(max(scale))

            scale = scale * 1.25

            xmin, ymin, xmax, ymax = center[0] - scale*0.5, center[1] - scale*0.5, center[0] + scale*0.5, center[1] + scale*0.5

            if not(xmin < 1280-5 and ymin < 720-5 and xmax > 5 and ymax > 5):
                continue
            
            width = 1280
            height = 720
            K_mat, fl_mm = get_cam_info(img_name, height, width)

            items.append(img_path)
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': img_cnt,
                'img_path': img_path,
                'img_name': img_name,
                'width': width,
                'height': height,
                'joint_cam_17': joint_rel_17.copy(),
                'joint_vis_17': joint_vis_17.copy(),
                'joint_cam_29': joint_cam_29.copy(),
                'joint_vis_29': joint_vis_29.copy(),
                'joint_img_29': joint_img_29.copy(),
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'f': [K_mat[0][0], K_mat[1][1]],
                'c': [K_mat[0][2], K_mat[1][2]],
                'beta_kid': beta_kid,
            })
            
            img_cnt += 1

        return items, labels

    @property
    def joint_pairs_17(self):
        return ((1, 4), (2, 5), (3, 6), (11, 14), (12, 15), (13, 16))

    @property
    def joint_pairs_24(self):
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23))

    @property
    def joint_pairs_29(self):
        return ((1, 2), (4, 5), (7, 8), (10, 11), (13, 14), (16, 17), (18, 19), (20, 21), (22, 23), (25, 26), (27, 28))
    
    @property
    def joint_pairs_11(self):
        return ((1, 2), (3, 4), (5, 8), (6, 9), (7, 10))

def focalLength_mm2px(focalLength, dslr_sens, focalPoint):
    focal_pixel = (focalLength / dslr_sens) * focalPoint * 2
    return focal_pixel

def get_cam_info(imgPath, imgHeight, imgWidth):
    dslr_sens_width = 36
    dslr_sens_height = 20.25
    cx = imgWidth / 2
    cy = imgHeight / 2

    # unofficial
    if 'hdri' in imgPath: focalLength = 50
    elif 'cam00' in imgPath: focalLength = 18
    elif 'cam01' in imgPath: focalLength = 18
    elif 'cam02' in imgPath: focalLength = 18
    elif 'cam03' in imgPath: focalLength = 18
    elif 'ag2' in imgPath: focalLength = 28
    else: focalLength = 28
    
    focalLength_x = focalLength_mm2px(focalLength, dslr_sens_width, cx)
    focalLength_y = focalLength_mm2px(focalLength, dslr_sens_height, cy)

    intrinsic_mat = np.array([[focalLength_x, 0, cx],
                              [0, focalLength_y, cy],
                              [0, 0, 1]])
    return intrinsic_mat, focalLength