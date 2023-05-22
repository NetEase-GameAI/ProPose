import copy
import os
import pickle as pk

import cv2
import numpy as np
import torch.utils.data as data

from propose.datasets.convert import s_coco_2_smpl_jt
from propose.utils.wrapper import SMPL3DCamWrapper
from .convert import generate_extra_jts


class EFTCOCO(data.Dataset):
    """ EFT-COCO dataset. """

    def __init__(self, cfg, ann_file, root='./data/eft', train=True, lazy_import=False):

        self._cfg = cfg
        self._ann_file = os.path.join(root, 'eft_fit/new_post', ann_file)
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

        self._items, self._labels = self._lazy_load_json()

        self.transformation = SMPL3DCamWrapper(
            self, input_size=self._input_size, output_size=self._output_size, train=self._train,
            scale_factor=self._scale_factor, color_factor=self._color_factor, occlusion=self._occlusion,
            add_dpg=self._dpg, rot=self._rot, flip=self._flip, scale_mult=1.25, with_smpl=True)

    def __getitem__(self, idx):
        img_path = self._items[idx]

        label = copy.deepcopy(self._labels[idx])

        # For EFT, extra 3D joints from SMPL vertices are not accurate enough.
        # _, joint_cam_11 \
            # = generate_extra_jts(label['beta'], label['theta'], label['joint_cam_29'][0:1], label['f'], label['c'])
        # label['joint_cam_11'] = joint_cam_11

        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        target = self.transformation(orig_img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        return img, target, img_path, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file):
            print('Lazy load EFT ...')
            with open(self._ann_file, 'rb') as fid:
                items, labels = pk.load(fid)
            
            new_labels = []
            for cnt, label in enumerate(labels):
                bbox = label['bbox']
                joint_img_29 = np.zeros((29, 3), dtype=np.float32)
                joint_vis_29 = np.zeros((29, 3), dtype=np.float32)
                joint_img_coco_17 = label['joint_img_17'].copy()
                for i in range(24):
                    id1 = i
                    id2 = s_coco_2_smpl_jt[i]
                    if id2 >= 0:
                        joint_img_29[id1, :2] = joint_img_coco_17[id2, :2, 0].copy()
                        joint_vis_29[id1, :2] = joint_img_coco_17[id2, :2, 1].copy()

                joint_cam_29 = label['joints_29'].copy() * 1000.
                joint_cam_17 = label['joints_17'].copy() * 1000.

                root_cam = label['root_cam'].reshape(-1) * 1000.
                beta = label['beta']
                theta = label['theta']
                f = label['f']
                c = [(bbox[0]+bbox[2])/2, (bbox[1]+bbox[3])/2]  # label['c']

                joint_img_11 = np.zeros((11, 3, 2))
                # COCO order to [Nose, REye, LEye, REar, LEar]
                joint_img_11[:5] = joint_img_coco_17[[0, 2, 1, 4, 3]]

                new_label = {
                    'bbox': bbox,
                    'img_id': cnt,
                    'img_path': os.path.join(self._root, label['img_path']),
                    'width': label['width'],
                    'height': label['height'],
                    'joint_cam_17': joint_cam_17,
                    'joint_vis_17': np.ones((17, 3), dtype=np.float32),
                    'joint_cam_29': joint_cam_29,
                    'joint_vis_xyz': np.ones((29, 3), dtype=np.float32),
                    'joint_vis_29': joint_vis_29,
                    'joint_img_29': joint_img_29,
                    'joint_img_11': joint_img_11,
                    'beta': beta,
                    'theta': theta,
                    'root_cam': root_cam,
                    'f': f,
                    'c': c
                }

                new_labels.append(new_label)

        return items, new_labels

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