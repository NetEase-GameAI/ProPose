import copy
import json
import os
import pickle as pk

import cv2
import numpy as np
import torch.utils.data as data

from propose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from propose.utils.camera import cam2pixel
from propose.utils.wrapper import SMPL3DCamWrapper
from .convert import generate_extra_jts


class H36m(data.Dataset):
    """ Human3.6M smpl dataset. """

    EVAL_JOINTS = [6, 5, 4, 1, 2, 3, 16, 15, 14, 11, 12, 13, 8, 10]

    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )
    joints_name_29 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb',            # 23
        'head', 'left_middle', 'right_middle',  # 26
        'left_bigtoe', 'right_bigtoe'           # 28
    )

    action_name = ['Directions', 'Discussion', 'Eating', 'Greeting', 'Phoning', 'Posing', 'Purchases',
                   'Sitting', 'SittingDown', 'Smoking', 'Photo', 'Waiting', 'Walking', 'WalkDog', 'WalkTogether']

    block_list = ['s_09_act_05_subact_02_ca', 's_09_act_10_subact_02_ca', 's_09_act_13_subact_01_ca']

    def __init__(self, cfg, ann_file, root='./data/h36m', train=True, lazy_import=False):

        self._cfg = cfg
        self.protocol = 2
        self._ann_file = os.path.join(root, 'annotations', ann_file + f'_protocol_{self.protocol}.json')
        self._root = root
        self._train = train
        self._lazy_import = lazy_import

        self._det_bbox_file = None
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

        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.root_idx_smpl = self.joints_name_29.index('pelvis')
        assert self.root_idx_17 == 0
        assert self.root_idx_smpl == 0

        self._items, self._labels = self._lazy_load_json()

        self.transformation = SMPL3DCamWrapper(
            self, input_size=self._input_size, output_size=self._output_size, train=self._train,
            scale_factor=self._scale_factor, color_factor=self._color_factor, occlusion=self._occlusion, 
            add_dpg=self._dpg, rot=self._rot, flip=self._flip, scale_mult=1.1, with_smpl=True)

    def __getitem__(self, idx):
        img_path = self._items[idx]

        label = copy.deepcopy(self._labels[idx])

        joint_img_11, joint_cam_11 \
            = generate_extra_jts(label['beta'], label['theta'], label['joint_cam_29'][0:1], label['f'], label['c'])
        label['joint_img_11'] = joint_img_11
        label['joint_cam_11'] = joint_cam_11

        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        
        target = self.transformation(orig_img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        return img, target, img_path, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_smpl_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load Human3.6M ...')   
            with open(self._ann_file + '_smpl_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_smpl_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')

        return items, labels

    def _load_jsons(self):
        """Load all image paths and labels from JSON annotation files into buffer."""
        items, labels = [], []

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)

        det_bbox_set = {}
        if self._det_bbox_file is not None:
            bbox_list = json.load(open(os.path.join(
                self._root, 'annotations', self._det_bbox_file + f'_protocol_{self.protocol}.json'), 'r'))
            for item in bbox_list:
                image_id = item['image_id']
                det_bbox_set[image_id] = item['bbox']

        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v
            skip = False
            for name in self.block_list:
                if name in ann['file_name']:
                    skip = True
            if skip:
                continue

            image_id = ann['image_id']

            width, height = ann['width'], ann['height']
            if self._det_bbox_file is not None:
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                    bbox_xywh_to_xyxy(det_bbox_set[ann['file_name']]), width, height)
            else:
                xmin, ymin, xmax, ymax = bbox_clip_xyxy(
                    bbox_xywh_to_xyxy(ann['bbox']), width, height)

            f = np.array(ann['cam_param']['f'], dtype=np.float32)
            c = np.array(ann['cam_param']['c'], dtype=np.float32)

            joint_cam_17 = np.array(ann['h36m_joints']).reshape(17, 3)
            joint_vis_17 = np.ones((17, 3))
            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]

            joint_cam = np.array(ann['smpl_joints'])
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)

            joint_vis_29 = np.ones((29, 3))
            joint_img_29 = cam2pixel(joint_cam_29, f, c)
            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_cam_29[self.root_idx_smpl, 2]

            root_cam = np.array(ann['root_coord'])

            beta = np.array(ann['betas'])
            theta = np.array(ann['thetas']).reshape(24, 3)

            abs_path = os.path.join(self._root, 'images', ann['file_name'])

            items.append(abs_path)

            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'img_path': abs_path,
                'width': width,
                'height': height,
                'joint_cam_17': joint_cam_17,
                'joint_vis_17': joint_vis_17,
                'joint_relative_17': joint_relative_17,
                'joint_cam_29': joint_cam_29,
                'joint_vis_29': joint_vis_29,
                'joint_img_29': joint_img_29,
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'f': f,
                'c': c
            })

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