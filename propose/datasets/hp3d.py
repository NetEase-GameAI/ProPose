import copy
import json
import os
import pickle as pk

import cv2
import numpy as np
import torch.utils.data as data

from propose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from propose.utils.wrapper import SMPL3DCamWrapper
from propose.utils.camera import cam2pixel_matrix


class HP3D(data.Dataset):
    """ MPI-INF-3DHP dataset. """
    
    EVAL_JOINTS = [i - 1 for i in [8, 6, 15, 16, 17, 10, 11, 12, 24, 25, 26, 19, 20, 21, 5, 4, 7]]
    joints_name_17 = (
        'Pelvis',                               # 0
        'L_Hip', 'L_Knee', 'L_Ankle',           # 3
        'R_Hip', 'R_Knee', 'R_Ankle',           # 6
        'Torso', 'Neck',                        # 8
        'Nose', 'Head',                         # 10
        'L_Shoulder', 'L_Elbow', 'L_Wrist',     # 13
        'R_Shoulder', 'R_Elbow', 'R_Wrist',     # 16
    )

    joints_name = ('spine3', 'spine4', 'spine2', 'spine', 'pelvis',                         # 4
                   'neck', 'head', 'head_top',                                              # 7
                   'left_clavicle', 'left_shoulder', 'left_elbow',                          # 10
                   'left_wrist', 'left_hand', 'right_clavicle',                             # 13
                   'right_shoulder', 'right_elbow', 'right_wrist',                          # 16
                   'right_hand', 'left_hip', 'left_knee',                                   # 19
                   'left_ankle', 'left_foot', 'left_toe',                                   # 22
                   'right_hip', 'right_knee', 'right_ankle', 'right_foot', 'right_toe')     # 27

    test_seqs = (1, 2, 3, 4, 5, 6)
    joint_groups = {'Head': [0], 'Neck': [1], 'Shou': [2, 5], 'Elbow': [3, 6], 'Wrist': [4, 7], 'Hip': [8, 11], 'Knee': [9, 12], 'Ankle': [10, 13]}
    # activity_name full name: ('Standing/Walking','Exercising','Sitting','Reaching/Crouching','On The Floor','Sports','Miscellaneous')
    activity_name = ('Stand', 'Exe', 'Sit', 'Reach', 'Floor', 'Sports', 'Miscell')
    pck_thres = 150
    auc_thres = list(range(0, 155, 5))

    def __init__(self, cfg, ann_file, root='./data/3dhp', train=True, lazy_import=False):
        
        self._cfg = cfg
        self._ann_file = os.path.join(root, ann_file)
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

        self.upper_body_ids = (0, 1, 2, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17) # spine2 is not sure
        self.lower_body_ids = (3, 4, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27)

        # ATTN: root_idx = 4 !!
        self.root_idx = self.joints_name.index('pelvis')
        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        
        self._items, self._labels = self._lazy_load_json()

        self.transformation = SMPL3DCamWrapper(
            self, input_size=self._input_size, output_size=self._output_size, train=self._train,
            scale_factor=self._scale_factor, color_factor=self._color_factor, occlusion=self._occlusion,
            add_dpg=self._dpg, rot=self._rot, flip=self._flip, scale_mult=1.25, with_smpl=False, root_idx=self.root_idx)

    def __getitem__(self, idx):
        img_path = self._items[idx]

        label = copy.deepcopy(self._labels[idx])

        orig_img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)

        target = self.transformation(orig_img, label)

        img = target.pop('image')
        bbox = target.pop('bbox')

        return img, target, img_path, bbox

    def __len__(self):
        return len(self._items)

    def _lazy_load_json(self):
        if os.path.exists(self._ann_file + '_annot_keypoint.pkl') and self._lazy_import:
            print('Lazy load MPI-INF ...')
            with open(self._ann_file + '_annot_keypoint.pkl', 'rb') as fid:
                items, labels = pk.load(fid)
        else:
            items, labels = self._load_jsons()
            try:
                with open(self._ann_file + '_annot_keypoint.pkl', 'wb') as fid:
                    pk.dump((items, labels), fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')

        return items, labels

    def _load_jsons(self):
        items, labels = [], []

        with open(self._ann_file, 'r') as fid:
            database = json.load(fid)

        for ann_image, ann_annotations in zip(database['images'], database['annotations']):
            ann = dict()
            for k, v in ann_image.items():
                assert k not in ann.keys()
                ann[k] = v
            for k, v in ann_annotations.items():
                ann[k] = v

            image_id = ann['image_id']

            width, height = ann['width'], ann['height']
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann['bbox']), width, height)

            intrinsic_param = np.array(ann['cam_param']['intrinsic_param'], dtype=np.float32)

            f = np.array([intrinsic_param[0, 0], intrinsic_param[1, 1]], dtype=np.float32)
            c = np.array([intrinsic_param[0, 2], intrinsic_param[1, 2]], dtype=np.float32)

            joint_cam = np.array(ann['keypoints_cam'])

            joint_vis = np.ones((len(self.joints_name), 3))
            joint_img = cam2pixel_matrix(joint_cam, intrinsic_param)
            joint_img[:, 2] = joint_img[:, 2] - joint_cam[self.root_idx, 2]

            root_cam = joint_cam[self.root_idx]

            abs_path = os.path.join(self._root, 'mpi_inf_3dhp_{}_set'.format('train' if self._train else 'test'), ann['file_name'])

            items.append(abs_path)

            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': image_id,
                'img_path': abs_path,
                'img_name': ann['file_name'],
                'width': width,
                'height': height,
                'joint_cam': joint_cam,
                'joint_vis': joint_vis,
                'joint_img': joint_img,
                'root_cam': root_cam,
                'f': f,
                'c': c
            })

        return items, labels

    @property
    def joint_pairs(self):
        hp3d_joint_pairs = ((8, 13), (9, 14), (10, 15), (11, 16), (12, 17),
                            (18, 23), (19, 24), (20, 25), (21, 26), (22, 27))
        return hp3d_joint_pairs
