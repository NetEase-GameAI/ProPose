import copy
import os

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch.utils.data as data

from propose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from propose.utils.wrapper import SMPL3DCamWrapper
from .convert import generate_extra_jts

class PW3D(data.Dataset):
    """ 3DPW dataset. """

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
    joints_name_24 = (
        'pelvis', 'left_hip', 'right_hip',      # 2
        'spine1', 'left_knee', 'right_knee',    # 5
        'spine2', 'left_ankle', 'right_ankle',  # 8
        'spine3', 'left_foot', 'right_foot',    # 11
        'neck', 'left_collar', 'right_collar',  # 14
        'jaw',                                  # 15
        'left_shoulder', 'right_shoulder',      # 17
        'left_elbow', 'right_elbow',            # 19
        'left_wrist', 'right_wrist',            # 21
        'left_thumb', 'right_thumb'             # 23
    )

    def __init__(self, cfg, ann_file, root='./data/pw3d', train=True, lazy_import=False):

        self._cfg = cfg
        self._ann_file = os.path.join(root, 'json', ann_file)
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

        self.root_idx_17 = self.joints_name_17.index('Pelvis')
        self.root_idx_smpl = self.joints_name_24.index('pelvis')
        assert self.root_idx_17 == 0
        assert self.root_idx_smpl == 0

        self._items, self._labels = self._lazy_load_json()

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

    def _lazy_load_json(self):
        print('Lazy load 3DPW ...')

        items, labels = [], []
        db = COCO(self._ann_file)
        cnt = 0
        for aid in db.anns.keys():
            ann = db.anns[aid]

            img_id = ann['image_id']

            img = db.loadImgs(img_id)[0]
            width, height = img['width'], img['height']

            sequence_name = img['sequence']
            img_name = img['file_name']
            abs_path = os.path.join(self._root, 'imageFiles', sequence_name, img_name)

            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(ann['bbox']), width, height)
            if xmin > xmax - 5 or ymin > ymax - 5:
                continue

            f = np.array(img['cam_param']['focal'], dtype=np.float32)
            c = np.array(img['cam_param']['princpt'], dtype=np.float32)

            joint_cam_17 = 1000 * np.array(ann['h36m_joints'], dtype=np.float32).reshape(17, 3)
            joint_vis_17 = np.ones((17, 3))
            joint_relative_17 = joint_cam_17 - joint_cam_17[self.root_idx_17, :]

            joint_cam = np.array(ann['smpl_joint_cam']) * 1000  # m -> mm
            if joint_cam.size == 24 * 3:
                joint_cam_29 = np.zeros((29, 3))
                joint_cam_29[:24, :] = joint_cam.reshape(24, 3)
            else:
                joint_cam_29 = joint_cam.reshape(29, 3)

            joint_vis_24 = np.ones((24, 3))
            joint_vis_29 = np.zeros((29, 3))
            joint_vis_29[:24, :] = joint_vis_24

            joint_img = np.array(ann['smpl_joint_img'], dtype=np.float32).reshape(24, 3)
            if joint_img.size == 24 * 3:
                joint_img_29 = np.zeros((29, 3))
                joint_img_29[:24, :] = joint_img.reshape(24, 3)
            else:
                joint_img_29 = joint_img.reshape(29, 3)
            joint_img_29[:, 2] = joint_img_29[:, 2] - joint_img_29[[self.root_idx_smpl], 2]
            joint_img_29[:, 2] *= 1000

            root_cam = joint_cam_29[self.root_idx_smpl]

            beta = np.array(ann['smpl_param']['shape']).reshape(10)
            theta = np.array(ann['smpl_param']['pose']).reshape(24, 3)

            # joint_img_11 = generate_extra_jts(beta, theta, joint_cam_29[0:1], f, c)

            items.append(abs_path)
            
            labels.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'img_id': cnt,
                'img_path': abs_path,
                'img_name': img_name,
                'width': width,
                'height': height,
                'joint_cam_17': joint_cam_17,
                'joint_vis_17': joint_vis_17,
                'joint_relative_17': joint_relative_17,
                'joint_cam_29': joint_cam_29,
                'joint_vis_29': joint_vis_29,
                'joint_img_29': joint_img_29,
                # 'joint_img_11': joint_img_11,
                'beta': beta,
                'theta': theta,
                'root_cam': root_cam,
                'f': f,
                'c': c
            })
            cnt += 1

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