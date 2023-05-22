import copy
import os
import pickle as pk

import cv2
import numpy as np
from pycocotools.coco import COCO
import torch.utils.data as data

from propose.utils.bbox import bbox_clip_xyxy, bbox_xywh_to_xyxy
from propose.utils.wrapper import Skel2DCamWrapper


class Mscoco(data.Dataset):
    """ COCO Person dataset. """

    CLASSES = ['person']
    joints_name = ('nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',    # 4
                   'left_shoulder', 'right_shoulder',                           # 6
                   'left_elbow', 'right_elbow',                                 # 8
                   'left_wrist', 'right_wrist',                                 # 10
                   'left_hip', 'right_hip',                                     # 12
                   'left_knee', 'right_knee',                                   # 14
                   'left_ankle', 'right_ankle',                                 # 16
                   'left_big_toe', 'left_small_toe', 'left_heel',               # 19 (whole body)
                   'right_big_toe', 'right_small_toe', 'right_heel')            # 22

    def __init__(self, cfg, ann_file, root='./data/coco', train=True, skip_empty=True, lazy_import=False):

        self._cfg = cfg
        self._ann_file = os.path.join(root, 'annotations', ann_file)
        self._root = root
        self._train = train
        self._skip_empty = skip_empty
        self._lazy_import = lazy_import
        self.num_joints = 23 if 'wholebody' in ann_file else 17

        self.bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))

        self._input_size = cfg.MODEL.IMAGE_SIZE
        self._output_size = cfg.MODEL.HEATMAP_SIZE

        self._scale_factor = cfg.DATASET.SCALE_FACTOR
        self._color_factor = cfg.DATASET.COLOR_FACTOR
        self._occlusion = cfg.DATASET.OCCLUSION
        self._rot = cfg.DATASET.ROT_FACTOR
        self._flip = cfg.DATASET.FLIP
        self._sigma = cfg.MODEL.EXTRA.SIGMA
        self._dpg = cfg.DATASET.DPG

        self.num_joints_half_body = cfg.DATASET.NUM_JOINTS_HALF_BODY
        self.prob_half_body = cfg.DATASET.PROB_HALF_BODY

        self.upper_body_ids = (0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10)
        self.lower_body_ids = (11, 12, 13, 14, 15, 16)

        self._check_centers = False

        self.num_class = len(self.CLASSES)

        self._items, self._labels = self._lazy_load_json()

        self.transformation = Skel2DCamWrapper(
            self, input_size=self._input_size, output_size=self._output_size, train=self._train,
            scale_factor=self._scale_factor, color_factor=self._color_factor, occlusion=self._occlusion,
            add_dpg=self._dpg, rot=self._rot, flip=self._flip, sigma=self._sigma, scale_mult=1.25)

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
            print('Lazy load COCO ...')
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

        _coco = self._lazy_load_ann_file()
        classes = [c['name'] for c in _coco.loadCats(_coco.getCatIds())]
        assert classes == self.CLASSES, "Incompatible category names with COCO. "

        self.json_id_to_contiguous = {
            v: k for k, v in enumerate(_coco.getCatIds())}

        # iterate through the annotations
        image_ids = sorted(_coco.getImgIds())
        for entry in _coco.loadImgs(image_ids):
            dirname, filename = entry['coco_url'].split('/')[-2:]
            abs_path = os.path.join(self._root, dirname, filename)
            if not os.path.exists(abs_path):
                raise IOError('Image: {} not exists.'.format(abs_path))
            label = self._check_load_keypoints(_coco, entry)
            if not label:
                continue

            # num of items are relative to person, not image
            for obj in label:
                items.append(abs_path)
                labels.append(obj)

        return items, labels

    def _lazy_load_ann_file(self):
        if os.path.exists(self._ann_file + '.pkl') and self._lazy_import:
            print('Lazy load json...')
            with open(self._ann_file + '.pkl', 'rb') as fid:
                return pk.load(fid)
        else:
            _coco = COCO(self._ann_file)
            try:
                with open(self._ann_file + '.pkl', 'wb') as fid:
                    pk.dump(_coco, fid, pk.HIGHEST_PROTOCOL)
            except Exception as e:
                print(e)
                print('Skip writing to .pkl file.')
            return _coco
        
    def _check_load_keypoints(self, coco, entry):
        """Check and load ground-truth keypoints"""
        ann_ids = coco.getAnnIds(imgIds=entry['id'], iscrowd=False)
        objs = coco.loadAnns(ann_ids)
        # check valid bboxes
        valid_objs = []
        width = entry['width']
        height = entry['height']

        for obj in objs:
            contiguous_cid = self.json_id_to_contiguous[obj['category_id']]
            if contiguous_cid >= self.num_class:
                # not class of interest
                continue
            if max(obj['keypoints']) == 0:
                continue
            # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax) and clip bound
            xmin, ymin, xmax, ymax = bbox_clip_xyxy(bbox_xywh_to_xyxy(obj['bbox']), width, height)
            # require non-zero box area
            if obj['area'] <= 0 or xmax <= xmin or ymax <= ymin:
                continue
            if obj['num_keypoints'] == 0:
                continue

            # joints 3d: (num_joints, 3, 2); 3 is for x, y, z; 2 is for position, visibility
            joints_3d = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            annot_kpts = obj['keypoints']

            if self.num_joints == 23:
                annot_kpts += obj['foot_kpts']
            for i in range(self.num_joints):
                joints_3d[i, 0, 0] = annot_kpts[i * 3 + 0]
                joints_3d[i, 1, 0] = annot_kpts[i * 3 + 1]
                visible = min(1, annot_kpts[i * 3 + 2])
                joints_3d[i, :2, 1] = visible

            if np.sum(joints_3d[:, 0, 1]) < 1:
                # no visible keypoint
                continue

            if self._check_centers and self._train:
                bbox_center, bbox_area = self._get_box_center_area((xmin, ymin, xmax, ymax))
                kp_center, num_vis = self._get_keypoints_center_count(joints_3d)
                ks = np.exp(-2 * np.sum(np.square(bbox_center - kp_center)) / bbox_area)
                if (num_vis / 80.0 + 47 / 80.0) > ks:
                    continue

            valid_objs.append({
                'bbox': (xmin, ymin, xmax, ymax),
                'width': width,
                'height': height,
                'joints_3d': joints_3d,
            })

        if not valid_objs:
            if not self._skip_empty:
                # dummy invalid labels if no valid objects are found
                valid_objs.append({
                    'bbox': np.array([-1, -1, 0, 0]),
                    'width': width,
                    'height': height,
                    'joints_3d': np.zeros((self.num_joints, 2, 2), dtype=np.float32)
                })
        return valid_objs

    @property
    def joint_pairs(self):
        if self.num_joints == 23:
            return [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16],
                    [17, 20], [18, 21], [19, 22]]
        else: # 17
            return [[1, 2], [3, 4], [5, 6], [7, 8], [9, 10], [11, 12], [13, 14], [15, 16]]

    def _get_box_center_area(self, bbox):
        """ Get bbox center. """
        c = np.array([(bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0])
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        return c, area

    def _get_keypoints_center_count(self, keypoints):
        """ Get geometric center of all keypoints. """
        keypoint_x = np.sum(keypoints[:, 0, 0] * (keypoints[:, 0, 1] > 0))
        keypoint_y = np.sum(keypoints[:, 1, 0] * (keypoints[:, 1, 1] > 0))
        num = float(np.sum(keypoints[:, 0, 1]))
        return np.array([keypoint_x / num, keypoint_y / num]), num
