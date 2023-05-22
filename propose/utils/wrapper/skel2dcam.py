import math
import random

import cv2
import numpy as np
import torch

from .augmentation import add_occlusion, add_DPG
from ..bbox import bbox_xywh_to_cs, bbox_cs_to_xyxy
from ..transforms import im_to_torch, affine_transform, get_affine_transform, flip_joints_vis


class Skel2DCamWrapper(object):
    def __init__(self, dataset, input_size, output_size, train,
                 scale_factor, color_factor, occlusion, add_dpg, rot, flip,
                 sigma, scale_mult=1.25, focal_length=1000.):
        
        self._joint_pairs = dataset.joint_pairs

        self._input_size = input_size
        self._heatmap_size = output_size
        self._train = train

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self._add_dpg = add_dpg
        self._rot = rot
        self._flip = flip

        self.scale_mult = scale_mult
        self.focal_length = focal_length
        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h
        self._feat_stride = np.array(input_size) / np.array(output_size)
        self._sigma = sigma

        self.pixel_std = 1

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids

    def _uvd_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = np.ones((num_joints, 2), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target_weight[:, 1] = joints_3d[:, 0, 1]

        target = np.zeros((num_joints, 2), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5

        target = target.reshape((-1))
        target_weight = target_weight.reshape((-1))
        return target, target_weight

    def _gaussian_target_generator(self, joints_3d, num_joints):
        target_weight = np.ones((num_joints, 1), dtype=np.float32)
        target_weight[:, 0] = joints_3d[:, 0, 1]
        target = np.zeros((num_joints, self._heatmap_size[0], self._heatmap_size[1]), dtype=np.float32)
        tmp_size = self._sigma * 3

        for i in range(num_joints):
            mu_x = int(joints_3d[i, 0, 0] / self._feat_stride[0] + 0.5)
            mu_y = int(joints_3d[i, 1, 0] / self._feat_stride[1] + 0.5)
            # check if any part of the gaussian is in-bounds
            ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
            br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
            if (ul[0] >= self._heatmap_size[1] or ul[1] >= self._heatmap_size[0] or br[0] < 0 or br[1] < 0):
                target_weight[i] = 0
                continue

            # generate gaussian
            size = 2 * tmp_size + 1
            x = np.arange(0, size, 1, np.float32)
            y = x[:, np.newaxis]
            x0 = y0 = size // 2
            # the gaussian is not normalized, we want the center value to be equal to 1
            g = np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (2 * (self._sigma ** 2)))

            # usable gaussian range
            g_x = max(0, -ul[0]), min(br[0], self._heatmap_size[1]) - ul[0]
            g_y = max(0, -ul[1]), min(br[1], self._heatmap_size[0]) - ul[1]
            # image range
            img_x = max(0, ul[0]), min(br[0], self._heatmap_size[1])
            img_y = max(0, ul[1]), min(br[1], self._heatmap_size[0])

            v = target_weight[i]
            if v > 0.5:
                target[i, img_y[0]:img_y[1], img_x[0]:img_x[1]] = g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

        return target, np.expand_dims(target_weight, -1)

    def __call__(self, src, label):
        if label['bbox'] is not None:
            bbox = list(label['bbox'])
        else:
            bbox = None
        imgwidth, imgheight = label['width'], label['height']
        assert imgwidth == src.shape[1] and imgheight == src.shape[0]

        gt_joints = label['joints_3d']  # (J, 3, 2)
        self.num_joints = gt_joints.shape[0]

        joints_vis = np.zeros((self.num_joints, 1), dtype=np.float32)
        joints_vis[:, 0] = gt_joints[:, 0, 1]

        input_size = self._input_size

        if bbox is not None:
            if self._train and self._add_dpg:
                bbox = add_DPG(bbox, imgwidth, imgheight)

            xmin, ymin, xmax, ymax = bbox
            center, scale = bbox_xywh_to_cs(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, self.scale_mult)
            xmin, ymin, xmax, ymax = bbox_cs_to_xyxy(center, scale)
        else:
            center = np.array([imgwidth / 2, imgheight / 2], dtype=np.float32)
            scale = np.array([imgwidth, imgheight], dtype=np.float32)
        
        # occlusion
        if self._train and self._occlusion and bbox is not None:
            add_occlusion(src, xmin, xmax, ymin, ymax, imgwidth, imgheight)

        #----------------------------- Half body transform -----------------------------#
        if self._train and (np.sum(joints_vis[:, 0]) > self.num_joints_half_body \
                       and np.random.rand() < self.prob_half_body):
            
            c_half_body, s_half_body = self.half_body_transform(gt_joints[:, :, 0], joints_vis)
            if c_half_body is not None and s_half_body is not None:
                center, scale = c_half_body, s_half_body

        #----------------------------- Scale -----------------------------#
        if self._train and self._scale_factor > 0:
            sf = self._scale_factor
            scale = scale * np.clip(np.random.randn() * sf + 1, 1 - sf, 1 + sf)
        else:
            scale = scale * 1.0

        #----------------------------- Rotation -----------------------------#
        if self._train and self._rot > 0:
            rf = self._rot
            r = np.clip(np.random.randn() * rf, -rf * 2, rf * 2) if random.random() <= 0.6 else 0
        else:
            r = 0

        #----------------------------- Flip -----------------------------#
        joints = gt_joints

        if self._train and self._flip and random.random() > 0.5:
            assert src.shape[2] == 3
            src = src[:, ::-1, :]

            joints = flip_joints_vis(joints, imgwidth, self._joint_pairs)
            center[0] = imgwidth - center[0] - 1

        #----------------------------- Transform image -----------------------------#
        inp_h, inp_w = input_size
        trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = bbox_cs_to_xyxy(center, scale)

        #----------------------------- Transform joints -----------------------------#
        for i in range(self.num_joints):
            if joints[i, 0, 1] > 0.0:
                joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

        #----------------------------- Target generation -----------------------------#
        if True:
            target_uvd, target_weight = self._uvd_target_generator(joints, self.num_joints, inp_h, inp_w)
        else: # gaussian
            target_uvd, target_weight = self._gaussian_target_generator(joints, self.num_joints)

        #----------------------------- Others -----------------------------#
        trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)

        intrinsic_param = np.array([[self.focal_length, 0., imgwidth*0.5],
                                    [0., self.focal_length, imgheight*0.5], 
                                    [0., 0., 1.]]).astype(np.float32)
        
        joint_root = np.array([0., 0., 3.]).astype(np.float32)

        assert img.shape[2] == 3
        #----------------------------- Color augmentation -----------------------------#
        if self._train:
            c_high = 1 + self._color_factor
            c_low = 1 - self._color_factor
            img[:, :, 0] = np.clip(img[:, :, 0] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 1] = np.clip(img[:, :, 1] * random.uniform(c_low, c_high), 0, 255)
            img[:, :, 2] = np.clip(img[:, :, 2] * random.uniform(c_low, c_high), 0, 255)

        #----------------------------- Convert image to tensor -----------------------------#
        img = im_to_torch(img)

        # mean
        img[0].add_(-0.485)
        img[1].add_(-0.456)
        img[2].add_(-0.406)

        # std
        img[0].div_(0.229)
        img[1].div_(0.224)
        img[2].div_(0.225)

        img_center = np.array([float(imgwidth) * 0.5, float(imgheight) * 0.5])

        output = {
            'type': '2d_skel',
            'image': img,
            'target_uvd': torch.from_numpy(target_uvd).float(),
            'target_weight': torch.from_numpy(target_weight).float(),
            'bbox': torch.Tensor(bbox).float(),
            'trans_inv': torch.from_numpy(trans_inv).float(),
            'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
            'joint_root': torch.from_numpy(joint_root).float(),
            'img_center': torch.from_numpy(img_center).float(),
        }
        return output

    def half_body_transform(self, joints, joints_vis):
        upper_joints = []
        lower_joints = []
        for joint_id in range(joints.shape[0]):
            if joints_vis[joint_id][0] > 0:
                if joint_id in self.upper_body_ids:
                    upper_joints.append(joints[joint_id])
                else:
                    lower_joints.append(joints[joint_id])

        if np.random.randn() < 0.5 and len(upper_joints) > 2:
            selected_joints = upper_joints
        else:
            selected_joints = lower_joints if len(lower_joints) > 2 else upper_joints

        if len(selected_joints) < 2:
            return None, None

        selected_joints = np.array(selected_joints, dtype=np.float32)
        center = selected_joints.mean(axis=0)[:2]

        left_top = np.amin(selected_joints, axis=0)
        right_bottom = np.amax(selected_joints, axis=0)

        w = right_bottom[0] - left_top[0]
        h = right_bottom[1] - left_top[1]

        if w > self._aspect_ratio * h:
            h = w * 1.0 / self._aspect_ratio
        elif w < self._aspect_ratio * h:
            w = h * self._aspect_ratio

        scale = np.array([w * 1.0 / self.pixel_std,
                          h * 1.0 / self.pixel_std], dtype=np.float32)

        scale = scale * 1.5

        return center, scale