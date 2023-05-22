import random

import cv2
import numpy as np
import torch

from .augmentation import add_occlusion, add_DPG
from ..bbox import bbox_xywh_to_cs, bbox_cs_to_xyxy
from ..camera import get_intrinsic_metrix
from ..transforms import (im_to_torch, affine_transform, get_affine_transform, batch_rodrigues_numpy,
                          flip_joints_vis, flip_xyz_joints_3d, flip_thetas,
                          rotate_xyz_jts, rotate_axis_angle)


class SMPL3DCamWrapper(object):
    def __init__(self, dataset, input_size, output_size, train,
                 scale_factor, color_factor, occlusion, add_dpg, rot, flip,
                 scale_mult=1.25, with_smpl=True, focal_length=1000., root_idx=0):
        """ Data augmentation.
        Occlusion, Truncation(half-body), Scale, Rotation, Flip, Color.
        """

        if not with_smpl:
            self._joint_pairs = dataset.joint_pairs
        else:
            self._joint_pairs_17 = dataset.joint_pairs_17
            self._joint_pairs_24 = dataset.joint_pairs_24
            self._joint_pairs_29 = dataset.joint_pairs_29
            self._joint_pairs_11 = dataset.joint_pairs_11

        self._input_size = input_size
        self._heatmap_size = output_size
        self._train = train

        self._scale_factor = scale_factor
        self._color_factor = color_factor
        self._occlusion = occlusion
        self._add_dpg = add_dpg
        self._rot = rot
        self._flip = flip

        self._scale_mult = scale_mult
        self.with_smpl = with_smpl
        self.focal_length = focal_length
        self.root_idx = root_idx  # ATTN: the root index of mpi_inf is 4 !!!

        self._aspect_ratio = float(input_size[1]) / input_size[0]  # w / h

        self.pixel_std = 1

        self.bbox_3d_shape = dataset.bbox_3d_shape

        if train:
            self.num_joints_half_body = dataset.num_joints_half_body
            self.prob_half_body = dataset.prob_half_body

            self.upper_body_ids = dataset.upper_body_ids
            self.lower_body_ids = dataset.lower_body_ids

    def test_transform(self, src, bbox):
        if isinstance(src, str):
            src = cv2.cvtColor(cv2.imread(src), cv2.COLOR_BGR2RGB)

        xmin, ymin, xmax, ymax = bbox
        center, scale = bbox_xywh_to_cs(
            xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
        scale = scale * 1.0

        inp_h, inp_w = self._input_size
        trans = get_affine_transform(center, scale, 0, [inp_w, inp_h])

        img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
        bbox = bbox_cs_to_xyxy(center, scale)

        img = im_to_torch(img)
        # mean
        img[0].add_(-0.485)
        img[1].add_(-0.456)
        img[2].add_(-0.406)

        # std
        img[0].div_(0.229)
        img[1].div_(0.224)
        img[2].div_(0.225)

        # img_center = np.array([float(src.shape[1]) * 0.5, float(src.shape[0]) * 0.5])

        return img, bbox

    def _uvd_target_generator(self, joints_3d, num_joints, patch_height, patch_width):
        target_weight = joints_3d[:, :, 1].copy()

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0, 0] / patch_width - 0.5
        target[:, 1] = joints_3d[:, 1, 0] / patch_height - 0.5
        target[:, 2] = joints_3d[:, 2, 0] / self.bbox_3d_shape[2]

        target = target.reshape(-1)
        target_weight = target_weight.reshape(-1)
        return target, target_weight

    def _xyz_target_generator(self, joints_3d, joints_3d_vis, num_joints):
        target_weight = joints_3d_vis.copy()

        target = np.zeros((num_joints, 3), dtype=np.float32)
        target[:, 0] = joints_3d[:, 0] / self.bbox_3d_shape[0]
        target[:, 1] = joints_3d[:, 1] / self.bbox_3d_shape[1]
        target[:, 2] = joints_3d[:, 2] / self.bbox_3d_shape[2]

        target = target.reshape(-1)
        target_weight = target_weight.reshape(-1)
        return target, target_weight

    def __call__(self, src, label):
        #============================= Only 3D keypoints =============================#
        if not self.with_smpl:
            bbox = list(label['bbox'])
            imgwidth, imgheight = label['width'], label['height']
            assert imgwidth == src.shape[1] and imgheight == src.shape[0]

            joints_img = label['joint_img'].copy()
            joints_vis = label['joint_vis'].copy()
            joints_cam = label['joint_cam'].copy()

            self.num_joints = joints_img.shape[0]
            gt_joints = np.zeros((self.num_joints, 3, 2), dtype=np.float32)
            gt_joints[:, :, 0] = joints_img
            gt_joints[:, :, 1] = joints_vis

            input_size = self._input_size

            if self._train and self._add_dpg:
                bbox = add_DPG(bbox, imgwidth, imgheight)

            xmin, ymin, xmax, ymax = bbox
            center, scale = bbox_xywh_to_cs(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
            xmin, ymin, xmax, ymax = bbox_cs_to_xyxy(center, scale)

            #----------------------------- Occlusion -----------------------------#
            if self._train and self._occlusion:
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
            joints_cam = joints_cam.reshape(-1 , 3)
            joints_rel = joints_cam - joints_cam[[self.root_idx]] # the root index of mpi_inf is 4 !!!

            if self._train and self._flip and random.random() > 0.5:
                assert src.shape[2] == 3
                src = src[:, ::-1, :]

                joints = flip_joints_vis(joints, imgwidth, self._joint_pairs)
                joints_rel = flip_xyz_joints_3d(joints_rel, self._joint_pairs)
                center[0] = imgwidth - center[0] - 1

            #----------------------------- Transform image -----------------------------#
            inp_h, inp_w = input_size
            trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
            img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
            bbox = bbox_cs_to_xyxy(center, scale)

            #----------------------------- Transform joints -----------------------------#
            # Transform 2D joints
            for i in range(self.num_joints):
                if joints[i, 0, 1] > 0.0:
                    joints[i, 0:2, 0] = affine_transform(joints[i, 0:2, 0], trans)

            # Rotate 3D joints
            joints_rel = rotate_xyz_jts(joints_rel, r)

            #----------------------------- Target generation -----------------------------#
            target_uvd, target_weight = self._uvd_target_generator(joints, self.num_joints, inp_h, inp_w)
            target_xyz, target_xyz_weight = self._xyz_target_generator(joints_rel, joints_vis, self.num_joints)
            target_weight *= joints_vis.reshape(-1)

            #----------------------------- Others -----------------------------#
            trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)

            intrinsic_param = get_intrinsic_metrix(label['f'], label['c']).astype(np.float32) if 'f' in label.keys() \
                else np.array([[self.focal_length, 0., imgwidth*0.5],
                               [0., self.focal_length, imgheight*0.5],
                               [0., 0., 1.]]).astype(np.float32)
            
            joint_root = 0.001 * label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() \
                else np.array([0., 0., 3.]).astype(np.float32)
            
        #============================= With SMPL annotations =============================#
        else:
            bbox = list(label['bbox'])
            imgwidth, imgheight = label['width'], label['height']
            assert imgwidth == src.shape[1] and imgheight == src.shape[0]

            joints_vis_17 = label['joint_vis_17'].copy()
            joints_cam_17 = label['joint_cam_17'].copy()

            joints_img_29 = label['joint_img_29'].copy()
            joints_vis_29 = label['joint_vis_29'].copy()
            joints_cam_29 = label['joint_cam_29'].copy()
            if 'joint_vis_xyz' in label:
                # For EFT-COCO only, where joints_vis_29(coco label) does not have depth visibility.
                joints_vis_xyz = label['joint_vis_xyz'].copy()
            else:
                joints_vis_xyz = None
            
            if 'joint_img_11' in label:
                joints_11_uvd = label['joint_img_11'].copy()
            else:
                joints_11_uvd = np.zeros((11, 3, 2), dtype=np.float32)
            if 'joint_cam_11' in label:
                joints_cam_11 = label['joint_cam_11']
                joints_vis_11 = np.ones((11, 3), dtype=np.float32)
            else:
                joints_cam_11 = np.zeros((11, 3), dtype=np.float32)
                joints_vis_11 = np.zeros((11, 3), dtype=np.float32)

            gt_joints_29 = np.zeros((29, 3, 2), dtype=np.float32)
            gt_joints_29[:, :, 0] = joints_img_29.copy()
            gt_joints_29[:, :, 1] = joints_vis_29.copy()
            
            beta = label['beta'].copy()
            theta = label['theta'].copy()

            input_size = self._input_size

            if self._train and self._add_dpg:
                bbox = add_DPG(bbox, imgwidth, imgheight)
            
            xmin, ymin, xmax, ymax = bbox
            center, scale = bbox_xywh_to_cs(
                xmin, ymin, xmax - xmin, ymax - ymin, self._aspect_ratio, scale_mult=self._scale_mult)
            xmin, ymin, xmax, ymax = bbox_cs_to_xyxy(center, scale)

            #----------------------------- Occlusion -----------------------------#
            if self._train and self._occlusion:
                add_occlusion(src, xmin, xmax, ymin, ymax, imgwidth, imgheight)
                
            #----------------------------- Half body transform -----------------------------#
            if self._train and (np.sum(joints_vis_29[:, 0]) > self.num_joints_half_body \
                           and np.random.rand() < self.prob_half_body):
                
                c_half_body, s_half_body = self.half_body_transform(joints_img_29, joints_vis_29)
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
            joints_29_uvd = gt_joints_29
            joints_rel_17_xyz = joints_cam_17 - joints_cam_17[[self.root_idx]]
            joints_rel_24_xyz = joints_cam_29[:24] - joints_cam_29[[self.root_idx]]
            joints_rel_11_xyz = joints_cam_11 - joints_cam_29[[self.root_idx]]

            if self._train and self._flip and random.random() > 0.5:
                assert src.shape[2] == 3
                src = src[:, ::-1, :]

                joints_29_uvd = flip_joints_vis(joints_29_uvd, imgwidth, self._joint_pairs_29)
                joints_11_uvd = flip_joints_vis(joints_11_uvd, imgwidth, self._joint_pairs_11)
                joints_rel_17_xyz = flip_xyz_joints_3d(joints_rel_17_xyz, self._joint_pairs_17)
                joints_rel_24_xyz = flip_xyz_joints_3d(joints_rel_24_xyz, self._joint_pairs_24)
                joints_rel_11_xyz = flip_xyz_joints_3d(joints_rel_11_xyz, self._joint_pairs_11)
                theta = flip_thetas(theta, self._joint_pairs_24)
                center[0] = imgwidth - center[0] - 1

            #----------------------------- Transform image -----------------------------#
            inp_h, inp_w = input_size
            trans = get_affine_transform(center, scale, r, [inp_w, inp_h])
            img = cv2.warpAffine(src, trans, (int(inp_w), int(inp_h)), flags=cv2.INTER_LINEAR)
            bbox = bbox_cs_to_xyxy(center, scale)
            
            #----------------------------- Transform joints -----------------------------#
            # Transform 2D joints
            for i in range(joints_29_uvd.shape[0]):
                if joints_29_uvd[i, 0, 1] > 0.0:
                    joints_29_uvd[i, 0:2, 0] = affine_transform(joints_29_uvd[i, 0:2, 0], trans)
            for i in range(joints_11_uvd.shape[0]):
                if joints_11_uvd[i, 0, 1] > 0.0:
                    joints_11_uvd[i, 0:2, 0] = affine_transform(joints_11_uvd[i, 0:2, 0], trans)
    
            # Rotate global orientation
            theta[0, :3] = rotate_axis_angle(theta[0, :3], r)
            theta_rot_mat = batch_rodrigues_numpy(theta).reshape(24 * 9)
            theta_24_weights = np.ones((24 * 9))

            # Rotate 3D joints
            joints_rel_17_xyz = rotate_xyz_jts(joints_rel_17_xyz, r)
            joints_rel_24_xyz = rotate_xyz_jts(joints_rel_24_xyz, r)
            joints_rel_11_xyz = rotate_xyz_jts(joints_rel_11_xyz, r)

            #----------------------------- Target generation -----------------------------#
            target_uvd_29, target_weight_29 = self._uvd_target_generator(joints_29_uvd, 29, inp_h, inp_w)
            target_weight_29 *= joints_vis_29.reshape(-1)
            target_uvd_11, target_weight_11 = self._uvd_target_generator(joints_11_uvd, 11, inp_h, inp_w)
            target_xyz_17, target_weight_17 = self._xyz_target_generator(joints_rel_17_xyz, joints_vis_17, 17)
            target_weight_17 *= joints_vis_17.reshape(-1)
            if joints_vis_xyz is None:
                target_xyz_24, target_weight_24 = self._xyz_target_generator(joints_rel_24_xyz, joints_vis_29[:24, :], 24)
                target_weight_24 *= joints_vis_29[:24, :].reshape(-1)
            else:
                target_xyz_24, target_weight_24 = self._xyz_target_generator(joints_rel_24_xyz, joints_vis_xyz[:24, :], 24)
                target_weight_24 *= joints_vis_xyz[:24, :].reshape(-1)
            target_xyz_11, target_weight_11_xyz = self._xyz_target_generator(joints_rel_11_xyz, joints_vis_11, 11)
            target_weight_11_xyz *= joints_vis_11.reshape(-1)

            #----------------------------- Others -----------------------------#
            trans_inv = get_affine_transform(center, scale, r, [inp_w, inp_h], inv=True).astype(np.float32)

            intrinsic_param = get_intrinsic_metrix(label['f'], label['c']).astype(np.float32) if 'f' in label.keys() \
                else np.array([[self.focal_length, 0., imgwidth*0.5],
                               [0., self.focal_length, imgheight*0.5], 
                               [0., 0., 1.]]).astype(np.float32)
            
            joint_root = 0.001 * label['root_cam'].astype(np.float32) if 'root_cam' in label.keys() \
                else np.array([0., 0., 3.]).astype(np.float32)

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

        if 'beta_kid' not in label:
            kid_betas = np.array([0], dtype=np.float32)
        else:
            kid_betas = label['beta_kid'].copy()
        
        if not self.with_smpl:
            output = {
                'type': '3d_skel',
                'image': img,
                'target_uvd': torch.from_numpy(target_uvd).float(),
                'target_weight': torch.from_numpy(target_weight).float(),
                'target_xyz': torch.from_numpy(target_xyz).float(),
                'target_xyz_weight': torch.from_numpy(target_xyz_weight).float(),
                'bbox': torch.Tensor(bbox).float(),
                'trans_inv': torch.from_numpy(trans_inv).float(),
                'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
                'joint_root': torch.from_numpy(joint_root).float(),
                'img_center': torch.from_numpy(img_center).float(),
                'kid_betas' : torch.from_numpy(kid_betas).float(),
            }

        else:
            output = {
                'type': '3d_smpl',
                'image': img,
                'target_theta': torch.from_numpy(theta_rot_mat).float(),
                'target_theta_weight': torch.from_numpy(theta_24_weights).float(),
                'target_beta': torch.from_numpy(beta).float(),
                'target_smpl_weight': torch.ones(1).float(),
                'target_uvd_29': torch.from_numpy(target_uvd_29).float(),
                'target_weight_29': torch.from_numpy(target_weight_29).float(),
                'target_uvd_11': torch.from_numpy(target_uvd_11).float(),
                'target_weight_11': torch.from_numpy(target_weight_11).float(),
                'target_xyz_24': torch.from_numpy(target_xyz_24).float(),
                'target_weight_24': torch.from_numpy(target_weight_24).float(),
                'target_xyz_17': torch.from_numpy(target_xyz_17).float(),
                'target_weight_17': torch.from_numpy(target_weight_17).float(),
                'target_xyz_11': torch.from_numpy(target_xyz_11).float(),
                'target_weight_11_xyz': torch.from_numpy(target_weight_11_xyz).float(),
                'bbox': torch.Tensor(bbox).float(),
                'trans_inv': torch.from_numpy(trans_inv).float(),
                'intrinsic_param': torch.from_numpy(intrinsic_param).float(),
                'joint_root': torch.from_numpy(joint_root).float(),
                'img_center': torch.from_numpy(img_center).float(),
                'kid_betas' : torch.from_numpy(kid_betas).float(),
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