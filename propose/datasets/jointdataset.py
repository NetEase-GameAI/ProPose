import bisect
import random

import torch
import torch.utils.data as data

from .h36m import H36m
from .mscoco import Mscoco
from .hp3d import HP3D
from .pw3d import PW3D
from .eftcoco import EFTCOCO
from .agora import AGORA
from .convert import s_coco_2_smpl_jt, s_3dhp_2_smpl_jt


class JointDatasetCam(data.Dataset):
    data_domain = set([
        'type',
        'target_theta', 'target_theta_weight', 'target_beta', 'target_smpl_weight',
        'target_uvd_29', 'target_weight_29',
        'target_xyz_24', 'target_weight_24',
        'target_xyz_17', 'target_weight_17',
        'target_uvd_11', 'target_weight_11', 
        'target_xyz_11', 'target_weight_11_xyz', 
        'trans_inv', 'intrinsic_param', 'joint_root',
        'img_center', 'kid_betas',
    ])

    def __init__(self, cfg, train=True, lazy_import=True):

        self._train = train
        
        if train:
            all_datasets = ['h36m', 'coco', 'hp3d', '3dpw', 'eft', 'agora']
            all_partition = [0.3,    0.1,    0.1,    0.2,    0.2,    0.1]
            
            used_datasets, used_partition = [], []
            for i, prob in enumerate(all_partition):
                if prob > 1e-5:
                    used_datasets.append(all_datasets[i])
                    used_partition.append(prob)
            self.used_datasets = used_datasets

            self.used_subsets = []
            if 'h36m' in used_datasets:
                db0 = H36m(cfg=cfg, ann_file='Sample_5_train_Human36M_smpl_leaf_twist', 
                           train=True, lazy_import=lazy_import)
                self.used_subsets.append(db0)
           
            if 'coco' in used_datasets:
                db1 = Mscoco(cfg=cfg, ann_file='coco_wholebody_train_v1.0.json', # person_keypoints_train2017.json
                             train=True, lazy_import=lazy_import)
                self.used_subsets.append(db1)

            if 'hp3d' in used_datasets:
                db2 = HP3D(cfg=cfg, ann_file='annotation_mpi_inf_3dhp_train_v2.json',
                           train=True, lazy_import=lazy_import)
                self.used_subsets.append(db2)
            
            if '3dpw' in used_datasets:
                db3 = PW3D(cfg=cfg, ann_file='3DPW_train_new.json', 
                           train=True, lazy_import=lazy_import)
                self.used_subsets.append(db3)

            if 'eft' in used_datasets:
                db4 = EFTCOCO(cfg=cfg, ann_file='NEW_COCO2014-All-ver01.pkl',
                              train=True, lazy_import=lazy_import)
                self.used_subsets.append(db4)

            if 'agora' in used_datasets:
                db5 = AGORA(cfg=cfg, ann_file='train_all_SMPL_withjv_withkid.pt',
                            train=True, lazy_import=lazy_import)
                self.used_subsets.append(db5)

            self.subset_size = [len(item) for item in self.used_subsets]
            print('dataset: ', used_datasets)
            print('partition: ', used_partition)
            print('subset size: ', self.subset_size)
            assert len(used_partition) == len(self.subset_size)
            self.max_db_data_num = max(self.subset_size)
            self.tot_size = int(1.0 * max(self.subset_size))
        else:
            self.used_datasets = ['h36m']
            db0 = H36m(cfg=cfg, ann_file='Sample_20_test_Human36M_smpl',
                       train=train, lazy_import=lazy_import)
            self.used_subsets = [db0]
            self.tot_size = len(db0)
            used_partition = [1]

        self.cumulative_sizes = self.cumsum(used_partition)

    @staticmethod
    def cumsum(sequence):
        r, s = [], 0
        for e in sequence:
            r.append(e + s)
            s += e
        return r

    def __len__(self):
        return self.tot_size

    def __getitem__(self, idx):
        assert idx >= 0
        if self._train:
            p = random.uniform(0, 1)

            dataset_idx = bisect.bisect_right(self.cumulative_sizes, p)

            _db_len = self.subset_size[dataset_idx]

            # last batch: random sampling
            if idx >= _db_len * (self.tot_size // _db_len):
                sample_idx = random.randint(0, _db_len - 1)
            else:  # before last batch: use modular
                sample_idx = idx % _db_len
        else:
            dataset_idx = 0
            sample_idx = idx

        # Load subset data
        img, target, img_path, bbox = self.used_subsets[dataset_idx][sample_idx]

        if self.used_datasets[dataset_idx] in ['coco', 'hp3d']:
            # COCO, 3DHP
            label_jts_origin = target.pop('target_uvd')
            label_jts_mask_origin = target.pop('target_weight')

            label_uvd_29 = torch.zeros(29, 3)
            label_uvd_29_mask = torch.zeros(29, 3)
            label_xyz_24 = torch.zeros(24, 3)
            label_xyz_29_mask = torch.zeros(29, 3)
            label_xyz_17 = torch.zeros(17, 3)
            label_xyz_17_mask = torch.zeros(17, 3)
            label_uvd_11 = torch.zeros(11, 3)
            label_uvd_11_mask = torch.zeros(11, 3)
            label_xyz_11 = torch.zeros(11, 3)
            label_xyz_11_mask = torch.zeros(11, 3)

            # COCO
            if self.used_datasets[dataset_idx] == 'coco':
                label_jts_origin = label_jts_origin.reshape(-1, 2)
                label_jts_mask_origin = label_jts_mask_origin.reshape(-1, 2)

                for i in range(24):
                    id1 = i
                    id2 = s_coco_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :2] = label_jts_origin[id2, :2].clone()
                        label_uvd_29_mask[id1, :2] = label_jts_mask_origin[id2, :2].clone()
                
                # original face joints
                label_uvd_11[:5, :2] = label_jts_origin[[0, 2, 1, 4, 3], :2].clone()
                label_uvd_11_mask[:5, :2] = label_jts_mask_origin[[0, 2, 1, 4, 3], :2].clone()
                # foot joints from coco whole-body
                label_uvd_11[5:, :2] = label_jts_origin[-6:, :2].clone()
                label_uvd_11_mask[5:, :2] = label_jts_mask_origin[-6:, :2].clone()

            # 3DHP
            elif self.used_datasets[dataset_idx] == 'hp3d':
                assert label_jts_origin.dim() == 1 and label_jts_origin.shape[0] == 28 * 3, label_jts_origin.shape

                label_xyz_origin = target.pop('target_xyz').reshape(-1, 3)
                label_xyz_mask_origin = target.pop('target_xyz_weight').reshape(-1, 3)

                label_jts_origin = label_jts_origin.reshape(28, 3)
                label_jts_mask_origin = label_jts_mask_origin.reshape(28, 3)

                for i in range(24):
                    id1 = i
                    id2 = s_3dhp_2_smpl_jt[i]
                    if id2 >= 0:
                        label_uvd_29[id1, :3] = label_jts_origin[id2, :3].clone()
                        label_uvd_29_mask[id1, :3] = label_jts_mask_origin[id2, :3].clone()
                        label_xyz_24[id1, :3] = label_xyz_origin[id2, :3].clone()
                        label_xyz_29_mask[id1, :3] = label_xyz_mask_origin[id2, :3].clone()

            label_uvd_29 = label_uvd_29.reshape(-1)
            label_uvd_29_mask = label_uvd_29_mask.reshape(-1)
            label_xyz_24 = label_xyz_24.reshape(-1)
            label_xyz_24_mask = label_xyz_29_mask[:24, :].reshape(-1)
            label_xyz_17 = label_xyz_17.reshape(-1)
            label_xyz_17_mask = label_xyz_17_mask.reshape(-1)
            label_uvd_11 = label_uvd_11.reshape(-1)
            label_uvd_11_mask = label_uvd_11_mask.reshape(-1)
            label_xyz_11 = label_xyz_11.reshape(-1)
            label_xyz_11_mask = label_xyz_11_mask.reshape(-1)

            target['target_theta'] = torch.zeros(24 * 9)
            target['target_theta_weight'] = torch.zeros(24 * 9)
            target['target_beta'] = torch.zeros(10)
            target['target_smpl_weight'] = torch.zeros(1)

            target['target_uvd_29'] = label_uvd_29
            target['target_weight_29'] = label_uvd_29_mask
            target['target_xyz_24'] = label_xyz_24
            target['target_weight_24'] = label_xyz_24_mask
            target['target_xyz_17'] = label_xyz_17
            target['target_weight_17'] = label_xyz_17_mask
            target['target_uvd_11'] = label_uvd_11
            target['target_weight_11'] = label_uvd_11_mask
            target['target_xyz_11'] = label_xyz_11
            target['target_weight_11_xyz'] = label_xyz_11_mask

        else:
            assert set(target.keys()).issubset(self.data_domain), \
            (set(target.keys()) - self.data_domain, self.data_domain - set(target.keys()),)
        
        target.pop('type')

        if 'kid_betas' not in target.keys():
            target['kid_betas'] = torch.zeros(1).float()

        return img, target, img_path, bbox