from easydict import EasyDict as edict
import numpy as np
import pickle as pk
import torch
import torch.nn as nn

from .lbs import lbs, propose_core


def to_tensor(array, dtype=torch.float32):
    if 'torch.tensor' not in str(type(array)):
        return torch.tensor(array, dtype=dtype)

def to_np(array, dtype=np.float32):
    if 'scipy.sparse' in str(type(array)):
        array = array.todense()
    return np.array(array, dtype=dtype)


class SMPL_layer(nn.Module):
    NUM_BETAS = 10
    ROOT_IDX_SMPL = 0
    ROOT_IDX_X = 0

    def __init__(self, model_path, X_regressor=None, dtype=torch.float32, use_kid=True):
        super(SMPL_layer, self).__init__()

        with open(model_path, 'rb') as smpl_file:
            self.smpl_data = edict(**pk.load(smpl_file, encoding='latin1'))

        self.dtype = dtype
        self.faces = self.smpl_data.f

        """ Register Buffer """
        # Faces
        self.register_buffer('faces_tensor', to_tensor(to_np(self.smpl_data.f, dtype=np.int64), dtype=torch.long))

        # (6890, 3)
        self.register_buffer('v_template', to_tensor(to_np(self.smpl_data.v_template), dtype=dtype))

        # (6890, 3, 10)
        self.register_buffer('shapedirs', to_tensor(to_np(self.smpl_data.shapedirs), dtype=dtype))
        
        if use_kid:
            v_template_smil = np.load('./model_files/smpl/smpl_kid_template.npy')
            v_template_smil -= np.mean(v_template_smil, axis=0)
            v_template_diff = np.expand_dims(v_template_smil - to_np(self.smpl_data.v_template), axis=2)
            kid_shapedirs = np.concatenate(
                (to_np(self.smpl_data.shapedirs)[:, :, :self.NUM_BETAS], v_template_diff), axis=2)
            self.register_buffer('shapedirs', to_tensor(to_np(kid_shapedirs), dtype=dtype))

        # (6890, 3, 23*9) -> (23*9, 6890*3)
        num_pose_basis = self.smpl_data.posedirs.shape[-1]        
        posedirs = self.smpl_data.posedirs.reshape(-1, num_pose_basis).T
        self.register_buffer('posedirs', to_tensor(to_np(posedirs), dtype=dtype))

        # (6890, 24)
        self.register_buffer('lbs_weights', to_tensor(to_np(self.smpl_data.weights), dtype=dtype))

        # (24, 6890)
        self.register_buffer('J_regressor', to_tensor(to_np(self.smpl_data.J_regressor), dtype=dtype))

        # (X, 6890)
        if X_regressor is not None:
            self.register_buffer('X_regressor', to_tensor(to_np(X_regressor), dtype=dtype))
        else:
            self.X_regressor = None

        # indices of parents for each joints
        parents = torch.zeros((35), dtype=torch.long)
        parents[:24] = to_tensor(to_np(self.smpl_data.kintree_table[0])).long()
        parents[0] = -1
        parents[24:35] = torch.tensor([15, 15, 15, 15, 15, 7, 7, 7, 8, 8, 8], dtype=torch.long)
        self.register_buffer('parents', parents)

    def forward(self, poses, betas, global_orient=None, transl=None, kid_betas=None, pose2rot=False):
        if global_orient is not None:
            full_pose = torch.cat([global_orient, poses], dim=1)
        else:
            full_pose = poses

        verts, joints, rot_mats, joints_X = lbs(
            betas, full_pose, self.v_template, 
            self.shapedirs, self.posedirs, self.lbs_weights, self.J_regressor, self.parents, 
            pose2rot=pose2rot, X_regressor=self.X_regressor, dtype=self.dtype,
            kid_betas=kid_betas)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            verts += transl.unsqueeze(dim=1)
            if joints_X is not None:
                joints_X += transl.unsqueeze(dim=1)
        else:
            # verts = verts - joints_X[:, [self.ROOT_IDX_X], :].detach()
            verts = verts - joints[:, [self.ROOT_IDX_X], :].detach()
            joints = joints - joints[:, [self.ROOT_IDX_SMPL], :].detach()
            if joints_X is not None:
                joints_X = joints_X - joints_X[:, [self.ROOT_IDX_X], :].detach()

        output = edict(
            vertices=verts, 
            joints=joints, 
            rot_mats=rot_mats, 
            X_from_verts=joints_X)

        return output

    def propose(self, betas, joints_3d, mf_params, transl=None, 
                kappas=None, kid_betas=None, use_sample=False):

        verts, joints, rot_mats, joints_X, post_mf_params = propose_core(
            betas, joints_3d, mf_params,
            self.v_template, self.shapedirs, self.posedirs, self.lbs_weights, self.J_regressor, self.parents,
            X_regressor=self.X_regressor, dtype=self.dtype, train=self.training, 
            kappas=kappas, kid_betas=kid_betas, use_sample=use_sample)

        if transl is not None:
            joints += transl.unsqueeze(dim=1)
            verts += transl.unsqueeze(dim=1)
            if joints_X is not None:
                joints_X += transl.unsqueeze(dim=1)
        else:
            # ATTN: depend on the alignment strategy during evaluation
            # verts = verts - joints_X[:, [self.ROOT_IDX_X], :].detach()
            verts = verts - joints[:, [self.ROOT_IDX_SMPL], :].detach()
            joints = joints - joints[:, [self.ROOT_IDX_SMPL], :].detach()
            if joints_X is not None:
                joints_X = joints_X - joints_X[:, [self.ROOT_IDX_X], :].detach()

        output = edict(
            vertices=verts, 
            joints=joints, 
            rot_mats=rot_mats, 
            X_from_verts=joints_X, 
            post_mf_params=post_mf_params)

        return output
    