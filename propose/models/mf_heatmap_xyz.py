from easydict import EasyDict as edict
import numpy as np
import torch
import torch.nn as nn

from .builder import MODEL_FACTORY
from .layers.hrnet.hrnet import get_hrnet
from .layers.body_models.smpl import SMPL_layer
from propose.utils.transforms import norm_heatmap, flip_last
from propose.datasets.convert import EXTRA_VERT_IDS


def get_branch_mlp(out_dim, num_layers=1, inp_dim=2048, hid_dim=256):
    if num_layers == 1:
        return nn.Linear(inp_dim, out_dim) 
    
    module_list = [nn.Linear(inp_dim, hid_dim)]
    
    for i in range(1, num_layers-1):
        module_list.append(nn.Linear(hid_dim, hid_dim))

    module_list.append(nn.Linear(hid_dim, out_dim))
            
    return nn.Sequential(*module_list)    


@MODEL_FACTORY.register_module
class MFHeatmapXYZ(nn.Module):
    def __init__(self, **kwargs):
        super(MFHeatmapXYZ, self).__init__()
        self.num_joints = kwargs['NUM_JOINTS']
        self.norm_type = kwargs['EXTRA']['NORM_TYPE']
        self.input_size = float(max(kwargs['IMAGE_SIZE']))
        self.height_dim = kwargs['HEATMAP_SIZE'][0]
        self.width_dim = kwargs['HEATMAP_SIZE'][1]
        self.depth_dim = kwargs['DEPTH_DIM']
        self.use_kid = kwargs['USE_KID']

        #----------------------------- SMPL -----------------------------#
        self.smpl_dtype = torch.float32
        self.root_idx_smpl = 0
        self.flip_pairs_35 = [(1, 2), (4, 5), (7, 8), (10, 11), 
                              (13, 14), (16, 17), (18, 19), (20, 21), (22, 23),
                              (25, 26), (27, 28), (29, 32), (30, 33), (31, 34)]
        
        X_regressor = np.load('./model_files/smpl/J_regressor_h36m.npy')
        self.smpl = SMPL_layer(
            './model_files/smpl/SMPL_NEUTRAL.pkl', 
            X_regressor=X_regressor, dtype=self.smpl_dtype, use_kid=self.use_kid
        )

        #----------------------------- Network -----------------------------#   
        self.backbone_type = kwargs['EXTRA']['BACKBONE']

        if self.backbone_type == 'hrnet':
            self.preact = get_hrnet(48, num_joints=self.num_joints-1, depth_dim=self.depth_dim,
                                    is_train=True, generate_feat=True, generate_hm=True)
            feat_dim = 2048
        else:
            raise NotImplementedError

        # shape
        init_shape = np.load('./model_files/smpl/h36m_mean_beta.npy')
        self.register_buffer('init_shape', torch.Tensor(init_shape).float())
        self.shape_num_layers = 1
        self.decshape = get_branch_mlp(out_dim=10, num_layers=self.shape_num_layers, inp_dim=feat_dim)
        
        # kid shape (for SMIL if required)
        if self.use_kid:
            self.kid_num_layers = 1
            self.deckid = get_branch_mlp(out_dim=1, num_layers=self.kid_num_layers, inp_dim=feat_dim)

        # parameter F
        self.F_dim = 24
        self.F_num_layers = 1
        self.decF = get_branch_mlp(out_dim=self.F_dim*9, num_layers=self.F_num_layers, inp_dim=feat_dim)

        # camera
        self.register_buffer('init_cam', torch.Tensor([0.9, 0., 0.]).float())
        self.cam_num_layers = 1
        self.deccam = get_branch_mlp(out_dim=3, num_layers=self.cam_num_layers, inp_dim=feat_dim)

        #----------------------------- Setting -----------------------------# 
        self.focal_length = kwargs['FOCAL_LENGTH']
        
        bbox_3d_shape = kwargs['BBOX_3D_SHAPE'] if 'BBOX_3D_SHAPE' in kwargs else (2000, 2000, 2000)
        assert bbox_3d_shape[0] == bbox_3d_shape[1] == bbox_3d_shape[2]
        self.bbox_3d_shape = torch.Tensor(bbox_3d_shape).float() * 1e-3  # unit: meter, assume cube

        self.depth_factor = self.bbox_3d_shape[2]
        
    def _initialize(self, pretrain=''):
        self.preact.init_weights(pretrain)

    def flip_heatmap(self, heatmaps, shift=True):
        heatmaps = heatmaps.flip(dims=(4,))

        for pair in eval(f'self.flip_pairs_{self.num_joints}'):
            dim0, dim1 = pair
            idx = torch.Tensor((dim0-1, dim1-1)).long()
            inv_idx = torch.Tensor((dim1-1, dim0-1)).long()
            heatmaps[:, idx] = heatmaps[:, inv_idx]

        if shift:
            if heatmaps.dim() == 3:
                heatmaps[:, :, 1:] = heatmaps[:, :, 0:-1]
            elif heatmaps.dim() == 4:
                heatmaps[:, :, :, 1:] = heatmaps[:, :, :, 0:-1]
            else:
                heatmaps[:, :, :, :, 1:] = heatmaps[:, :, :, :, 0:-1]

        return heatmaps

    def forward(self, x, flip_test=False, use_sample=False, **kwargs):
        batch_size = x.shape[0]
        device = x.device

        #----------------------------- Backbone -----------------------------#  
        out, x0 = self.preact(x)    # (B, (J-1)*64, 64, 64), (B, 2048)

        out = out.reshape((out.shape[0], self.num_joints-1, -1))
        heatmaps = norm_heatmap(self.norm_type, out)

        x0 = x0.view(x0.size(0), -1)
        xc = x0

        #----------------------------- MLPs -----------------------------#  
        # shape
        init_shape = self.init_shape.expand(batch_size, -1)
        delta_shape = self.decshape(xc)
        pred_shape = delta_shape + init_shape

        # kid
        if self.use_kid:
            pred_kid = self.deckid(xc)

        # parameter F
        pred_F = self.decF(xc)

        # camera
        init_cam = self.init_cam.expand(batch_size, -1)
        pred_cam = self.deccam(xc).reshape(batch_size, -1) + init_cam

        if not self.training and flip_test:
            flip_x = flip_last(x)
            flip_out, flip_x0 = self.preact(flip_x)
            
            flip_out = flip_out.reshape(batch_size, self.num_joints-1, self.depth_dim, self.height_dim, self.width_dim)
            flip_out = self.flip_heatmap(flip_out).reshape((flip_out.shape[0], self.num_joints-1, -1))
            flip_heatmaps = norm_heatmap(self.norm_type, flip_out)
            heatmaps = (heatmaps + flip_heatmaps) / 2

            flip_xc = flip_x0

            flip_delta_shape = self.decshape(flip_xc)
            flip_pred_shape = flip_delta_shape + init_shape
            pred_shape = (pred_shape + flip_pred_shape) / 2

            if self.use_kid:
                flip_pred_kid = self.deckid(flip_xc)
                pred_kid = (pred_kid + flip_pred_kid) / 2
            
            flip_pred_cam = self.deccam(flip_xc).reshape(batch_size, -1) + init_cam
            flip_pred_cam[:, 1] = -flip_pred_cam[:, 1]
            pred_cam = (pred_cam + flip_pred_cam) / 2

            # TODO: flip_F
        
        #----------------------------- 3D Keypoints -----------------------------# 
        heatmaps = heatmaps.reshape((batch_size, self.num_joints-1, self.depth_dim, self.height_dim, self.width_dim))
        
        hm_x = heatmaps.sum((2, 3))
        hm_y = heatmaps.sum((2, 4))
        hm_z = heatmaps.sum((3, 4))

        range_tensor_x = torch.arange(hm_x.shape[-1], dtype=torch.float32, device=device).unsqueeze(-1)
        range_tensor_y = torch.arange(hm_y.shape[-1], dtype=torch.float32, device=device).unsqueeze(-1)
        range_tensor_z = torch.arange(hm_z.shape[-1], dtype=torch.float32, device=device).unsqueeze(-1)

        coord_x = hm_x.matmul(range_tensor_x)
        coord_y = hm_y.matmul(range_tensor_y)
        coord_z = hm_z.matmul(range_tensor_z)

        coord_x = coord_x / float(self.width_dim) - 0.5
        coord_y = coord_y / float(self.height_dim) - 0.5
        coord_z = coord_z / float(self.depth_dim) - 0.5

        pred_coord_from_hm = torch.cat((coord_x, coord_y, coord_z), dim=2)  # -0.5 ~ 0.5
        
        pred_rel_jts = pred_coord_from_hm * self.depth_factor  # unit: meter
        zero_root_jts = torch.zeros((batch_size, 1, 3), dtype=x.dtype, device=device)
        pred_rel_jts = torch.cat((zero_root_jts, pred_rel_jts), dim=1)

        #----------------------------- Translation -----------------------------#
        cam_depth = 2 * self.focal_length / (256 * pred_cam[:, 0] + 1e-6)
        transl = torch.stack([pred_cam[:, 1], pred_cam[:, 2], cam_depth], dim=-1).unsqueeze(1)
        
        #----------------------------- Posterior results -----------------------------#
        output = self.smpl.propose(
            betas=pred_shape.type(self.smpl_dtype),
            joints_3d=pred_rel_jts.type(self.smpl_dtype),
            mf_params=pred_F.type(self.smpl_dtype),
            kappas=None,
            kid_betas=pred_kid.type(self.smpl_dtype) if self.use_kid else None,
            use_sample=use_sample
        )

        #----------------------------- Output format -----------------------------#
        pred_vertices = output.vertices.float()
        pred_theta_mats = output.rot_mats.float().reshape(-1, 24 * 9)  # (BxS, 216), S is #samples
        
        pred_jts_24_post = output.joints.float()
        pred_jts_24_post = pred_jts_24_post.reshape(-1, 24, 3)
        pred_jts_X_post = output.X_from_verts.float()
        pred_jts_X_post = pred_jts_X_post.reshape(batch_size, -1, *pred_jts_X_post.shape[1:])[:, 0]
        pred_jts_ext_post = pred_vertices[:, EXTRA_VERT_IDS, :]
        pred_jts_24_post = torch.cat((pred_jts_24_post, pred_jts_ext_post), dim=1)
        
        if output.post_mf_params is not None:
            post_mf_params = output.post_mf_params.float()  # (B, 24, 3, 3)

        #----------------------------- Projected 2D keypoints -----------------------------#
        K = torch.zeros((batch_size, 3, 3), dtype=x.dtype, device=device)
        K[:, 0, 0] = K[:, 1, 1] = self.focal_length
        K[:, 2, 2] = 1

        # projection from observed 3D joints
        hm_abs_jts = pred_rel_jts + transl
        hm_2d_jts = hm_abs_jts / hm_abs_jts[..., 2:3]
        hm_2d_jts = torch.einsum('bij,bkj->bki', K, hm_2d_jts)  # relative to full image center
        hm_2d_jts = hm_2d_jts[..., :2] / self.input_size  # -0.5 ~ 0.5

        # projection from posterior 3D joints
        if not use_sample:
            pred_abs_jts = pred_jts_24_post + transl
            pred_2d_jts = pred_abs_jts / pred_abs_jts[..., 2:3]
            pred_2d_jts = torch.einsum('bij,bkj->bki', K, pred_2d_jts)
        else:
            pred_abs_jts = pred_jts_24_post.reshape(
                batch_size, -1, *pred_jts_24_post.shape[1:]) + transl.unsqueeze(1)
            pred_2d_jts = pred_abs_jts / pred_abs_jts[..., 2:3]
            pred_2d_jts = torch.einsum('bij,bskj->bski', K, pred_2d_jts)
        pred_2d_jts = pred_2d_jts[..., :2] / self.input_size

        output = edict(
            pred_F=pred_F.reshape(batch_size, self.F_dim, 3, 3),
            pred_shape=pred_shape,
            pred_theta_mats=pred_theta_mats,
            pred_vertices=pred_vertices,
            pred_coord_from_hm=pred_coord_from_hm.reshape(batch_size, -1),
            pred_rel_jts=pred_rel_jts,
            pred_abs_jts=pred_abs_jts,
            hm_2d_jts=hm_2d_jts,
            pred_jts_24_post=pred_jts_24_post[:, :24],
            pred_jts_X_post=pred_jts_X_post,
            pred_2d_jts=pred_2d_jts,
            post_mf_params=post_mf_params,
            transl=transl,
            pred_cam=pred_cam,
            kid_betas=pred_kid if self.use_kid else None
        )
        
        return output


    def forward_gt_theta(self, gt_theta, gt_beta, gt_kid_betas=None):

        output = self.smpl(poses=gt_theta, betas=gt_beta, kid_betas=gt_kid_betas, pose2rot=False)

        return output
