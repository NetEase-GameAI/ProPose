import math

import torch
import torch.nn as nn

from .builder import LOSS_FACTORY


def weighted_l1_loss(input, target, weights, use_avg, scale=1.):
    input = input * scale
    target = target * scale
    out = torch.abs(input - target)
    out = out * weights
    if use_avg and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()

def weighted_l2_loss(input, target, weights, use_avg, tol=-10):
    out = (input - target) ** 2
    if tol > 0:
        out[out<tol] = 0.
    out = out * weights
    if use_avg and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()

def weighted_laplace_loss(input, target, weights, sigma, use_avg):
    amp = 1 / math.sqrt(2 * math.pi)
    out = torch.log(sigma / amp) + torch.abs(input - target) / (math.sqrt(2) * sigma + 1e-5)
    out = out * weights
    if use_avg and weights.sum() > 0:
        return out.sum() / weights.sum()
    else:
        return out.sum()


@LOSS_FACTORY.register_module
class LossSMPLCam(nn.Module):
    def __init__(self, ELEMENTS, use_avg=True):
        super(LossSMPLCam, self).__init__()
        self.elements = ELEMENTS
        self.use_avg = use_avg

        self.xyz_hm_weight = self.elements['XYZ_HM_WEIGHT']
        self.uv_weight = self.elements['UV_WEIGHT']
        self.beta_weight = self.elements['BETA_WEIGHT']
        self.theta_weight = self.elements['THETA_WEIGHT']

        self.reg_weight = self.elements['REG_WEIGHT']
        self.vertice_weight = self.elements['VERTICE_WEIGHT']

    def forward(self, output, labels):
        batch_size = output.pred_shape.shape[0]
        smpl_weight = labels['target_smpl_weight']

        loss = 0.
        loss_dict = {}

        #----------------------------- Integral heatmap (metric space) -----------------------------#
        pred_hm = output.pred_coord_from_hm
        if 'target_xyz_11' in labels:
            target_hm = torch.cat((labels['target_xyz_24'][:, 3:72], labels['target_xyz_11']), dim=1)
            target_hm_weight = torch.cat((labels['target_weight_24'][:, 3:72], 
                                          labels['target_weight_11_xyz']), dim=1)
        else:
            target_hm = labels['target_xyz_24'][:, 3:72]
            target_hm_weight = labels['target_weight_24'][:, 3:72]
        
        loss_hm = weighted_l1_loss(pred_hm, target_hm, 
                                   target_hm_weight, self.use_avg, scale=64.)
        loss_hm *= self.xyz_hm_weight
        loss += loss_hm
        if self.xyz_hm_weight > 0:
            loss_dict['hm'] = loss_hm
        
        #----------------------------- Shape -----------------------------#
        tgt_beta_weight = smpl_weight.repeat(1, 10)
        loss_beta = weighted_l2_loss(output.pred_shape, labels['target_beta'], 
                                     tgt_beta_weight, self.use_avg)
        if output.kid_betas is not None:
            loss_beta += 5 * weighted_l2_loss(output.kid_betas, labels['kid_betas'], 
                                              tgt_beta_weight, self.use_avg)
        loss_beta *= self.beta_weight
        loss += loss_beta
        if self.beta_weight > 0:
            loss_dict['beta'] = loss_beta
        
        #----------------------------- Pose -----------------------------#
        tgt_theta_weight = smpl_weight * labels['target_theta_weight']
        all_samples = output.pred_theta_mats.shape[0]
        if all_samples > batch_size and all_samples % batch_size == 0: # sample version
            pred_theta_mats = output.pred_theta_mats.reshape(batch_size, -1, 24*9)
            loss_theta = weighted_l2_loss(pred_theta_mats[:, 0], labels['target_theta'], 
                                          tgt_theta_weight, self.use_avg)
            tol = 0
            for i in range(1, pred_theta_mats.shape[1]):
                loss_theta += 0.5 * weighted_l2_loss(pred_theta_mats[:, i], labels['target_theta'], 
                                                     tgt_theta_weight, self.use_avg, tol=tol)
        else:   # mode version
            loss_theta = weighted_l2_loss(output.pred_theta_mats, labels['target_theta'], 
                                          tgt_theta_weight, self.use_avg)
        loss_theta *= self.theta_weight
        loss += loss_theta
        if self.theta_weight > 0:
            loss_dict['theta'] = loss_theta
        
        #----------------------------- 2D projection -----------------------------#
        hm_2d_jts = output.hm_2d_jts
        pred_2d_jts = output.pred_2d_jts
        target_2d_jts = labels['target_uvd_29'].reshape(batch_size, -1, 3)[:, :24, :2]
        target_2d_weight = labels['target_weight_29'].reshape(batch_size, -1, 3)[:, :24, :2]

        target_ext_jts = labels['target_uvd_11'].reshape(batch_size, -1, 3)[:, :11, :2]
        target_ext_weight = labels['target_weight_11'].reshape(batch_size, -1, 3)[:, :11, :2]

        target_2d_jts = torch.cat((target_2d_jts, target_ext_jts), dim=1)
        target_2d_weight = torch.cat((target_2d_weight, target_ext_weight), dim=1)
        
        loss_2d = 64 * weighted_l2_loss(hm_2d_jts, target_2d_jts, 
                                        target_2d_weight, self.use_avg)
        if pred_2d_jts.ndim == 4:  # sample version
            loss_2d += 64 * weighted_l2_loss(pred_2d_jts[:, 0], target_2d_jts, 
                                             target_2d_weight, self.use_avg)
            for i in range(1, pred_2d_jts.shape[1]):
                loss_2d += 32 * weighted_l2_loss(pred_2d_jts[:, i], target_2d_jts, 
                                                 target_2d_weight, self.use_avg)
        else:  # mode version
            loss_2d += 64 * weighted_l2_loss(pred_2d_jts, target_2d_jts, 
                                             target_2d_weight, self.use_avg)
        
        loss_2d *= self.uv_weight
        loss += loss_2d
        if self.uv_weight > 0:
            loss_dict['2d'] = loss_2d

        #----------------------------- Regularize MF -----------------------------#
        F_dim = output.pred_F.shape[1]
        rot_norm = 1.7321 * torch.ones((batch_size, F_dim), device=output.pred_F.device)
        pred_norm = torch.linalg.norm(output.pred_F.reshape(batch_size, F_dim, -1), dim=-1)
        loss_reg_F = torch.mean((pred_norm - rot_norm)**2)
        loss_reg_F *= self.reg_weight
        loss += loss_reg_F
        if self.reg_weight > 0:
            loss_dict['reg'] = loss_reg_F

        return loss, loss_dict
