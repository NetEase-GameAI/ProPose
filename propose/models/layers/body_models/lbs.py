import numpy as np
import torch
import torch.nn.functional as F
from propose.utils.transforms import batch_rodrigues
from propose.datasets.convert import EXTRA_VERT_IDS
from .mf_utils import get_mf_mode, cal_skel_direction, glb2rel, cal_bone_direction


def lbs(betas, pose, 
        v_template, shapedirs, posedirs, lbs_weights, J_regressor, parents,
        pose2rot=True, X_regressor=None, dtype=torch.float32, kid_betas=None):

    batch_size = max(betas.shape[0], pose.shape[0])
    device = betas.device
    num_smpl_joints = J_regressor.shape[0]  # 24

    # Shape blending
    if kid_betas is not None:
        new_betas = torch.concat((betas, kid_betas), dim=1)
        v_shaped = v_template + blend_shapes(new_betas, shapedirs)
    else:
        v_shaped = v_template + blend_shapes(betas, shapedirs)

    # Get rest joints
    J = vertices2joints(J_regressor, v_shaped)

    # Get rotation matrices
    if pose2rot:
        rot_mats = batch_rodrigues(
            pose.view(-1, 3), dtype=dtype).view([batch_size, -1, 3, 3])
    else:
        rot_mats = pose.view([batch_size, -1, 3, 3])

    # Pose blending
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).view([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped

    # Get global joints
    J_transformed, A = batch_rigid_transform(
        rot_mats, J, parents[:num_smpl_joints])

    # Skinning
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    T = torch.matmul(W, A.view(batch_size, num_smpl_joints, 16)).view(batch_size, -1, 4, 4)

    # Get vertices
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    # Get regressed joints
    X_from_verts = None
    if X_regressor is not None:
        X_from_verts = vertices2joints(X_regressor, verts)

    return verts, J_transformed, rot_mats, X_from_verts


def propose_core(betas, joints_3d, mf_params, 
                 v_template, shapedirs, posedirs, lbs_weights, J_regressor, parents, 
                 X_regressor=None, dtype=torch.float32, train=False, 
                 kappas=None, kid_betas=None, use_sample=False):
    """ Probabilistic fusion of rotational priors and directional likelihood.

    Args:
    - betas: (B, 10), SMPL shape parameters.
    - joints_3d: (B, J, 3), 3D keypoints as likelihood.
    - mf_params: (B, 24*9), prior MF parameters of 24 SMPL joints.
    - kappas: (B, 24, 1), optional, concentration parameters. Note: not J dims.
    - kid_betas: (B, 1), optional, last one of SMIL kid shape parameters.
    - use_sample: bool, whether use sampling or not.
    
    Returns:
    - verts: (BxS, 6890, 3), SMPL vertices. S=1: mode version; S>1: sample version.
    - J_transformed: (BxS, 24, 3), fused 24 SMPL joints.
    - rot_mats: (BxS, 24, 3, 3), fused SMPL pose parameters via SVD.
    - X_from_verts: (BxS, X, 3), 3D joints from vertices via X_regressor.
    - post_mf_params: (B, 24, 3, 3), posterior MF parameters of 24 SMPL joints.
    """

    device = betas.device
    num_smpl_joints = J_regressor.shape[0]  # 24
    num_joints_3d = joints_3d.shape[1]

    #----------------------------- Shape blending -----------------------------#
    # SMIL for kids
    if kid_betas is not None:
        new_betas = torch.concat((betas, kid_betas), dim=1)
        v_shaped = v_template + blend_shapes(new_betas, shapedirs)
    # SMPL
    else:
        v_shaped = v_template + blend_shapes(betas, shapedirs)

    #----------------------------- Prepare human template -----------------------------#
    if num_joints_3d > num_smpl_joints:
        leaf_idx = EXTRA_VERT_IDS
        rest_J = torch.zeros((v_shaped.shape[0], num_joints_3d, 3), dtype=dtype, device=device)
        rest_J[:, :num_smpl_joints] = vertices2joints(J_regressor, v_shaped)
        leaf_vertices = v_shaped[:, leaf_idx].clone()
        rest_J[:, num_smpl_joints:] = leaf_vertices
    else:
        rest_J = vertices2joints(J_regressor, v_shaped)
    
    children = torch.ones_like(parents) * -1
    for ch, pa in enumerate(parents):
        if pa >= 0 and children[pa] == -1:
            children[pa] = ch

    #----------------------------- Get rotation matrices -----------------------------#
    if train:
        mf_rot_mats, post_mf_params = mf2rot_global(
            joints_3d, mf_params, rest_J.clone(), children, parents,
            dtype=dtype, kappas=kappas, use_sample=use_sample)
    else:
        mf_rot_mats, post_mf_params = mf2rot_global(
            joints_3d, mf_params, rest_J.clone(), children, parents,
            dtype=dtype, kappas=kappas, use_sample=use_sample)

    rot_mats = mf_rot_mats

    #----------------------------- For sample-verson -----------------------------#
    n_samples = 1
    if use_sample and rot_mats.ndim == 5:
        n_samples = rot_mats.shape[1]
    rot_mats = rot_mats.reshape(-1, 24, 3, 3)
    batch_size = rot_mats.shape[0]
    
    rest_J_new = rest_J[:, :num_smpl_joints].clone()
    if use_sample:
        rest_J_new = \
            rest_J_new.unsqueeze(1).repeat(1, n_samples, 1, 1).reshape(batch_size, 24, 3)
        v_shaped = \
            v_shaped.unsqueeze(1).repeat(1, n_samples, 1, 1).reshape(batch_size, *v_shaped.shape[1:])

    #----------------------------- Get global joints -----------------------------#
    J_transformed, A = batch_rigid_transform(
        rot_mats, rest_J_new, parents[:num_smpl_joints])

    #----------------------------- Pose blending -----------------------------#    
    ident = torch.eye(3, dtype=dtype, device=device)
    pose_feature = (rot_mats[:, 1:] - ident).reshape([batch_size, -1])
    pose_offsets = torch.matmul(pose_feature, posedirs).view(batch_size, -1, 3)
    v_posed = pose_offsets + v_shaped

    #----------------------------- Skinning -----------------------------#   
    W = lbs_weights.unsqueeze(dim=0).expand([batch_size, -1, -1])
    T = torch.matmul(W, A.view(batch_size, num_smpl_joints, 16)).view(batch_size, -1, 4, 4)

    #----------------------------- Get vertices -----------------------------#   
    homogen_coord = torch.ones([batch_size, v_posed.shape[1], 1], dtype=dtype, device=device)
    v_posed_homo = torch.cat([v_posed, homogen_coord], dim=2)
    v_homo = torch.matmul(T, torch.unsqueeze(v_posed_homo, dim=-1))
    verts = v_homo[:, :, :3, 0]

    #----------------------------- Get other regressed joints -----------------------------#   
    X_from_verts = None
    if X_regressor is not None:
        X_from_verts = vertices2joints(X_regressor, verts)

    return verts, J_transformed, rot_mats, X_from_verts, post_mf_params


def mf2rot_global(joints_3d, mf_params, rest_pose, children, parents, 
                  dtype=torch.float32, kappas=None, use_sample=False):

    batch_size = mf_params.shape[0]
    device = mf_params.device

    num_samples = 8 if use_sample else 0
    sample_weight = 3.0
    kappa_weight = 100.0

    #----------------------------- Skeleton definition -----------------------------#
    num_smpl_joints = 24
    root_idx = 0  # (parents==-1).nonzero().item()

    # Divide joints, ATTN: assume root_idx = mt_child_joints[0] = 0
    mt_child_joints = [root_idx, 9, 15]
    if joints_3d.shape[1] > num_smpl_joints:
        leaf_joints = [10, 11, 22, 23]
    else:
        leaf_joints = [10, 11, 15, 22, 23]
    sg_child_joints = [j for j in range(num_smpl_joints) 
                       if j not in mt_child_joints and j not in leaf_joints]
    
    #----------------------------- Process kappa -----------------------------#
    if kappas is None:
        kappas = torch.ones((batch_size, num_smpl_joints, 1), dtype=dtype, device=device)
    scaled_kappas = kappa_weight * torch.ones_like(kappas)

    mf_mats = mf_params.reshape(batch_size, num_smpl_joints, 3, 3)

    #----------------------------- Process root -----------------------------#
    root_children = [ch for (ch, pa) in enumerate(parents) if pa == root_idx] # [1, 2, 3]
    root_rest_orit = cal_bone_direction(rest_pose[:, [root_idx]], rest_pose[:, root_children])
    root_pose_orit = cal_bone_direction(joints_3d[:, [root_idx]], joints_3d[:, root_children])
    root_rest_orit = root_rest_orit.permute(0, 2, 1)

    root_dlT = torch.matmul(root_pose_orit, root_rest_orit)  # (B, 3, 3)
    root_post_mf_mats = mf_mats[:, root_idx] + scaled_kappas[:, [root_idx]] * root_dlT

    global_orient_mat = get_mf_mode(root_post_mf_mats, num_samples=0)  # not sample global rotation

    #----------------------------- Process other joints -----------------------------#
    # Warp human with the calculated global rotation
    rotated_rest_pose = torch.matmul(rest_pose, global_orient_mat.transpose(1, 2))

    # Calculate unit orientation from other joints
    sg_pose_orit, mt_pose_orit = cal_skel_direction(
        joints_3d, children, parents, num_smpl_joints, mt_child_joints, 
        exclude_root=True, leaf_idx=leaf_joints)
    sg_rest_orit, mt_rest_orit = cal_skel_direction(
        rotated_rest_pose, children, parents, num_smpl_joints, mt_child_joints, 
        exclude_root=True, leaf_idx=leaf_joints)

    sg_rest_orit = sg_rest_orit.transpose(2, 3)
    sg_orit = torch.einsum('abcd,abdf->abcf', sg_pose_orit, sg_rest_orit)  # (B, N_single, 3, 3)
    
    mt_orit = []
    for j in range(len(mt_pose_orit)):
        mt_pose_orit_j = mt_pose_orit[j]  # (B, 3, Nc)
        mt_rest_orit_j = mt_rest_orit[j].transpose(1, 2)
        mt_orit_j = torch.matmul(mt_pose_orit_j, mt_rest_orit_j)
        mt_orit.append(mt_orit_j)
    mt_orit = torch.stack(mt_orit).transpose(0, 1)  # (B, N_multi, 3, 3)

    # Measured orientation (likelihood)
    orit = torch.zeros_like(mf_mats, dtype=dtype, device=device)
    orit[:, sg_child_joints, :, :] = sg_orit
    orit[:, mt_child_joints[1:], :, :] = mt_orit
    orit = orit[:, 1:]   # remove global rotation

    mea_orit = torch.einsum('ab,abcd->abcd', scaled_kappas[:, 1:, 0], orit)
    post_mf_mats = mf_mats[:, 1:] + mea_orit

    if not use_sample:
        # (B, 23, 3, 3)
        rot_mats_est = get_mf_mode(
            post_mf_mats.reshape(-1, 3, 3)).reshape(batch_size, num_smpl_joints-1, 3, 3)
    else:
        # (B, 23, S, 3, 3)
        rot_mats_est = get_mf_mode(
            post_mf_mats.reshape(-1, 3, 3)*sample_weight, num_samples=num_samples).reshape(
            batch_size, num_smpl_joints-1, num_samples+1, 3, 3)

    #----------------------------- Transform to standard -----------------------------#
    # Warp back with global rotation
    if not use_sample:
        ident = torch.eye(3, device=device, dtype=dtype) \
                     .reshape(1, 1, 3, 3).repeat(batch_size, 1, 1, 1)
    else:
        ident = torch.eye(3, device=device, dtype=dtype) \
                     .reshape(1, 1, 1, 3, 3).repeat(batch_size, 1, num_samples+1, 1, 1)
    rot_mats_est = torch.concat((ident, rot_mats_est), axis=1)
    
    if not use_sample:
        rot_mats_est = torch.einsum('bjcd,bde->bjce', rot_mats_est, global_orient_mat)
    else:
        rot_mats_est = torch.einsum('bjscd,bde->bjsce', rot_mats_est, global_orient_mat) \
                            .transpose(1, 2).reshape(-1, num_smpl_joints, 3, 3)

    # Global rotation to local rotation relative to parent joint
    rot_mats_est = glb2rel(rot_mats_est, parents[:num_smpl_joints], root_idx=root_idx)
    if use_sample:
        rot_mats_est = rot_mats_est.reshape(batch_size, num_samples+1, num_smpl_joints, 3, 3)

    # Posterior matrix-Fisher matrix
    post_mf_mats = torch.concat((root_post_mf_mats.unsqueeze(1), post_mf_mats), dim=1)

    return rot_mats_est, post_mf_mats


def vertices2joints(J_regressor, vertices):
    """ Calculates the 3D joint locations from the vertices.

    Args:
    - J_regressor: (J, V), the regressor to calculate joints from vertices.
    - vertices : (B, V, 3), the tensor of mesh vertices.

    Returns: 
    - (B, J, 3), the location of the joints.
    """

    return torch.einsum('bik,ji->bjk', [vertices, J_regressor])


def blend_shapes(betas, shape_disps):
    """ Calculates the per vertex displacement due to the blend shapes.
    Displacement[b, m, k] = sum_{l} betas[b, l] * shape_disps[m, k, l].
    Multiply each shape displacement by its corresponding beta and then sum them.

    Args:
    - betas: (B, num_betas), blend shape coefficients.
    - shape_disps: (V, 3, num_betas), blend shapes.

    Returns:
    - blend_shape: (B, V, 3), the per-vertex displacement due to shape deformation.
    """

    blend_shape = torch.einsum('bl,mkl->bmk', [betas, shape_disps])
    return blend_shape


def transform_mat(R, t):
    """ Creates a batch of transformation matrices.

    Args:
    - R: (B, 3, 3), array of a batch of rotation matrices.
    - t: (B, 3, 1), array of a batch of translation vectors.
    
    Returns:
    - (B, 4, 4), transformation matrix.
    """
    # No padding left or right, only add an extra row
    return torch.cat([F.pad(R, [0, 0, 0, 1]),
                      F.pad(t, [0, 0, 0, 1], value=1)], dim=2)


def batch_rigid_transform(rot_mats, joints, parents):
    """ Applies a batch of rigid transformations to the joints.

    Args:
    - rot_mats : (B, J, 3, 3), tensor of rotation matrices.
    - joints : (B, J, 3), locations of joints (Template Pose).
    - parents : the kinematic tree of each object.

    Returns:
    - posed_joints :(B, J, 3), the locations of the joints after applying the pose rotations.
    - rel_transforms : (B, J, 4, 4), relative (wrt root joint) rigid transformations for all joints.
    """

    root_idx = 0

    joints = torch.unsqueeze(joints, dim=-1)
    rel_joints = joints.clone()
    rel_joints[:, 1:] -= joints[:, parents[1:]].clone()

    # (B, J, 4, 4)
    transforms_mat = transform_mat(
        rot_mats.reshape(-1, 3, 3),
        rel_joints.reshape(-1, 3, 1)).reshape(-1, joints.shape[1], 4, 4)

    transform_chain = [transforms_mat[:, root_idx]]
    for i in range(1, parents.shape[0]):
        curr_res = torch.matmul(transform_chain[parents[i]], transforms_mat[:, i])
        transform_chain.append(curr_res)

    # (B, J, 4, 4)
    transforms = torch.stack(transform_chain, dim=1)

    # The last column of the transformations contains the posed joints
    posed_joints = transforms[:, :, :3, 3]

    joints_homogen = F.pad(joints, [0, 0, 0, 1])

    rel_transforms = transforms - F.pad(
        torch.matmul(transforms, joints_homogen), [3, 0, 0, 0, 0, 0, 0, 0])

    return posed_joints, rel_transforms
