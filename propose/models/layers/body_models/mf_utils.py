import numpy as np
import torch


def get_mf_mode(params, num_samples=0):
    """ Get the mode of matrix fisher distribution.
    U, S, Vh = F
    R_est = U @ [[1,0,0],[0,1,0],[0,0,|UV|]] @ Vh, where Vh = V.T
    
    Args:
    - params: (B, 3, 3), the parameter of matrix-Fisher distribution.
    - num_samples: scalar.

    Returns:
    - mode (B, 3, 3) or samples (B, S+1, 3, 3), where the first is the mode.

    ATTN 1: If the condition number of F is large, the solution of svd may be unstable.
    ATTN 2: The api of torch.linalg.svd (U, S, Vh) is different from torch.svd (U, S, V).
    ATTN 3: Batch svd is somewhat different from single svd.
    """

    batch_size = params.shape[0]
    
    diag_mat = 1e-5 * torch.eye(3, device=params.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    U, S, Vh = torch.linalg.svd(params + diag_mat, full_matrices=True)

    with torch.no_grad():
        det_u_v = torch.linalg.det(torch.bmm(U, Vh.transpose(1, 2)))
    det_modify_mat = torch.eye(3, device=U.device).unsqueeze(0).expand(batch_size, -1, -1).clone()
    det_modify_mat[:, 2, 2] = det_u_v

    mode = torch.bmm(torch.bmm(U, det_modify_mat), Vh)

    if num_samples > 0:
        V = Vh.transpose(-2, -1)
        # samples : (B, S, 3, 3)
        samples = mf_sampling_torch(U, S, V, num_samples, b=1.5, oversampling_ratio=8, sample_on_cpu=False)

        return torch.concat((mode.unsqueeze(1), samples), dim=1)

    return mode


def cal_skel_direction(joints, children, parents, num_joint, mt_child_joints, 
                       exclude_root=False, leaf_idx=None):
    """ Calculate unit bone directions of the whole skeleton from 3D joints.
    
    Args:    
    - joints: (B, J, 3), 3D joints.
    - mt_child_joints: list, index of joints with multiple children.
    - exclude_root: bool, if True then the root orientation is not calculated.
    - leaf_idx: list, index of leaf joints.
    
    Returns:
    - st_orit_list: (B, N_single 3, 1), where N_single is the number of single-child node.
    - mt_orit_list: list, each element is (B, 3, Nc), where Nc is the number of its children.
    """

    eps = 1e-8

    sg_child_joints = []
    for j in range(num_joint):
        if j in mt_child_joints: continue
        if leaf_idx is not None and j in leaf_idx: continue
        sg_child_joints.append(j)

    # single-child node
    sg_orit_list = [] 
    for pa in sg_child_joints:
        ch = children[pa]
        sg_pa_orit = joints[:, ch, :] - joints[:, pa, :]
        unit_sg_pa_orit = sg_pa_orit / (torch.linalg.norm(sg_pa_orit, dim=-1, keepdim=True) + eps)
        sg_orit_list.append(unit_sg_pa_orit)
    sg_orit_list = torch.stack(sg_orit_list, dim=0).transpose(0, 1).unsqueeze(-1)
    
    # multi-child node
    if exclude_root:
        new_mt_child_joints = mt_child_joints[1:]
    else:
        new_mt_child_joints = mt_child_joints
    mt_orit_list = [[] for i in range(len(new_mt_child_joints))]
    for i, pa in enumerate(new_mt_child_joints):
        for ch, map_pa in enumerate(parents):
            if pa == map_pa:
                mt_pa_orit = joints[:, ch, :] - joints[:, pa, :]
                unit_mt_pa_orit = mt_pa_orit / (torch.linalg.norm(mt_pa_orit, dim=-1, keepdim=True) + eps)
                mt_orit_list[i].append(unit_mt_pa_orit)

    mt_orit_list = [torch.stack(imt, dim=0).permute(1, 2, 0) for imt in mt_orit_list]  # list of (B, 3, Nc)
    
    return sg_orit_list, mt_orit_list


def cal_bone_direction(parent_pose, children_pose):
    """ Calculate unit bone directions from adjacent 3D joints.
    
    Args:
    - parent_pose: (B, 1, 3)
    - children_pose: (B, Nc, 3), where Nc is the number of children.

    Returns:
    - orit: (B, 3, Nc), The 3D direction vector of bones.
    """

    orit = children_pose - parent_pose # broadcast
    orit = orit / (torch.linalg.norm(orit, dim=-1, keepdim=True) + 1e-8)
    orit = orit.permute(0, 2, 1)

    return orit


def glb2rel(glb_transforms, parents, root_idx=0):
    """ Calculate local rotations relative to parents from global rotations.
    
    Args:
    - glb_transforms: (B, J, 3, 3), global rotation matrix.
    - parents: list.
    - root_idx: int, index of root joint.

    Returns:
    - rel_transforms: (B, J, 3, 3), relative rotation matrix.
    """
    
    rel_transforms = torch.zeros_like(glb_transforms)
    rel_transforms[:, root_idx, :, :] = glb_transforms[:, root_idx, :, :]
    for ch, pa in enumerate(parents):
        if pa < 0: continue
        rel_transforms[:, ch] = torch.matmul(glb_transforms[:, pa].transpose(1, 2), glb_transforms[:, ch])
    
    return rel_transforms


def quat_to_rotmat(quat):
    """Convert quaternion coefficients to rotation matrix.

    Args:
    - quat: (B, 4) in (w, x, y, z) representation.
    
    Returns:
    - (B, 3, 3), rotation matrix corresponding to the quaternion.
    """
    norm_quat = quat
    norm_quat = norm_quat / (norm_quat.norm(p=2, dim=1, keepdim=True) + 1e-8)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def bingham_sampling_torch(A, num_samples, Omega=None, Gaussian_std=None,
                           b=1.5, M_star=None, oversampling_ratio=8):
    """ Sampling from a Bingham distribution with 4x4 matrix parameter A.
    Here we assume that A is a diagonal matrix (needed for matrix-Fisher sampling).
    Bing(A) is simulated by rejection sampling from ACG(I + 2A/b) (since ACG > Bingham everywhere).
    Rejection sampling is batched + differentiable (using re-parameterisation trick).
    From https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman
    For further details, see: https://arxiv.org/pdf/1310.8110.pdf and
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution

    Args:
    - A: (4,) tensor parameter of Bingham distribution on 3-sphere.
        Represents the diagonal of a 4x4 diagonal matrix.
    - num_samples: scalar. Number of samples to draw.
    - Omega: (4,) Optional tensor parameter of ACG distribution on 3-sphere.
    - Gaussian_std: (4,) Optional tensor parameter (standard deviations) of diagonal Gaussian in R^4.
    - b: Hyperparameter for rejection sampling using envelope ACG distribution with
        Omega = I + 2A/b
    - oversampling_ratio: scalar. To make rejection sampling batched, we sample num_samples * oversampling_ratio,
        then reject samples according to rejection criterion, and hopeffully the number of samples remaining is
        > num_samples.

    Returns:
    - samples: (num_samples, 4) and accept_ratio.
    """
    assert A.shape == (4,)
    assert A.min() >= 0

    if Omega is None:
        Omega = torch.ones(4, device=A.device) + 2*A/b  # Will sample from ACG(Omega) with Omega = I + 2A/b.
    if Gaussian_std is None:
        Gaussian_std = Omega ** (-0.5)  # Sigma^0.5 = (Omega^-1)^0.5 = Omega^-0.5
    if M_star is None:
        M_star = np.exp(-(4 - b) / 2) * ((4 / b) ** 2)  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    samples_obtained = False
    while not samples_obtained:
        eps = torch.randn(num_samples * oversampling_ratio, 4, device=A.device).float()
        y = Gaussian_std * eps
        samples = y / torch.norm(y, dim=1, keepdim=True)  # (num_samples * oversampling_ratio, 4)

        with torch.no_grad():
            p_Bing_star = torch.exp(-torch.einsum('bn,n,bn->b', samples, A, samples))  # (num_samples * oversampling_ratio,)
            p_ACG_star = torch.einsum('bn,n,bn->b', samples, Omega, samples) ** (-2)  # (num_samples * oversampling_ratio,)
            # assert torch.all(p_Bing_star <= M_star * p_ACG_star + 1e-6)

            w = torch.rand(num_samples * oversampling_ratio, device=A.device)
            accept_vector = w < p_Bing_star / (M_star * p_ACG_star)  # (num_samples * oversampling_ratio,)
            num_accepted = accept_vector.sum().item()
        if num_accepted >= num_samples:
            samples = samples[accept_vector, :]  # (num_accepted, 4)
            samples = samples[:num_samples, :]  # (num_samples, 4)
            samples_obtained = True
            accept_ratio = num_accepted / num_samples * 4
        else:
            print('Failed sampling. {} samples accepted, {} samples required.'.format(num_accepted, num_samples))

    return samples, accept_ratio


def mf_sampling_torch(pose_U, pose_S, pose_V, num_samples,
                      b=1.5, oversampling_ratio=8, sample_on_cpu=False):
    """ Sampling from matrix-Fisher distributions defined on 3D rotation matrices.
    MF distribution is simulated by sampling quaternions Bingham distribution and
    converting quaternions to rotation matrices.
    From https://github.com/akashsengupta1997/HierarchicalProbabilistic3DHuman
    For further details, see: https://arxiv.org/pdf/1310.8110.pdf and
    https://github.com/tylee-fdcl/Matrix-Fisher-Distribution

    Args:
    - pose_U: (B, 3, 3)
    - pose_S: (B, 3)
    - pose_V: (B, 3, 3)
    - num_samples: scalar, number of samples.
    - b: scalar, hyperparameter for rejection sampling using envelope ACG distribution, which can also be computed.
    - oversampling_ratio: scalar. To make rejection sampling batched, we sample num_samples * oversampling_ratio,
        then reject samples according to rejection criterion, and hopeffully the number of samples remaining is
        > num_samples.
    - sample_on_cpu: do sampling on CPU instead of GPU.
    
    Returns: 
    - R_samples: (B, num samples, 3, 3)
    """

    batch_size = pose_U.shape[0]

    # Proper SVD
    with torch.no_grad():
        detU, detV = torch.det(pose_U.detach().cpu()).to(pose_U.device), torch.det(pose_V.detach().cpu()).to(pose_V.device)
    pose_U_proper = pose_U.clone()
    pose_S_proper = pose_S.clone()
    pose_V_proper = pose_V.clone()
    pose_S_proper[:, 2] *= detU * detV  # Proper singular values: s3 = s3 * det(UV)
    pose_U_proper[:, :, 2] *= detU.unsqueeze(-1)  # Proper U = U diag(1, 1, det(U))
    pose_V_proper[:, :, 2] *= detV.unsqueeze(-1)

    # Sample quaternions from Bingham(A)
    if sample_on_cpu:
        sample_device = 'cpu'
    else:
        sample_device = pose_S_proper.device
    bingham_A = torch.zeros(batch_size, 4, device=sample_device)
    bingham_A[:, 1] = 2 * (pose_S_proper[:, 1] + pose_S_proper[:, 2])
    bingham_A[:, 2] = 2 * (pose_S_proper[:, 0] + pose_S_proper[:, 2])
    bingham_A[:, 3] = 2 * (pose_S_proper[:, 0] + pose_S_proper[:, 1])

    Omega = torch.ones(batch_size, 4, device=bingham_A.device) + 2 * bingham_A / b  # sample from ACG(Omega) with Omega = I + 2A/b.
    Gaussian_std = Omega ** (-0.5)  # Sigma^0.5 = (Omega^-1)^0.5 = Omega^-0.5
    M_star = np.exp(-(4 - b) / 2) * ((4 / b) ** 2)  # Bound for rejection sampling: Bing(A) <= Mstar(b)ACG(I+2A/b)

    pose_quat_samples_batch = torch.zeros(batch_size, num_samples, 4, device=pose_U.device).float()
    for i in range(batch_size):
        quat_samples, accept_ratio = bingham_sampling_torch(A=bingham_A[i, :],
                                                            num_samples=num_samples,
                                                            Omega=Omega[i, :],
                                                            Gaussian_std=Gaussian_std[i, :],
                                                            b=b,
                                                            M_star=M_star,
                                                            oversampling_ratio=oversampling_ratio)
        pose_quat_samples_batch[i, :, :] = quat_samples

    pose_R_samples_batch = quat_to_rotmat(
        quat=pose_quat_samples_batch.view(-1, 4)).view(batch_size, num_samples, 3, 3)
    pose_R_samples_batch = torch.matmul(pose_U_proper[:, None, :, :],
                                        torch.matmul(pose_R_samples_batch, 
                                                     pose_V_proper[:, None, :, :].transpose(dim0=-1, dim1=-2)))

    return pose_R_samples_batch