import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms

from propose.models.layers.body_models.smpl import SMPL_layer
from propose.utils.render import SMPLRenderer
from propose.utils.vis import vis_vertices, vis_2d_jts

        
def check_label(inps, labels, img_paths, bboxes, out_dir='./dump_annot'):
    os.makedirs(out_dir, exist_ok=True)

    pairs_24 = [[0, 1], [0, 2], [0, 3], [1, 4], [2, 5], [3, 6], [4, 7], [5, 8], 
                [6, 9], [7, 10], [8, 11], [9, 12], [9, 13], [9, 14], [12, 15], [13, 16], 
                [14, 17], [16, 18], [17, 19], [18, 20], [19, 21], [20, 22], [21, 23]]
    pairs_29 = pairs_24 + [[15, 24], [22, 25], [23, 26], [10, 27], [11, 28]]
    pairs_11 = [[0, 1], [0, 2], [1, 3], [2, 4], [5, 6], [5, 7], [8, 9], [8, 10]]

    mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)
    std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)
    unnormalize = transforms.Normalize((-mean / std).tolist(), (1.0 / std).tolist())
    orig_inps = unnormalize(inps.clone())
    depth_factor = 2.0

    smpl = SMPL_layer('./model_files/smpl/SMPL_NEUTRAL.pkl',
                      X_regressor=None, dtype=torch.float32, use_kid=True)

    batch_size = inps.shape[0]

    poses = labels['target_theta'].reshape(batch_size, 24, 3, 3)
    output = smpl(poses=poses, betas=labels['target_beta'], 
                  kid_betas=labels['kid_betas'], pose2rot=False)
    vertices =  output.vertices.cpu().numpy()
    
    transl = labels['joint_root'].reshape(batch_size, 1, 3).cpu().numpy()

    for b in range(batch_size):
        img = cv2.imread(img_paths[b])
        bbox = bboxes[b].numpy() # xmin, ymin, xmax, ymax
        
        #----------------------------- Cropped image annotations -----------------------------#
        uvd = labels['target_uvd_29'][b].reshape(-1, 3).numpy()
        uvd_weight = labels['target_weight_29'][b].reshape(-1, 3).numpy()
        uvd_11 = labels['target_uvd_11'][b].reshape(-1, 3).numpy()
        uvd_weight_11 = labels['target_weight_11'][b].reshape(-1, 3).numpy()
                
        inp = orig_inps[b].permute(1, 2, 0).clone().numpy()
        inp[inp > 1] = 1
        inp = np.ascontiguousarray(inp*255, dtype=np.uint8)
        inp_size = np.array([inp.shape[1], inp.shape[0]]).reshape(1, 2)

        jts_2d_crop = (uvd[:, :2] + 0.5) * inp_size
        inp = vis_2d_jts(inp, jts_2d_crop, uvd_weight, pairs_29, color=(0, 255, 0))
        jts_2d_crop_11 = (uvd_11[:, :2] + 0.5) * inp_size
        inp = vis_2d_jts(inp, jts_2d_crop_11, uvd_weight_11, pairs_11, color=(0, 0, 255))

        #----------------------------- Whole image annotations -----------------------------#
        jts_3d_cam = labels['target_xyz_24'][b].reshape(-1, 3).numpy()
        xyz_weight = labels['target_weight_24'][b].reshape(-1, 3).numpy()
        jtd_3d_cam_11 = labels['target_xyz_11'][b].reshape(-1, 3).numpy()
        xyz_weight_11 = labels['target_weight_11_xyz'][b].reshape(-1, 3).numpy()
        Kgt = labels['intrinsic_param'][b].numpy()
        trans = labels['joint_root'][b].reshape(1, 3).numpy()

        # bbox
        img = cv2.rectangle(img, tuple(map(int,bbox[:2])), tuple(map(int,bbox[2:])), color=(255,0,0), thickness=4)
        
        # 2D joints in the whole image (aligned)
        trans_inv = labels['trans_inv'][b].numpy()
        trans_jts_2d = jts_2d_crop @ trans_inv[:, :2].T + trans_inv[:, 2:].T
        img = vis_2d_jts(img, trans_jts_2d, uvd_weight, pairs_24)

        # projection of 3D joints (not aligned due to augmentation)
        proj = (jts_3d_cam * depth_factor + trans) @ Kgt.T
        proj = proj[:, :2] / (proj[:, 2:] + 1e-6)
        img = vis_2d_jts(img, proj, xyz_weight, pairs_24, (255, 0, 0))

        proj11 = (jtd_3d_cam_11 * depth_factor + trans) @ Kgt.T
        proj11 = proj11[:, :2] / (proj11[:, 2:] + 1e-6)
        img = vis_2d_jts(img, proj11, xyz_weight_11, pairs_11, (0, 0, 255))

        # SMPL (not aligned)
        if labels['target_smpl_weight'][b, 0] > 0:
            focal = Kgt[[0,1], [0,1]]
            princpt = Kgt[:2, 2]
            renderer = SMPLRenderer(faces=smpl.faces, img_size=img.shape[:2], focal=focal, princpt=princpt)

            vert_shifted = vertices[b] + transl[b].reshape(1, 3)
            img_smpl = vis_vertices(vert_shifted, renderer=renderer, c=princpt, img=img.copy())
            
            img = np.concatenate((img, img_smpl), axis=1)

        cv2.imwrite(os.path.join(out_dir, f'{b:06d}_full.jpg'), img)
        cv2.imwrite(os.path.join(out_dir, f'{b:06d}_crop.jpg'), inp[..., ::-1])

        import ipdb; ipdb.set_trace()