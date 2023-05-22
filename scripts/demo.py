import argparse
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
from easydict import EasyDict as edict
import numpy as np
import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
import torchvision.transforms as transforms
from tqdm import tqdm

from propose.models import builder
from propose.utils.bbox import bbox_xyxy_to_xywh, get_one_box
from propose.utils.config import update_config
from propose.utils.render import SMPLRenderer
from propose.utils.vis import vis_vertices
from propose.utils.wrapper import SMPL3DCamWrapper


def main_worker(opt, cfg):
    #----------------------------- Transformation -----------------------------#
    bbox_3d_shape = getattr(cfg.MODEL, 'BBOX_3D_SHAPE', (2000, 2000, 2000))
    dummpy_set = edict()
    for j in [17, 24, 29, 11]:
        dummpy_set[f'joint_pairs_{j}'] = None
    dummpy_set['bbox_3d_shape'] = bbox_3d_shape

    hmr_transform = SMPL3DCamWrapper(
        dummpy_set, 
        input_size=cfg.MODEL.IMAGE_SIZE, output_size=cfg.MODEL.HEATMAP_SIZE, 
        train=False, scale_factor=0, color_factor=0, occlusion=False, 
        add_dpg=False, rot=0, flip=False, scale_mult=1.25, with_smpl=True)

    det_transform = transforms.Compose([transforms.ToTensor()])

    #----------------------------- Model -----------------------------#
    det_model = fasterrcnn_resnet50_fpn(pretrained=True)

    hmr_model = builder.build_model(cfg.MODEL)
    print(f'Loading model from {opt.ckpt}...')
    model_state = hmr_model.state_dict()
    pretrained_state = torch.load(opt.ckpt, map_location='cpu')
    matched_pretrained_state = {k: v for k, v in pretrained_state.items()
                                if not k.startswith('smpl.')}
    
    model_state.update(matched_pretrained_state)
    hmr_model.load_state_dict(model_state)

    det_model.cuda(opt.gpu)
    hmr_model.cuda(opt.gpu)
    det_model.eval()
    hmr_model.eval()

    #----------------------------- Data -----------------------------#
    files = os.listdir(opt.img_dir)

    if not os.path.exists(opt.out_dir):
        os.makedirs(opt.out_dir)

    #----------------------------- Loop -----------------------------#
    for file in tqdm(files):
        if os.path.isdir(file) or file[-4:] not in ['.jpg', '.png']:
            continue
   
        img_path = os.path.join(opt.img_dir, file)
        res_path = os.path.join(opt.out_dir, file)

        #----------------------------- Detect -----------------------------#
        try:
            input_image = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        except:
            continue

        det_input = det_transform(input_image).to(opt.gpu)
        det_output = det_model([det_input])[0]

        image_vis = input_image.copy()
        img_size = (image_vis.shape[0], image_vis.shape[1])

        if opt.mode == 'single':
            tight_bboxes = [get_one_box(det_output)]  # xyxy
        else:
            tight_bboxes = []
            for ib, det_label in enumerate(det_output['labels']):
                if det_label.item() == 1 and det_output['scores'][ib].item() > 0.8:
                    tight_bboxes.append(det_output['boxes'][ib].detach().cpu().numpy())
                if ib > 10:
                    break

        if tight_bboxes[0] is None:
            continue
        
        #----------------------------- Interence -----------------------------#
        inps_batch, bbox_batch = [], []
        for tight_bbox in tight_bboxes:
            inps, bbox = hmr_transform.test_transform(img_path, tight_bbox)
            inps_batch.append(inps)
            bbox_batch.append(bbox)

        inps_batch = torch.stack(inps_batch)
        inps_batch = inps_batch.to(opt.gpu)
        with torch.no_grad():
            pose_output = hmr_model(inps_batch, flip_test=False)
        
        #----------------------------- Visualization -----------------------------#
        focal_length = float(cfg.MODEL.FOCAL_LENGTH) 
        focal = np.asarray([focal_length, focal_length], dtype=np.float32)
        inp_width = float(cfg.MODEL.IMAGE_SIZE[1])

        transl_batch, princpt_batch = [], []
        num_person = len(inps_batch)
        for ib in range(num_person):
            bbox = bbox_batch[ib]
            bbox = bbox_xyxy_to_xywh(bbox)
            
            princpt = np.asarray([bbox[0]+bbox[2]/2, bbox[1]+bbox[3]/2], dtype=np.float32)
            princpt_batch.append(princpt)
            
            transl = pose_output.transl[ib].cpu().numpy()
            transl[..., 2] = transl[..., 2] * inp_width / bbox[2]
            transl_batch.append(transl)

        # plot person according to the depth order (far to near)      
        depth_batch = np.concatenate(transl_batch).reshape(-1, 3)[:, 2]
        depth_order = np.argsort(-depth_batch, axis=0)

        for ib in depth_order:
            princpt = princpt_batch[ib]
            transl = transl_batch[ib]

            renderer = SMPLRenderer(faces=hmr_model.smpl.faces,
                                    img_size=img_size, focal=focal, princpt=princpt)

            vertices = pose_output.pred_vertices.reshape(num_person, -1, 6890, 3)[ib, 0].cpu().numpy()
            vert_shifted = vertices + transl.reshape(1, 3)
            image_vis = vis_vertices(vert_shifted, renderer=renderer, c=princpt, img=image_vis, color_id=0)

        image_vis = cv2.cvtColor(image_vis, cv2.COLOR_RGB2BGR)

        cv2.imwrite(res_path, image_vis)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='ProPose Demo')

    parser.add_argument('--gpu', default=0, type=int,
                        help='gpu')
    parser.add_argument('--img-dir', default='', type=str,
                        help='image folder')
    parser.add_argument('--out-dir', default='dump_demo', type=str,
                        help='output folder')
    parser.add_argument('--ckpt', default='./model_files/propose_hrnet_w48.pth', type=str,
                        help='checkpoint path')
    parser.add_argument('--mode', default='single', type=str, choices=['single', 'multi'],
                        help='detect single person or multiple people')
    opt = parser.parse_args()

    cfg_file = 'configs/smpl_hm_xyz.yaml'
    cfg = update_config(cfg_file)

    main_worker(opt, cfg)