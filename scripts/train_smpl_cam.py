import logging
import os
import random
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch
import torch.multiprocessing as mp
import torch.utils.data
from torch.nn.utils import clip_grad
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from propose.datasets import JointDatasetCam
from propose.models import builder
from propose.opt import opt, cfg, logger
from propose.utils.env import setup_seed, init_dist, NullWriter, print_config
from propose.utils.metric import DataLogger, cal_mpjpe


def init_worker(worker_id):
    np.random.seed(opt.seed + worker_id)
    random.seed(opt.seed + worker_id)

def train(opt, train_loader, model, criterion, optimizer, writer):
    loss_logger = DataLogger()
    mpjpe_kpt_logger = DataLogger()
    mpjpe_post_logger = DataLogger()

    model.train()
    
    hm_shape = cfg.MODEL.get('HEATMAP_SIZE')
    depth_dim = cfg.MODEL.EXTRA.get('DEPTH_DIM')
    hm_shape = (hm_shape[1], hm_shape[0], depth_dim)

    depth_factor = float(cfg.MODEL.get('BBOX_3D_SHAPE')[2]) / 1000.

    if opt.log:
        train_loader = tqdm(train_loader, dynamic_ncols=True)

    for (inps, labels, img_paths, bboxes) in train_loader:
        #----------------------------- Data -----------------------------#
        if False:
            from scripts.check import check_label
            check_label(inps, labels, img_paths, bboxes)
            continue

        if isinstance(inps, list):
            inps = [inp.cuda(opt.gpu).requires_grad_() for inp in inps]
        else:
            inps = inps.cuda(opt.gpu).requires_grad_()

        for k, _ in labels.items():
            labels[k] = labels[k].cuda(opt.gpu)

        bboxes = bboxes.cuda(opt.gpu)

        #----------------------------- Forward -----------------------------#
        # torch.autograd.set_detect_anomaly(True)
        output = model(inps, flip_test=opt.flip_test, use_sample=True)
        
        #----------------------------- Loss -----------------------------#
        loss, loss_dict = criterion(output, labels)
        loss_details = ''
        for loss_term in loss_dict:
            loss_details += f'{loss_term}: {loss_dict[loss_term].item():.4f} | '
        loss_details = loss_details[:-3]

        #----------------------------- Metrics -----------------------------#
        if isinstance(inps, list):
            batch_size = inps[0].size(0)
        else:
            batch_size = inps.size(0)
        
        loss_logger.update(loss.item(), batch_size)

        pred_rel_jts = output.pred_rel_jts.detach().cpu().numpy()
        target_rel_jts = depth_factor * torch.cat(
            (labels['target_xyz_24'], labels['target_xyz_11']), dim=1).reshape(batch_size, -1, 3).cpu().numpy()
        target_weight = torch.cat(
            (labels['target_weight_24'], labels['target_weight_11_xyz']), dim=1).reshape(batch_size, -1, 3).cpu().numpy()
        
        mpjpe_kpt, cnt_annot_kpt = cal_mpjpe(
            pred_rel_jts[:, 1:], target_rel_jts[:, 1:], target_weight[:, 1:])
        if cnt_annot_kpt > 0:
            mpjpe_kpt_logger.update(mpjpe_kpt*1000, cnt_annot_kpt)

        pred_X_jts = output.pred_jts_X_post.detach().cpu().numpy()
        target_X_jts = depth_factor * labels['target_xyz_17'].reshape(batch_size, -1, 3).cpu().numpy()
        target_weight_X = labels['target_weight_17'].reshape(batch_size, -1, 3).cpu().numpy()
        if pred_X_jts.shape[0] > batch_size:  # sample version
            pred_X_jts = pred_X_jts.reshape(batch_size, -1, *target_X_jts.shape[1:])
            pred_X_jts = pred_X_jts[:, 0]
        pred_X_jts = pred_X_jts.reshape(batch_size, -1, 3)
        
        mpjpe_post, cnt_annot_post = cal_mpjpe(
            pred_X_jts[:, 1:], target_X_jts[:, 1:], target_weight_X[:, 1:])
        if cnt_annot_post > 0:
            mpjpe_post_logger.update(mpjpe_post*1000, cnt_annot_post)

        #----------------------------- Backward -----------------------------#
        optimizer.zero_grad()
        loss.backward()

        for group in optimizer.param_groups:
            for param in group["params"]:
                clip_grad.clip_grad_norm_(param, 5)

        optimizer.step()

        #----------------------------- Log -----------------------------#
        if opt.log:
            train_loader.set_description(
                f'loss: {loss_logger.avg:.4f} ({loss_details}) ' + \
                f'err_kpt: {mpjpe_kpt_logger.avg:.4f} | err_post: {mpjpe_post_logger.avg:.4f} \n')
            
            writer.add_scalar('total', loss_logger.value, opt.train_iters)

        opt.train_iters += 1

    if opt.log:
        train_loader.close()   # close the progressbar
    
    train_metrics = {'err_kpt': mpjpe_kpt_logger.avg, 'err_post': mpjpe_post_logger.avg}

    return loss_logger.avg, train_metrics


def main_worker(gpu, opt, cfg):
    #----------------------------- Initialization -----------------------------#
    setup_seed(opt.seed)

    opt.gpu = gpu
    init_dist(opt)

    if not opt.log:
        logger.setLevel(logging.FATAL)
        null_writer = NullWriter()
        sys.stdout = null_writer

    if opt.log:
        print_config(logger, opt, cfg)

        if not os.path.exists('./exp/tb_logs'):
            os.mkdir('./exp/tb_logs')
        writer = SummaryWriter(f'./exp/tb_logs/{cfg.DATASET.DATASET}/{cfg.FILE_NAME}-{opt.exp_id}')
    else:
        writer = None

    #----------------------------- Model -----------------------------#
    model = load_model(cfg)
    if opt.params and opt.no_dist:
        from thop import clever_format, profile
        input = torch.randn(1, 3, 256, 256).cuda(opt.gpu)
        flops, params = profile(model.cuda(opt.gpu), inputs=(input, ))
        macs, params = clever_format([flops, params], "%.3f")
        logger.info(macs, params)

    model.cuda(opt.gpu)
    model = torch.nn.parallel.DistributedDataParallel(model)

    #----------------------------- Loss -----------------------------#
    criterion = builder.build_loss(cfg.LOSS).cuda(opt.gpu)
    
    #----------------------------- Optimizer -----------------------------#
    if cfg.TRAIN.OPTIMIZER == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.TRAIN.LR)
    elif cfg.TRAIN.OPTIMIZER == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=cfg.TRAIN.LR, momentum=0.9, weight_decay=5e-4)
    else:
        raise NotImplementedError

    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=cfg.TRAIN.LR_STEP, gamma=cfg.TRAIN.LR_FACTOR)

    #----------------------------- Data -----------------------------#
    if cfg.DATASET.DATASET == 'mix_smpl_cam':
        train_dataset = JointDatasetCam(cfg=cfg, train=True)
    else:
        raise NotImplementedError

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_dataset, num_replicas=opt.world_size, rank=opt.rank)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=cfg.TRAIN.BATCH_SIZE, 
        num_workers=int(opt.num_threads/opt.ngpus_per_node), worker_init_fn=init_worker,
        shuffle=(train_sampler is None), sampler=train_sampler, pin_memory=True)

    #----------------------------- Iteration -----------------------------#
    opt.train_iters = 0
    for i in range(cfg.TRAIN.BEGIN_EPOCH, cfg.TRAIN.END_EPOCH):
        opt.epoch = i
        if train_sampler is not None:
            train_sampler.set_epoch(i)

        current_lr = optimizer.state_dict()['param_groups'][0]['lr']

        logger.info(f'=============== Starting Epoch {opt.epoch} | LR: {current_lr} ===============')

        #----------------------------- Training -----------------------------#
        loss, train_metrics = train(opt, train_loader, model, criterion, optimizer, writer)
        #----------------------------- Training -----------------------------#

        logger.epochInfo('Train', opt.epoch, loss, **train_metrics)

        lr_scheduler.step()

        if (i + 1) % opt.snapshot == 0:
            if opt.log:
                torch.save(model.module.state_dict(), os.path.join(opt.output_dir, f'model_{opt.epoch}.pth'))

        torch.distributed.barrier()  # Sync

    torch.save(model.module.state_dict(), os.path.join(opt.output_dir, 'final_ckpt.pth'))


def load_model(cfg):
    model = builder.build_model(cfg.MODEL)

    if cfg.MODEL.PRETRAINED:
        logger.info(f'Loading model from {cfg.MODEL.PRETRAINED}...')
        model.load_state_dict(torch.load(cfg.MODEL.PRETRAINED))

    elif cfg.MODEL.TRY_LOAD:
        logger.info(f'Loading model from {cfg.MODEL.TRY_LOAD}...')
        pretrained_state = torch.load(cfg.MODEL.TRY_LOAD)
        model_state = model.state_dict()
        
        matched_pretrained_state = {k: v for k, v in pretrained_state.items()
                                    if k in model_state and v.size() == model_state[k].size() 
                                    and not k.startswith('smpl.')}

        # missing_keys = [k for k in model_state.keys() if k not in matched_pretrained_state.keys()]
        # logger.info('Missing keys: ' + ' || '.join(missing_keys))
        # unexpected_keys = [k for k in pretrained_state.keys() if k not in matched_pretrained_state.keys()]
        # logger.info('Unexpected keys: ' + ' || '.join(unexpected_keys))

        model_state.update(matched_pretrained_state)
        model.load_state_dict(model_state)
    else:
        logger.info('Create new model')
        logger.info('=> init weights')
        model._initialize()

    return model


def main():
    setup_seed(opt.seed)

    if opt.no_dist:
        main_worker(0, opt, cfg)
    else:
        mp.spawn(main_worker, nprocs=opt.ngpus_per_node, args=(opt, cfg))


if __name__ == "__main__":
    main()
