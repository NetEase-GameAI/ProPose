import argparse
import logging
import os
from types import MethodType

import torch

from .utils.config import update_config


parser = argparse.ArgumentParser(description='ProPose Training')

#----------------------------- Experiment options -----------------------------#
parser.add_argument('--cfg', required=True, type=str,
                    help='experiment configure file name')
parser.add_argument('--exp-id', default='default', type=str,
                    help='experiment ID')

#----------------------------- Training options -----------------------------#
parser.add_argument('--seed', default=42, type=int,
                    help='random seed')
parser.add_argument('--num_threads', default=8, type=int,
                    help='number of data loading threads')
parser.add_argument('--snapshot', default=2, type=int,
                    help='frequency of taking a snapshot of the model (0 = never)')

#----------------------------- Distributed options -----------------------------#
parser.add_argument('--rank', default=0, type=int,
                    help='node rank for distributed training')
parser.add_argument('--local_rank', default=0, type=int,
                    help='local rank')
parser.add_argument('--dist-url', default='tcp://192.168.1.1:12321', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')

#----------------------------- Other options -----------------------------#
parser.add_argument('--params', default=False, dest='params',
                    help='calculate model size', action='store_true')
parser.add_argument('--flip-test', default=False, dest='flip_test',
                    help='flip test', action='store_true')
parser.add_argument('--no_dist', default=False, dest='no_dist',
                    help='flip test', action='store_true')


opt = parser.parse_args()
cfg_file_name = opt.cfg.split('/')[-1]
cfg = update_config(opt.cfg)

cfg['FILE_NAME'] = cfg_file_name

#----------------------------- GPUs setting -----------------------------#
num_gpu = torch.cuda.device_count()
if num_gpu < cfg.TRAIN.WORLD_SIZE:
    cfg.TRAIN.WORLD_SIZE = num_gpu

opt.ngpus_per_node = cfg.TRAIN.WORLD_SIZE
opt.world_size = cfg.TRAIN.WORLD_SIZE * cfg.TRAIN.NUM_NODES

#----------------------------- Logger setting -----------------------------#
output_dir = f'./exp/{cfg.DATASET.DATASET}/{cfg.FILE_NAME}-{opt.exp_id}'
opt.output_dir = output_dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

logger = logging.getLogger('propose')
logger.setLevel(logging.INFO)

filehandler = logging.FileHandler(os.path.join(output_dir, 'train.log'))
logger.addHandler(filehandler)

streamhandler = logging.StreamHandler()
logger.addHandler(streamhandler)

def epochInfo(self, set, epoch, loss, **kwargs):
    info_line = f'{set}-epoch {epoch:d} | loss:{loss:.5f}'
    for name, val in kwargs.items():
        info_line += ' | '
        info_line += f'{name}:{val:.4f}'
    self.info(info_line)

logger.epochInfo = MethodType(epochInfo, logger)
