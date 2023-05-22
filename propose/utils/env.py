from datetime import datetime
import os
import random

import numpy as np
import torch
import torch.distributed as dist


def setup_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def init_dist(opt):
    opt.rank = opt.rank * opt.ngpus_per_node + opt.gpu
    
    dist.init_process_group(backend=opt.dist_backend, init_method=opt.dist_url,
                            world_size=opt.world_size, rank=opt.rank)
    
    torch.cuda.set_device(opt.gpu)
    
    print(f'{opt.dist_url}, worldsize:{opt.world_size}, rank:{opt.rank}')

    if opt.rank == 0:
        opt.log = True
    else:
        opt.log = False


class NullWriter(object):
    def write(self, args):
        pass

    def flush(self):
        pass


def print_config(logger, opt, cfg):    
    now = datetime.now().strftime("%D:%H:%M:%S")
    logger.info(f'=============== !!!!!!!!!!!!!!! ===============')
    logger.info(f'=============== Start at {now} ===============')

    logger.info('=============== OPT ===============')
    logger.info(opt)

    logger.info('=============== CONFIG ===============')
    logger.info(cfg)
    logger.info('==============================')