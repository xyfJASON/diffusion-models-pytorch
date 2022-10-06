import os
import sys
import random
import shutil
import datetime
import numpy as np

import torch
from torch.backends import cudnn
import torch.distributed as dist


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def set_device():
    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:                   # multiple gpus
            dist.init_process_group(backend='nccl')
            world_size = int(os.environ['WORLD_SIZE'])
            local_rank = int(os.environ['LOCAL_RANK'])
            global_rank = int(os.environ['RANK'])
            torch.cuda.set_device(local_rank)
            device = torch.device('cuda', local_rank)
        else:                                               # single gpu
            world_size, local_rank, global_rank = 1, 0, 0
            device = torch.device('cuda')
    else:                                                   # cpu
        world_size, local_rank, global_rank = 0, -1, -1
        device = torch.device('cpu')
    return device, world_size, local_rank, global_rank


def create_log_directory(config, config_path):
    if config.get('resume', None) and config['resume']['log_root'] is not None:
        assert os.path.isdir(config['resume']['log_root'])
        log_root = config['resume']['log_root']
    else:
        log_root = os.path.join('runs', datetime.datetime.now().strftime('exp-%Y-%m-%d-%H-%M-%S'))
        os.makedirs(log_root)
    print('log directory:', log_root)

    if config.get('save_freq'):
        os.makedirs(os.path.join(log_root, 'ckpt'), exist_ok=True)
    if config.get('sample_freq'):
        os.makedirs(os.path.join(log_root, 'samples'), exist_ok=True)
    if os.path.exists(os.path.join(log_root, 'config.yml')):
        print('Warning: config.yml exists and will be replaced by a new one.', file=sys.stderr)
    shutil.copyfile(config_path, os.path.join(log_root, 'config.yml'))
    return log_root


def reduce_tensor(tensor, n):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= n
    return rt
