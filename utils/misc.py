import os
import yaml
import random
import shutil
import datetime
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn

from utils.dist import master_only


def init_seeds(seed=0, cuda_deterministic=False):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # Speed-reproducibility tradeoff https://pytorch.org/docs/stable/notes/randomness.html
    if cuda_deterministic:  # slower, more reproducible
        cudnn.deterministic = True
        cudnn.benchmark = False
    else:  # faster, less reproducible
        cudnn.deterministic = False
        cudnn.benchmark = True


def get_device(dist_info):
    if torch.cuda.is_available():
        if dist_info['is_dist']:
            device = torch.device('cuda', dist_info['local_rank'])
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def parse_config(config_path, args=None):
    with open(config_path, 'r') as f:
        original_config = yaml.safe_load(f)
    if args is None:
        return original_config
    config = original_config.copy()
    for k, v in original_config.items():
        if hasattr(args, k):
            config[k] = getattr(args, k)
    return config


def check_freq_epoch(freq: int, epoch: int):
    if freq is None or freq == 0 or not freq >= 1:
        return False
    assert isinstance(freq, int), f'freq >= 1 should be an integer, get {freq}'
    return (epoch + 1) % freq == 0


def check_freq_iteration(freq: float, iteration: int, n_iter_per_epochs: int):
    if freq is None or freq == 0 or not 0. < freq < 1.:
        return False
    freq_iter = int(freq * n_iter_per_epochs)
    return freq_iter > 0 and (iteration + 1) % freq_iter == 0


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


@master_only
def create_log_directory(config, config_path):
    log_root = os.path.join('runs', 'exp-' + get_time_str())
    os.makedirs(log_root)

    if config.get('save_freq'):
        os.makedirs(os.path.join(log_root, 'ckpt'), exist_ok=True)

    if config.get('sample_freq'):
        os.makedirs(os.path.join(log_root, 'samples'), exist_ok=True)

    config_filename = os.path.splitext(os.path.basename(config_path))[0]
    shutil.copyfile(config_path, os.path.join(log_root, config_filename + '.yml'))
    return log_root


def get_bare_model(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model
