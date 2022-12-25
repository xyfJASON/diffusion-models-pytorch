import random
import datetime
import argparse
import numpy as np

import torch
import torch.nn as nn
from torch.backends import cudnn


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
        if dist_info.is_dist:
            device = torch.device('cuda', dist_info.local_rank)
        else:
            device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    return device


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def check_freq(freq: int, step: int):
    assert isinstance(freq, int)
    return freq >= 1 and (step + 1) % freq == 0


def get_time_str():
    return datetime.datetime.now().strftime('%Y-%m-%d-%H-%M-%S')


def get_bare_model(model):
    if isinstance(model, (nn.DataParallel, nn.parallel.DistributedDataParallel)):
        return model.module
    else:
        return model
