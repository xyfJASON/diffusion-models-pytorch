import os
import functools
from typing import List

import torch
from torch import Tensor
import torch.distributed as dist


def init_dist():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
        dist.init_process_group(backend='nccl')
        dist.barrier()
    else:
        print('Not using distributed mode')


def get_dist_info():
    if dist.is_available() and dist.is_initialized():
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
        global_rank = int(os.environ['RANK'])
        is_dist = True
    else:
        world_size = 1
        local_rank = global_rank = 0
        is_dist = False
    return dict(world_size=world_size,
                local_rank=local_rank,
                global_rank=global_rank,
                is_master=(global_rank == 0),
                is_dist=is_dist)


def master_only(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if get_dist_info()['is_master']:
            return func(*args, **kwargs)
    return wrapper


def reduce_tensor(tensor: Tensor, n: int = None):
    dist_info = get_dist_info()
    if dist_info['is_dist'] is False:
        return tensor
    if n is None:
        n = dist_info['world_size']
    rt = tensor.clone() / n
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    return rt


def reduce_tensors(tensors: List[Tensor], n: int = None):
    return [reduce_tensor(tensor, n) for tensor in tensors]
