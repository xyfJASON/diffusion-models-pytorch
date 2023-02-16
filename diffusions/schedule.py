import math
import torch


def get_beta_schedule(beta_schedule: str = 'linear',
                      total_steps: int = 1000,
                      beta_start: float = 0.0001,
                      beta_end: float = 0.02):
    if beta_schedule == 'linear':
        return torch.linspace(beta_start, beta_end, total_steps, dtype=torch.float64)
    elif beta_schedule == 'quad':
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, total_steps, dtype=torch.float64) ** 2
    elif beta_schedule == 'const':
        return torch.full((total_steps, ), fill_value=beta_end, dtype=torch.float64)
    else:
        raise ValueError(f'Beta schedule {beta_schedule} is not supported.')


def get_skip_seq(skip_type: str = 'uniform', skip_steps: int = 1000, total_steps: int = 1000):
    if skip_type == 'uniform':
        skip = total_steps // skip_steps
        seq = torch.arange(0, total_steps, skip)
    elif skip_type == 'quad':
        seq = torch.linspace(0, math.sqrt(total_steps * 0.8), skip_steps) ** 2
        seq = torch.floor(seq).to(dtype=torch.int64)
    else:
        raise ValueError(f'skip_type {skip_type} is not valid')
    return seq
