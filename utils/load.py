import os

import torch
from safetensors.torch import load_file


def load_weights(path: str):
    ext = os.path.splitext(path)[-1]
    if ext == '.safetensors':
        weights = load_file(path, device='cpu')         # saved by stable diffusion
    else:
        weights = torch.load(path, map_location='cpu')
        if 'state_dict' in weights:
            weights = weights['state_dict']             # saved by stable diffusion
        elif 'ema' in weights:
            weights = weights['ema']['shadow']          # saved by this repo (legacy)
        elif 'model' in weights:
            weights = weights['model']                  # saved by this repo
    return weights
