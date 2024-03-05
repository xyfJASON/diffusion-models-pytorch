import torch
import torch.nn as nn
from torch import Tensor


class BaseLatent(nn.Module):
    def __init__(self, scale_factor: float = 1.0):
        super().__init__()
        self.register_buffer('scale_factor', torch.tensor(scale_factor))
        self.device = self.scale_factor.device

    def to(self, *args, **kwargs):
        super().to(*args, **kwargs)
        self.device = self.scale_factor.device
        return self

    def forward(self, x: Tensor, timesteps: Tensor):
        raise NotImplementedError

    def encode_latent(self, x: Tensor):
        raise NotImplementedError

    def decode_latent(self, z: Tensor):
        raise NotImplementedError
