import math

import torch
import torch.nn as nn
from torch import Tensor


def init_weights(init_type=None, gain=0.02):

    def init_func(m):
        classname = m.__class__.__name__

        if classname.find('BatchNorm') != -1:
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.normal_(m.weight, 1.0, gain)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

        elif hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight, 0.0, gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=gain)
            elif init_type == 'xavier_uniform':
                nn.init.xavier_uniform_(m.weight, gain=1.0)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=gain)
            elif init_type is None:
                m.reset_parameters()
            else:
                raise ValueError(f'invalid initialization method: {init_type}.')
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_func


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, X: Tensor):
        """
        Args:
            X (Tensor): [bs]
        Returns:
            Sinusoidal embeddings of shape [bs, dim]
        """
        half_dim = self.dim // 2
        embed = math.log(10000) / (half_dim - 1)
        embed = torch.exp(torch.arange(half_dim, device=X.device) * -embed)
        embed = X[:, None] * embed[None, :]
        embed = torch.cat((embed.sin(), embed.cos()), dim=-1)
        return embed


def Upsample(in_channels: int, out_channels: int):
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode='nearest'),
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
    )


def Downsample(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 1, groups: int = 32):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(groups, dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1)
        self.k = nn.Conv2d(dim, dim, kernel_size=1)
        self.v = nn.Conv2d(dim, dim, kernel_size=1)
        self.proj = nn.Conv2d(dim, dim, kernel_size=1)
        self.scale = (dim // n_heads) ** -0.5

    def forward(self, X: Tensor):
        bs, C, H, W = X.shape
        normX = self.norm(X)
        q = self.q(normX).view(bs * self.n_heads, -1, H*W)
        k = self.k(normX).view(bs * self.n_heads, -1, H*W)
        v = self.v(normX).view(bs * self.n_heads, -1, H*W)
        q = q * self.scale
        attn = torch.bmm(q.permute(0, 2, 1), k).softmax(dim=-1)
        output = torch.bmm(v, attn.permute(0, 2, 1)).view(bs, -1, H, W)
        output = self.proj(output)
        return output + X


class AdaGN(nn.Module):
    def __init__(self, num_groups: int, num_channels: int, embed_dim: int):
        super().__init__()
        self.gn = nn.GroupNorm(num_groups, num_channels)
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, num_channels * 2),
        )

    def forward(self, X: Tensor, embed: Tensor):
        """
        Args:
            X (Tensor): [bs, C, H, W]
            embed (Tensor): [bs, embed_dim]
        """
        ys, yb = torch.chunk(self.proj(embed), 2, dim=-1)
        ys = ys[:, :, None, None]
        yb = yb[:, :, None, None]
        return self.gn(X) * (1 + ys) + yb
