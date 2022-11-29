import math
from typing import List

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
    def __init__(self, embed_dim: int):
        super().__init__()
        self.embed_dim = embed_dim

    def forward(self, X: Tensor):
        """
        Args:
            X (Tensor): [bs]
        Returns:
            Sinusoidal embeddings of shape [bs, embed_dim]
        """
        half_dim = self.embed_dim // 2
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
    # return nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)


def Downsample(in_channels: int, out_channels: int):
    return nn.Conv2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1)


class ConvNormAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int,
                 stride: int,
                 padding: int,
                 norm: str = None,
                 activation: str = None,
                 init_type: str = None,
                 groups: int = 8):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=True)

        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        elif norm == 'gn':
            self.norm = nn.GroupNorm(groups, out_channels)
        elif norm is None:
            self.norm = None
        else:
            raise ValueError(f'norm {norm} is not valid.')

        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'elu':
            self.activation = nn.ELU()
        elif activation == 'gelu':
            self.activation = nn.GELU()
        elif activation == 'silu':
            self.activation = nn.SiLU()
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation is None:
            self.activation = None
        else:
            raise ValueError(f'activation {activation} is not valid.')

        self.conv.apply(init_weights(init_type))

    def forward(self, X: Tensor):
        X = self.conv(X)
        if self.norm:
            X = self.norm(X)
        if self.activation:
            X = self.activation(X)
        return X


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int = 4, groups: int = 1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.norm = nn.GroupNorm(groups, dim)
        self.q = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.k = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
        self.v = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
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
        return output


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, groups: int = 8):
        super().__init__()
        self.conv1 = ConvNormAct(in_channels, out_channels, 3, stride=1, padding=1, norm='gn', activation='silu', groups=groups)
        self.proj = nn.Sequential(nn.SiLU(), nn.Linear(embed_dim, out_channels))
        self.conv2 = ConvNormAct(out_channels, out_channels, 3, stride=1, padding=1, norm='gn', activation='silu', groups=groups)
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, X: Tensor, time_embed: Tensor = None):
        """
        Args:
            X (Tensor): [bs, C, H, W]
            time_embed (Tensor): [bs, embed_dim]
        Returns:
            [bs, C', H, W]
        """
        shortcut = self.shortcut(X)
        X = self.conv1(X)
        if time_embed is not None:
            X = X + self.proj(time_embed)[:, :, None, None]
        X = self.conv2(X)
        return X + shortcut


class UNet(nn.Module):
    def __init__(self,
                 img_channels: int = 3,
                 dim: int = 64,
                 dim_mults: List[int] = (1, 2, 4, 8),
                 use_attn: List[int] = (False, True, False, False),
                 resblock_groups: int = 8,
                 attn_groups: int = 1,
                 attn_heads: int = 4):
        super().__init__()
        self.img_channels = img_channels
        n_stages = len(dim_mults)
        dims = [dim * i for i in dim_mults]

        # Time embeddings
        time_embed_dim = dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(time_embed_dim),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # First convolution
        self.first_conv = nn.Conv2d(img_channels, dim, 5, stride=1, padding=2)

        # Down-sample blocks
        # Default: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.down_blocks = nn.ModuleList([])
        for i in range(n_stages):
            self.down_blocks.append(
                nn.ModuleList([
                    ResBlock(dims[i], dims[i], embed_dim=time_embed_dim, groups=resblock_groups),
                    SelfAttentionBlock(dims[i], n_heads=attn_heads, groups=attn_groups) if use_attn[i] else nn.Identity(),
                    ResBlock(dims[i], dims[i], embed_dim=time_embed_dim, groups=resblock_groups),
                    Downsample(dims[i], dims[i+1]) if i < n_stages - 1 else nn.Identity(),
                ])
            )

        # Bottleneck block
        self.bottleneck_block = nn.ModuleList([
            ResBlock(dims[-1], dims[-1], embed_dim=time_embed_dim),
            SelfAttentionBlock(dims[-1]),
            ResBlock(dims[-1], dims[-1], embed_dim=time_embed_dim),
        ])

        # Up-sample blocks
        # Default: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.up_blocks = nn.ModuleList([])
        for i in range(n_stages):
            self.up_blocks.append(
                nn.ModuleList([
                    ResBlock(dims[i] * 2, dims[i], embed_dim=time_embed_dim, groups=resblock_groups),
                    SelfAttentionBlock(dims[i], n_heads=attn_heads, groups=attn_groups) if use_attn[i] else nn.Identity(),
                    ResBlock(dims[i], dims[i], embed_dim=time_embed_dim, groups=resblock_groups),
                    Upsample(dims[i], dims[i-1]) if i > 0 else nn.Identity(),
                ])
            )

        # Last convolution
        self.last_conv = nn.Conv2d(dim, img_channels, 1)

    def forward(self, X: Tensor, T: Tensor):
        time_embed = self.time_embed(T)
        X = self.first_conv(X)

        skips = []
        for blk1, attn, blk2, down in self.down_blocks:
            X = blk1(X, time_embed)
            X = attn(X)
            X = blk2(X, time_embed)
            skips.append(X)
            X = down(X)

        X = self.bottleneck_block[0](X, time_embed)
        X = self.bottleneck_block[1](X)
        X = self.bottleneck_block[2](X, time_embed)

        for (blk1, attn, blk2, up), skip in zip(reversed(self.up_blocks), reversed(skips)):
            X = blk1(torch.cat((X, skip), dim=1), time_embed)
            X = attn(X)
            X = blk2(X, time_embed)
            X = up(X)

        X = self.last_conv(X)
        return X


def _test():
    unet = UNet()
    X = torch.empty((10, 3, 32, 32))
    T = torch.arange(10)
    out = unet(X, T)
    print(out.shape)
    print(sum(p.numel() for p in unet.parameters()))

    unet = UNet(
        img_channels=1,
        dim=128,
        dim_mults=[1, 1, 2, 2, 4, 4],
        use_attn=[False, False, False, False, True, False],
    )
    X = torch.empty((10, 1, 256, 256))
    T = torch.arange(10)
    out = unet(X, T)
    print(out.shape)
    print(sum(p.numel() for p in unet.parameters()))


if __name__ == '__main__':
    _test()
