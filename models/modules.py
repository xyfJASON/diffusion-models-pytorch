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


class ResBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, groups: int = 32, dropout: float = 0.1):
        super().__init__()
        self.blk1 = nn.Sequential(
            nn.GroupNorm(groups, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.proj = nn.Sequential(
            nn.SiLU(),
            nn.Linear(embed_dim, out_channels),
        )
        self.blk2 = nn.Sequential(
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
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
        X = self.blk1(X)
        if time_embed is not None:
            X = X + self.proj(time_embed)[:, :, None, None]
        X = self.blk2(X)
        return X + shortcut


class UNet(nn.Module):
    def __init__(self,
                 img_channels: int = 3,
                 dim: int = 128,
                 dim_mults: List[int] = (1, 2, 2, 2),
                 use_attn: List[int] = (False, True, False, False),
                 num_res_blocks: int = 2,
                 resblock_groups: int = 32,
                 attn_groups: int = 32,
                 attn_heads: int = 1,
                 dropout: float = 0.1):
        super().__init__()
        self.img_channels = img_channels
        n_stages = len(dim_mults)
        dims = [dim]

        # Time embeddings
        time_embed_dim = dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )

        # First convolution
        self.first_conv = nn.Conv2d(img_channels, dim, 3, stride=1, padding=1)
        cur_dim = dim

        # Down-sample blocks
        # Default: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.down_blocks = nn.ModuleList([])
        for i in range(n_stages):
            out_dim = dim * dim_mults[i]
            stage_blocks = nn.ModuleList([])
            for j in range(num_res_blocks):
                stage_blocks.append(ResBlock(cur_dim, out_dim, embed_dim=time_embed_dim,
                                             groups=resblock_groups, dropout=dropout))
                if use_attn[i]:
                    stage_blocks.append(SelfAttentionBlock(out_dim, n_heads=attn_heads, groups=attn_groups))
                dims.append(out_dim)
                cur_dim = out_dim
            if i < n_stages - 1:
                stage_blocks.append(Downsample(out_dim, out_dim))
                dims.append(out_dim)
            self.down_blocks.append(stage_blocks)

        # Bottleneck block
        self.bottleneck_block = nn.ModuleList([
            ResBlock(cur_dim, cur_dim, embed_dim=time_embed_dim, dropout=dropout),
            SelfAttentionBlock(cur_dim),
            ResBlock(cur_dim, cur_dim, embed_dim=time_embed_dim, dropout=dropout),
        ])

        # Up-sample blocks
        # Default: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.up_blocks = nn.ModuleList([])
        for i in range(n_stages-1, -1, -1):
            out_dim = dim * dim_mults[i]
            stage_blocks = nn.ModuleList([])
            for j in range(num_res_blocks + 1):
                stage_blocks.append(ResBlock(dims.pop() + cur_dim, out_dim, embed_dim=time_embed_dim,
                                             groups=resblock_groups, dropout=dropout))
                if use_attn[i]:
                    stage_blocks.append(SelfAttentionBlock(out_dim, n_heads=attn_heads, groups=attn_groups))
                cur_dim = out_dim
            if i > 0:
                stage_blocks.append(Upsample(out_dim, out_dim))
            self.up_blocks.append(stage_blocks)

        # Last convolution
        self.last_conv = nn.Sequential(
            nn.GroupNorm(resblock_groups, cur_dim),
            nn.SiLU(),
            nn.Conv2d(cur_dim, img_channels, 3, stride=1, padding=1),
        )

    def forward(self, X: Tensor, T: Tensor):
        time_embed = self.time_embed(T)
        X = self.first_conv(X)
        skips = [X]

        for stage_blocks in self.down_blocks:
            for blk in stage_blocks:  # noqa
                if isinstance(blk, ResBlock):
                    X = blk(X, time_embed)
                    skips.append(X)
                elif isinstance(blk, SelfAttentionBlock):
                    X = blk(X)
                    skips[-1] = X
                else:  # Downsample
                    X = blk(X)
                    skips.append(X)

        X = self.bottleneck_block[0](X, time_embed)
        X = self.bottleneck_block[1](X)
        X = self.bottleneck_block[2](X, time_embed)

        for stage_blocks in self.up_blocks:
            for blk in stage_blocks:  # noqa
                if isinstance(blk, ResBlock):
                    X = blk(torch.cat((X, skips.pop()), dim=1), time_embed)
                elif isinstance(blk, SelfAttentionBlock):
                    X = blk(X)
                else:  # Upsample
                    X = blk(X)

        X = self.last_conv(X)
        return X


def _test():
    unet = UNet()
    print(unet)
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
        dropout=0.0,
    )
    X = torch.empty((10, 1, 256, 256))
    T = torch.arange(10)
    out = unet(X, T)
    print(out.shape)
    print(sum(p.numel() for p in unet.parameters()))


if __name__ == '__main__':
    _test()
