from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from models.modules import SinusoidalPosEmb, SelfAttentionBlock, Downsample, Upsample


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
