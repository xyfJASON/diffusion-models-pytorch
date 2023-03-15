from typing import List
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.modules import SinusoidalPosEmb, SelfAttentionBlock, Downsample, Upsample, AdaGN


class ResBlock(nn.Module):
    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            embed_dim: int,
            dropout: float = 0.1,
            up: bool = False,
            down: bool = False,
    ):
        super().__init__()
        assert not (up and down), 'up and down cannot both be True'
        if up:
            self.updown = partial(F.interpolate, scale_factor=2, mode='nearest')
        elif down:
            self.updown = partial(F.avg_pool2d, kernel_size=2, stride=2)
        else:
            self.updown = None

        self.blk1 = nn.Sequential(
            nn.GroupNorm(32, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, stride=1, padding=1),
        )
        self.adagn = AdaGN(32, out_channels, embed_dim)
        self.blk2 = nn.Sequential(
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, stride=1, padding=1),
        )
        self.shortcut = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, X: Tensor, embed: Tensor):
        """
        Args:
            X (Tensor): [bs, C, H, W]
            embed (Tensor): [bs, embed_dim]
        Returns:
            [bs, C', H, W]
        """
        if self.updown:
            blk1_1, blk1_2 = self.blk1[:-1], self.blk1[-1]
            h = blk1_1(X)
            h = self.updown(h)
            X = self.updown(X)
            h = blk1_2(h)
        else:
            h = self.blk1(X)
        h = self.adagn(h, embed)
        h = self.blk2(h)
        return h + self.shortcut(X)


class ResBlockUpsample(ResBlock):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, dropout: float = 0.1):
        super().__init__(in_channels, out_channels, embed_dim, dropout, up=True)


class ResBlockDownsample(ResBlock):
    def __init__(self, in_channels: int, out_channels: int, embed_dim: int, dropout: float = 0.1):
        super().__init__(in_channels, out_channels, embed_dim, dropout, down=True)


class UNetCategorialAdaGN(nn.Module):
    """ UNet conditioned on categorial labels with AdaGN """
    def __init__(
            self,
            in_channels: int = 3,
            out_channels: int = 3,
            dim: int = 128,
            dim_mults: List[int] = (1, 2, 2, 2),
            use_attn: List[int] = (False, True, True, False),
            num_res_blocks: int = 2,
            num_classes: int = None,
            attn_head_dims: int = 64,
            resblock_updown: bool = True,
            dropout: float = 0.1,
    ):
        super().__init__()
        n_stages = len(dim_mults)
        dims = [dim]

        # Time embeddings
        embed_dim = dim * 4
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(dim),
            nn.Linear(dim, embed_dim),
            nn.SiLU(),
            nn.Linear(embed_dim, embed_dim),
        )

        # Class embeddings
        self.class_embed = nn.Embedding(num_classes, embed_dim) if num_classes is not None else None

        # First convolution
        self.first_conv = nn.Conv2d(in_channels, dim, 3, stride=1, padding=1)
        cur_dim = dim

        # Down-sample blocks
        # Default: 32x32 -> 16x16 -> 8x8 -> 4x4
        self.down_blocks = nn.ModuleList([])
        for i in range(n_stages):
            out_dim = dim * dim_mults[i]
            stage_blocks = nn.ModuleList([])
            for j in range(num_res_blocks):
                stage_blocks.append(ResBlock(cur_dim, out_dim, embed_dim=embed_dim, dropout=dropout))
                if use_attn[i]:
                    assert out_dim % attn_head_dims == 0
                    attn_heads = out_dim // attn_head_dims
                    stage_blocks.append(SelfAttentionBlock(out_dim, n_heads=attn_heads))
                dims.append(out_dim)
                cur_dim = out_dim
            if i < n_stages - 1:
                if resblock_updown:
                    stage_blocks.append(ResBlockDownsample(out_dim, out_dim, embed_dim=embed_dim, dropout=dropout))
                else:
                    stage_blocks.append(Downsample(out_dim, out_dim))
                dims.append(out_dim)
            self.down_blocks.append(stage_blocks)

        # Bottleneck block
        self.bottleneck_block = nn.ModuleList([
            ResBlock(cur_dim, cur_dim, embed_dim=embed_dim, dropout=dropout),
            SelfAttentionBlock(cur_dim),
            ResBlock(cur_dim, cur_dim, embed_dim=embed_dim, dropout=dropout),
        ])

        # Up-sample blocks
        # Default: 4x4 -> 8x8 -> 16x16 -> 32x32
        self.up_blocks = nn.ModuleList([])
        for i in range(n_stages-1, -1, -1):
            out_dim = dim * dim_mults[i]
            stage_blocks = nn.ModuleList([])
            for j in range(num_res_blocks + 1):
                stage_blocks.append(ResBlock(dims.pop() + cur_dim, out_dim, embed_dim=embed_dim, dropout=dropout))
                if use_attn[i]:
                    attn_heads = out_dim // attn_head_dims
                    stage_blocks.append(SelfAttentionBlock(out_dim, n_heads=attn_heads))
                cur_dim = out_dim
            if i > 0:
                if resblock_updown:
                    stage_blocks.append(ResBlockUpsample(out_dim, out_dim, embed_dim=embed_dim, dropout=dropout))
                else:
                    stage_blocks.append(Upsample(out_dim, out_dim))
            self.up_blocks.append(stage_blocks)

        # Last convolution
        self.last_conv = nn.Sequential(
            nn.GroupNorm(32, cur_dim),
            nn.SiLU(),
            nn.Conv2d(cur_dim, out_channels, 3, stride=1, padding=1),
        )

    def forward(self, X: Tensor, T: Tensor, y: Tensor = None):
        """
        Args:
            X (Tensor): [bs, C, H, W]
            T (Tensor): [bs]
            y (Tensor): [bs] or None, categorial labels
        """
        time_embed = self.time_embed(T)
        if self.class_embed is not None and y is not None:
            time_embed = time_embed + self.class_embed(y)

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
                else:
                    X = blk(X)
                    skips.append(X)

        X = self.bottleneck_block[0](X, time_embed)
        X = self.bottleneck_block[1](X)
        X = self.bottleneck_block[2](X, time_embed)

        for stage_blocks in self.up_blocks:
            for blk in stage_blocks:  # noqa
                if isinstance(blk, ResBlockUpsample):
                    X = blk(X, time_embed)
                elif isinstance(blk, ResBlock):
                    X = torch.cat((X, skips.pop()), dim=1)
                    X = blk(X, time_embed)
                elif isinstance(blk, SelfAttentionBlock):
                    X = blk(X)
                else:
                    X = blk(X)

        X = self.last_conv(X)
        return X


def _test():
    unet = UNetCategorialAdaGN()
    X = torch.empty((10, 3, 32, 32))
    T = torch.arange(10)
    y = torch.arange(10)
    out = unet(X, T, y)
    print(out.shape)
    print(sum(p.numel() for p in unet.parameters()))

    unet = UNetCategorialAdaGN(
        in_channels=1,
        out_channels=1,
        dim=128,
        dim_mults=[1, 1, 2, 2, 4, 4],
        use_attn=[False, False, False, False, True, False],
        dropout=0.0,
    )
    X = torch.empty((10, 1, 256, 256))
    T = torch.arange(10)
    y = torch.arange(10)
    out = unet(X, T, y)
    print(out.shape)
    print(sum(p.numel() for p in unet.parameters()))


if __name__ == '__main__':
    _test()
