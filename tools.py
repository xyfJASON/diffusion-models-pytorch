from yacs.config import CfgNode as CN

import torch.optim as optim

import models


def build_model(cfg: CN, with_ema: bool = False):
    if cfg.model.type.lower() == 'unet':
        from models.unet import UNet
        model = UNet(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            dim=cfg.model.dim,
            dim_mults=cfg.model.dim_mults,
            use_attn=cfg.model.use_attn,
            num_res_blocks=cfg.model.num_res_blocks,
            n_heads=cfg.model.n_heads,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.type.lower() == 'unet_categorial_adagn':
        from models.unet_categorial_adagn import UNetCategorialAdaGN
        model = UNetCategorialAdaGN(
            in_channels=cfg.model.in_channels,
            out_channels=cfg.model.out_channels,
            dim=cfg.model.dim,
            dim_mults=cfg.model.dim_mults,
            use_attn=cfg.model.use_attn,
            num_res_blocks=cfg.model.num_res_blocks,
            num_classes=cfg.model.num_classes,
            attn_head_dims=cfg.model.attn_head_dims,
            resblock_updown=cfg.model.resblock_updown,
            dropout=cfg.model.dropout,
        )
    elif cfg.model.type.lower() == 'openai/guided_diffusion/unet':
        from models.openai.guided_diffusion.unet import UNetModel
        model = UNetModel(
            image_size=cfg.model.image_size,
            in_channels=cfg.model.in_channels,
            model_channels=cfg.model.model_channels,
            out_channels=cfg.model.out_channels,
            num_res_blocks=cfg.model.num_res_blocks,
            attention_resolutions=cfg.model.attention_resolutions,
            dropout=cfg.model.dropout,
            channel_mult=cfg.model.channel_mult,
            conv_resample=cfg.model.conv_resample,
            dims=cfg.model.dims,
            num_classes=cfg.model.num_classes,
            use_checkpoint=cfg.model.use_checkpoint,
            use_fp16=cfg.model.use_fp16,
            num_heads=cfg.model.num_heads,
            num_head_channels=cfg.model.num_head_channels,
            num_heads_upsample=cfg.model.num_heads_upsample,
            use_scale_shift_norm=cfg.model.use_scale_shift_norm,
            resblock_updown=cfg.model.resblock_updown,
            use_new_attention_order=cfg.model.use_new_attention_order,
        )
    elif cfg.model.type.lower() == 'pesser/pytorch_diffusion/model':
        from models.pesser.pytorch_diffusion.model import Model
        model = Model(
            resolution=cfg.model.resolution,
            in_channels=cfg.model.in_channels,
            out_ch=cfg.model.out_ch,
            ch=cfg.model.ch,
            ch_mult=tuple(cfg.model.ch_mult),
            num_res_blocks=cfg.model.num_res_blocks,
            attn_resolutions=cfg.model.attn_resolutions,
            dropout=cfg.model.dropout,
            resamp_with_conv=cfg.model.resamp_with_conv,
        )
    else:
        raise ValueError

    if not with_ema:
        return model

    ema = models.EMA(
        model=model,
        decay=cfg.model.ema_decay,
        gradual=getattr(cfg.model, 'ema_gradual', True),
    )
    return model, ema


def build_optimizer(params, cfg: CN):
    cfg = cfg.train.optim
    if cfg.type.lower() == 'sgd':
        optimizer = optim.SGD(
            params=params,
            lr=cfg.lr,
            momentum=getattr(cfg, 'momentum', 0),
            weight_decay=getattr(cfg, 'weight_decay', 0),
            nesterov=getattr(cfg, 'nesterov', False),
        )
    elif cfg.type.lower() == 'adam':
        optimizer = optim.Adam(
            params=params,
            lr=cfg.lr,
            betas=getattr(cfg, 'betas', (0.9, 0.999)),
            weight_decay=getattr(cfg, 'weight_decay', 0),
        )
    elif cfg.type.lower() == 'adamw':
        optimizer = optim.AdamW(
            params=params,
            lr=cfg.lr,
            betas=getattr(cfg, 'betas', (0.9, 0.999)),
            weight_decay=getattr(cfg, 'weight_decay', 0.01),
        )
    else:
        raise ValueError(f"Optimizer {cfg.type} is not supported.")
    return optimizer
