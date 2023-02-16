from yacs.config import CfgNode as CN
import torch.optim as optim
import models


def build_model(cfg: CN):
    if cfg.MODEL.TYPE.lower() == 'unet':
        model = models.UNet(
            img_channels=cfg.DATA.IMG_CHANNELS,
            dim=cfg.MODEL.DIM,
            dim_mults=cfg.MODEL.DIM_MULTS,
            use_attn=cfg.MODEL.USE_ATTN,
            num_res_blocks=cfg.MODEL.NUM_RES_BLOCKS,
            resblock_groups=cfg.MODEL.RESBLOCK_GROUPS,
            attn_groups=cfg.MODEL.ATTN_GROUPS,
            attn_heads=cfg.MODEL.ATTN_HEADS,
            dropout=cfg.MODEL.DROPOUT,
        )
    elif cfg.MODEL.TYPE.lower() == 'unet_cond':
        model = models.UNetConditional(

        )
    else:
        raise ValueError
    return model


def build_optimizer(params, cfg: CN):
    cfg = cfg.TRAIN.OPTIM
    if cfg.TYPE == 'SGD':
        optimizer = optim.SGD(
            params=params,
            lr=cfg.LR,
            momentum=getattr(cfg, 'MOMENTUM', 0),
            weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0),
            nesterov=getattr(cfg, 'NESTEROV', False),
        )
    elif cfg.TYPE == 'Adam':
        optimizer = optim.Adam(
            params=params,
            lr=cfg.LR,
            betas=getattr(cfg, 'BETAS', (0.9, 0.999)),
            weight_decay=getattr(cfg, 'WEIGHT_DECAY', 0),
        )
    else:
        raise ValueError(f"Optimizer {cfg.TYPE} is not supported.")
    return optimizer
