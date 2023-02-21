from argparse import Namespace
import torch.optim as optim
import models


def build_model(args: Namespace):
    if args.model_type.lower() == 'unet':
        model = models.UNet(
            in_channels=args.model_in_channels,
            out_channels=args.model_out_channels,
            dim=args.model_dim,
            dim_mults=args.model_dim_mults,
            use_attn=args.model_use_attn,
            num_res_blocks=args.model_num_res_blocks,
            resblock_groups=args.model_resblock_groups,
            attn_groups=args.model_attn_groups,
            attn_heads=args.model_attn_heads,
            dropout=args.model_dropout,
        )
    elif args.model_type.lower() == 'unet_cond':
        model = models.UNetConditional(
            in_channels=args.model_in_channels,
            out_channels=args.model_out_channels,
            dim=args.model_dim,
            dim_mults=args.model_dim_mults,
            use_attn=args.model_use_attn,
            num_res_blocks=args.model_num_res_blocks,
            num_classes=args.model_num_classes,
            resblock_groups=args.model_resblock_groups,
            attn_groups=args.model_attn_groups,
            attn_head_dims=args.model_attn_head_dims,
            resblock_updown=args.model_resblock_updown,
            dropout=args.model_dropout,
        )
    elif args.model_type.lower() == 'openai/guided_diffusion/unet':
        from models.openai.guided_diffusion.unet import UNetModel
        model = UNetModel(
            image_size=args.model_image_size,
            in_channels=args.model_in_channels,
            model_channels=args.model_model_channels,
            out_channels=args.model_out_channels,
            num_res_blocks=args.model_num_res_blocks,
            attention_resolutions=args.model_attention_resolutions,
            dropout=args.model_dropout,
            channel_mult=args.model_channel_mult,
            conv_resample=args.model_conv_resample,
            dims=args.model_dims,
            num_classes=args.model_num_classes,
            use_checkpoint=args.model_use_checkpoint,
            use_fp16=args.model_use_fp16,
            num_heads=args.model_num_heads,
            num_head_channels=args.model_num_head_channels,
            num_heads_upsample=args.model_num_heads_upsample,
            use_scale_shift_norm=args.model_use_scale_shift_norm,
            resblock_updown=args.model_resblock_updown,
            use_new_attention_order=args.model_use_new_attention_order,
        )
    else:
        raise ValueError
    return model


def build_optimizer(params, args: Namespace):
    if args.optim_type.lower() == 'sgd':
        optimizer = optim.SGD(
            params=params,
            lr=args.lr,
            momentum=getattr(args, 'momentum', 0),
            weight_decay=getattr(args, 'weight_decay', 0),
            nesterov=getattr(args, 'nesterov', False),
        )
    elif args.optim_type.lower() == 'adam':
        optimizer = optim.Adam(
            params=params,
            lr=args.lr,
            betas=getattr(args, 'betas', (0.9, 0.999)),
            weight_decay=getattr(args, 'weight_decay', 0),
        )
    elif args.optim_type.lower() == 'adamw':
        optimizer = optim.AdamW(
            params=params,
            lr=args.lr,
            betas=getattr(args, 'betas', (0.9, 0.999)),
            weight_decay=getattr(args, 'weight_decay', 0.01),
        )
    else:
        raise ValueError(f"Optimizer {args.optim_type} is not supported.")
    return optimizer
