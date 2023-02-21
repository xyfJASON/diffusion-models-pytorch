import os
import yaml
import math
import argparse
from functools import partial

import torch
from torchvision.utils import save_image

import diffusions
from engine.tools import build_model
from utils.logger import get_logger
from utils.misc import init_seeds, add_args
from utils.dist import init_distributed_mode, get_rank, get_world_size


def parse_args():
    parser = argparse.ArgumentParser()
    # data configuration file
    parser.add_argument(
        '--config_data', type=str, required=True,
        help='Path to data configuration file',
    )
    # model configuration file
    parser.add_argument(
        '--config_model', type=str, required=True,
        help='Path to model configuration file',
    )
    # diffusion configuration file
    parser.add_argument(
        '--config_diffusion', type=str, required=True,
        help='Path to diffusion configuration file',
    )
    # arguments for sampling
    parser.add_argument(
        '--seed', type=int, default=2022,
        help='Set random seed',
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to pretrained model weights',
    )
    parser.add_argument(
        '--load_ema', type=bool, default=True,
        help='Whether to load ema weights',
    )
    parser.add_argument(
        '--skip_steps', type=int, default=None,
        help='Number of timesteps for skip sampling',
    )
    parser.add_argument(
        '--n_samples_each_class', type=int, required=True,
        help='Number of samples in each class',
    )
    parser.add_argument(
        '--guidance_scale', type=float, required=True,
        help='guidance scale. 0 for unconditional generation, '
             '1 for non-guided generation, >1 for guided generation',
    )
    parser.add_argument(
        '--ddim', action='store_true',
        help='Use DDIM deterministic sampling',
    )
    parser.add_argument(
        '--ddim_eta', type=float, default=0.0,
        help='Parameter eta in DDIM sampling',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--batch_size', type=int, default=1,
        help='Batch size. Sample by batch is much faster',
    )

    tmp_args = parser.parse_known_args()[0]
    # merge data configurations
    with open(tmp_args.config_data, 'r') as f:
        config_data = yaml.safe_load(f)
    add_args(parser, config_data, prefix='data_')
    # merge model configurations
    with open(tmp_args.config_model, 'r') as f:
        config_model = yaml.safe_load(f)
    add_args(parser, config_model, prefix='model_')
    # merge diffusion configurations
    with open(tmp_args.config_diffusion, 'r') as f:
        config_diffusion = yaml.safe_load(f)
    add_args(parser, config_diffusion, prefix='diffusion_')

    return parser.parse_args()


@torch.no_grad()
def sample():
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    num_each_device = args.n_samples_each_class // get_world_size()

    total_folds = math.ceil(num_each_device / args.batch_size)
    img_shape = (args.data_img_channels, args.data_img_size, args.data_img_size)

    if args.ddim:
        sample_fn = partial(DiffusionModel.ddim_sample, eta=args.ddim_eta)
    else:
        sample_fn = DiffusionModel.sample

    for c in range(args.data_num_classes):
        logger.info(f'Sampling class {c}')
        for i in range(total_folds):
            n = min(args.batch_size, num_each_device - i * args.batch_size)
            init_noise = torch.randn((n, *img_shape), device=device)
            X = sample_fn(
                model=model,
                class_label=c,
                init_noise=init_noise,
                guidance_scale=args.guidance_scale,
            ).clamp(-1, 1)
            for j, x in enumerate(X):
                idx = get_rank() * num_each_device + i * args.batch_size + j
                save_image(
                    tensor=x.cpu(), fp=os.path.join(args.save_dir, f'class{c}-{idx}.png'),
                    nrow=1, normalize=True, value_range=(-1, 1),
                )
            logger.info(f'Progress {(i+1)/total_folds*100:.2f}%')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    args = parse_args()

    # INITIALIZE DISTRIBUTED MODE
    init_distributed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # INITIALIZE SEEDS
    init_seeds(args.seed + get_rank())

    # INITIALIZE LOGGER
    logger = get_logger()
    logger.info(f'Device: {device}')
    logger.info(f"Number of devices: {get_world_size()}")

    # BUILD DIFFUSER
    betas = diffusions.schedule.get_beta_schedule(
        beta_schedule=args.diffusion_beta_schedule,
        total_steps=args.diffusion_total_steps,
        beta_start=args.diffusion_beta_start,
        beta_end=args.diffusion_beta_end,
    )
    if args.skip_steps is None:
        DiffusionModel = diffusions.classifier_free.ClassifierFree(
            betas=betas,
            objective=args.diffusion_objective,
            var_type=args.diffusion_var_type,
        )
    else:
        skip = args.diffusion_total_steps // args.skip_steps
        timesteps = torch.arange(0, args.diffusion_total_steps, skip)
        DiffusionModel = diffusions.classifier_free.ClassifierFreeSkip(
            timesteps=timesteps,
            betas=betas,
            objective=args.diffusion_objective,
            var_type=args.diffusion_var_type,
        )

    # BUILD MODEL
    model = build_model(args)
    model.to(device=device)
    model.eval()

    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['ema']['shadow'] if args.load_ema else ckpt['model'])
    logger.info(f'Successfully load model from {args.weights}')

    # SAMPLE
    sample()
