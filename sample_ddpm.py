import os
import tqdm
import math
import argparse
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image

import accelerate

import models
import diffusions
from tools import build_model
from utils.logger import get_logger
from utils.misc import image_norm_to_float


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    # arguments related to sampling
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
        '--n_samples', type=int, required=True,
        help='Number of samples',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=500,
        help='Batch size on each process. Sample by batch is much faster',
    )
    parser.add_argument(
        '--mode', type=str, default='sample', choices=['sample', 'denoise', 'progressive'],
        help='Choose a sample mode',
    )
    parser.add_argument(
        '--n_denoise', type=int, default=20,
        help='Number of intermediate images when mode is denoise',
    )
    parser.add_argument(
        '--n_progressive', type=int, default=20,
        help='Number of intermediate images when mode is progressive',
    )
    return parser


def amortize(n_samples: int, batch_size: int):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


@torch.no_grad()
def sample():
    idx = 0
    img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
    micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
    batch_size = micro_batch * accelerator.num_processes
    for bs in tqdm.tqdm(amortize(args.n_samples, batch_size), desc='Sampling',
                        disable=not accelerator.is_main_process):
        init_noise = torch.randn((micro_batch, *img_shape), device=device)
        samples = diffuser.sample(model=model, init_noise=init_noise).clamp(-1, 1)
        samples = accelerator.gather(samples)[:bs]
        if accelerator.is_main_process:
            for x in samples:
                x = image_norm_to_float(x).cpu()
                save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
                idx += 1


@torch.no_grad()
def sample_denoise():
    idx = 0
    freq = diffuser.total_steps // args.n_denoise
    img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
    micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
    batch_size = micro_batch * accelerator.num_processes
    for bs in tqdm.tqdm(amortize(args.n_samples, batch_size), desc='Sampling',
                        disable=not accelerator.is_main_process):
        init_noise = torch.randn((micro_batch, *img_shape), device=device)
        sample_loop = diffuser.sample_loop(model=model, init_noise=init_noise)
        samples = [
            out['sample'] for timestep, out in enumerate(sample_loop)
            if (diffuser.total_steps - timestep - 1) % freq == 0
        ]
        samples = torch.stack(samples, dim=1).clamp(-1, 1)
        samples = accelerator.gather(samples)[:bs]
        if accelerator.is_main_process:
            for x in samples:
                x = image_norm_to_float(x).cpu()
                save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=len(x))
                idx += 1


@torch.no_grad()
def sample_progressive():
    idx = 0
    freq = diffuser.total_steps // args.n_progressive
    img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
    micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
    batch_size = micro_batch * accelerator.num_processes
    for bs in tqdm.tqdm(amortize(args.n_samples, batch_size), desc='Sampling',
                        disable=not accelerator.is_main_process):
        init_noise = torch.randn((micro_batch, *img_shape), device=device)
        sample_loop = diffuser.sample_loop(model=model, init_noise=init_noise)
        samples = [
            out['pred_X0'] for timestep, out in enumerate(sample_loop)
            if (diffuser.total_steps - timestep - 1) % freq == 0
        ]
        samples = torch.stack(samples, dim=1).clamp(-1, 1)
        samples = accelerator.gather(samples)[:bs]
        if accelerator.is_main_process:
            for x in samples:
                x = image_norm_to_float(x).cpu()
                save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=len(x))
                idx += 1


if __name__ == '__main__':
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # INITIALIZE LOGGER
    logger = get_logger(
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(args.seed, device_specific=True)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DIFFUSER
    betas = diffusions.get_beta_schedule(
        beta_schedule=cfg.diffusion.beta_schedule,
        total_steps=cfg.diffusion.total_steps,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
    )
    if args.skip_steps is None:
        diffuser = diffusions.ddpm.DDPM(
            betas=betas,
            objective=cfg.diffusion.objective,
            var_type=cfg.diffusion.var_type,
        )
    else:
        skip = cfg.diffusion.total_steps // args.skip_steps
        timesteps = torch.arange(0, cfg.diffusion.total_steps, skip)
        diffuser = diffusions.ddpm.DDPMSkip(
            timesteps=timesteps,
            betas=betas,
            objective=cfg.diffusion.objective,
            var_type=cfg.diffusion.var_type,
        )
    # BUILD MODEL
    model = build_model(cfg, with_ema=False)
    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    if isinstance(model, (models.UNet, models.UNetConditional)):
        model.load_state_dict(ckpt['ema']['shadow'] if args.load_ema else ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    logger.info(f'Successfully load model from {args.weights}')
    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model = accelerator.prepare(model)
    model.eval()

    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    if args.mode == 'sample':
        sample()
    elif args.mode == 'denoise':
        sample_denoise()
    elif args.mode == 'progressive':
        sample_progressive()
    else:
        raise ValueError
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
