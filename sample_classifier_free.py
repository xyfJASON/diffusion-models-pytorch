import os
import tqdm
import math
import argparse
from functools import partial
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image

import accelerate

import diffusions
from utils.logger import get_logger
from utils.misc import image_norm_to_float, instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
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
        '--skip_type', type=str, default='uniform',
        help='Type of skip sampling',
    )
    parser.add_argument(
        '--skip_steps', type=int, default=None,
        help='Number of timesteps for skip sampling',
    )
    parser.add_argument(
        '--guidance_scale', type=float, required=True,
        help='guidance scale. 0 for unconditional generation, '
             '1 for non-guided generation, >1 for guided generation',
    )
    parser.add_argument(
        '--n_samples_each_class', type=int, required=True,
        help='Number of samples in each class',
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
        '--micro_batch', type=int, default=500,
        help='Batch size on each process. Sample by batch is much faster',
    )
    return parser


def amortize(n_samples: int, batch_size: int):
    k = n_samples // batch_size
    r = n_samples % batch_size
    return k * [batch_size] if r == 0 else k * [batch_size] + [r]


@torch.no_grad()
def sample():
    img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
    micro_batch = min(args.micro_batch, math.ceil(args.n_samples_each_class / accelerator.num_processes))
    batch_size = micro_batch * accelerator.num_processes

    sample_fn = diffuser.sample
    if args.ddim:
        sample_fn = partial(diffuser.ddim_sample, eta=args.ddim_eta)

    for c in range(cfg.data.num_classes):
        idx = 0
        logger.info(f'Sampling class {c}')
        for bs in tqdm.tqdm(
                amortize(args.n_samples_each_class, batch_size), desc='Sampling',
                disable=not accelerator.is_main_process,
        ):
            init_noise = torch.randn((bs, *img_shape), device=device)
            labels = torch.full((bs, ), fill_value=c, device=device)
            samples = sample_fn(model=model, init_noise=init_noise, y=labels).clamp(-1, 1)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'class{c}', f'{idx}.png'), nrow=1)
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
    cfg.diffusion.params.update({
        'skip_type': None if args.skip_steps is None else args.skip_type,
        'skip_steps': args.skip_steps,
        'guidance_scale': args.guidance_scale,
        'device': device,
    })
    diffuser = diffusions.classifier_free.ClassifierFree(**cfg.diffusion.params)

    # BUILD MODEL
    model = instantiate_from_config(cfg.model)
    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    model.load_state_dict(ckpt['ema']['shadow'] if args.load_ema else ckpt['model'])
    logger.info(f'Successfully load model from {args.weights}')
    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model = accelerator.prepare(model)
    model.eval()

    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    for i in range(cfg.data.num_classes):
        os.makedirs(os.path.join(args.save_dir, f'class{i}'), exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
