import os
import math
import argparse
from functools import partial
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image

import accelerate

import diffusions
from utils.logger import get_logger
from utils.misc import image_norm_to_float, instantiate_from_config, amortize


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
        '--var_type', type=str, default=None,
        help='Type of variance of the reverse process',
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
        '--text', type=str, required=True,
        help='Text description of the generated image',
    )
    parser.add_argument(
        '--guidance_weight', type=float, default=100.,
        help='Weight of CLIP guidance',
    )
    parser.add_argument(
        '--n_samples', type=int, required=True,
        help='Number of samples',
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


@torch.no_grad()
def sample():
    idx = 0
    img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
    micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))

    sample_fn = diffuser.sample
    if args.ddim:
        sample_fn = partial(diffuser.ddim_sample, eta=args.ddim_eta)

    with open(os.path.join(args.save_dir, 'description.txt'), 'w') as f:
        f.write(args.text)

    folds = amortize(args.n_samples, micro_batch * accelerator.num_processes)
    for i, bs in enumerate(folds):
        init_noise = torch.randn((micro_batch, *img_shape), device=device)
        samples = sample_fn(
            model=model, init_noise=init_noise,
            tqdm_kwargs=dict(desc=f'Fold {i}/{len(folds)}', disable=not accelerator.is_main_process)
        ).clamp(-1, 1)
        samples = accelerator.gather(samples)[:bs]
        if accelerator.is_main_process:
            for x in samples:
                x = image_norm_to_float(x).cpu()
                save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
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
        'var_type': args.var_type or cfg.diffusion.params.var_type,
        'skip_type': None if args.skip_steps is None else args.skip_type,
        'skip_steps': args.skip_steps,
        'device': device,
        'guidance_weight': args.guidance_weight,
        'clip_pretrained': 'openai/clip-vit-base-patch32',
    })
    diffuser = diffusions.CLIPGuided(**cfg.diffusion.params)

    # BUILD MODEL
    model = instantiate_from_config(cfg.model)
    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema']['shadow'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    logger.info(f'Successfully load model from {args.weights}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model = accelerator.prepare(model)
    model.eval()

    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    logger.info(f'The input text description is: {args.text}')
    diffuser.set_text(args.text)
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
