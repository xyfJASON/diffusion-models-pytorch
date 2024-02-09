import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
from torchvision.utils import save_image

import diffusions
from utils.logger import get_logger
from utils.load import load_weights
from utils.misc import image_norm_to_float, instantiate_from_config, amortize


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '--seed', type=int, default=2022,
        help='Set random seed',
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to pretrained model weights',
    )
    parser.add_argument(
        '--n_samples', type=int, required=True,
        help='Number of samples to generate',
    )
    parser.add_argument(
        '-H', '--height', type=int, default=512,
        help='Height of generated images',
    )
    parser.add_argument(
        '-W', '--width', type=int, default=512,
        help='Width of generated images',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--respace_type', type=str, default='uniform',
        help='Type of respaced timestep sequence',
    )
    parser.add_argument(
        '--respace_steps', type=int, default=None,
        help='Length of respaced timestep sequence',
    )
    parser.add_argument(
        '--cfg', type=float, default=7.0,
        help='Classifier-Free Guidance scale. 0 for unconditional generation, '
             '1 for non-guided generation, >1 for guided generation',
    )
    parser.add_argument(
        '--prompt', type=str, default='',
        help='Text prompt for text-to-image generation',
    )
    parser.add_argument(
        '--ddim', action='store_true',
        help='Use DDIM deterministic sampling. Otherwise use DDPM sampling',
    )
    parser.add_argument(
        '--ddim_eta', type=float, default=0.0,
        help='Parameter eta in DDIM sampling',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=1,
        help='Batch size on each process',
    )
    return parser


if __name__ == '__main__':
    # PARSE ARGS AND CONFIGS
    args, unknown_args = get_parser().parse_known_args()
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

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
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DIFFUSER
    diffusion_params = OmegaConf.to_container(conf.diffusion.params)
    diffusion_params.update({
        'cond_kwarg': 'context',
        'respace_type': None if args.respace_steps is None else args.respace_type,
        'respace_steps': args.respace_steps,
        'guidance_scale': args.cfg,
        'device': device,
    })
    if args.ddim:
        diffusion_params.update({'eta': args.ddim_eta})
        diffuser = diffusions.DDIMCFG(**diffusion_params)
    else:
        diffuser = diffusions.DDPMCFG(**diffusion_params)

    # BUILD MODEL
    model = instantiate_from_config(conf.model)
    model.eval()

    # LOAD WEIGHTS
    weights = load_weights(args.weights)
    model.load_state_dict(weights)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load model from {args.weights}')
    logger.info('=' * 50)

    model.to(device)

    accelerator.wait_for_everyone()

    @torch.no_grad()
    def sample():
        idx = 0
        micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
        batch_size = micro_batch * accelerator.num_processes
        folds = amortize(args.n_samples, batch_size)
        for i, bs in enumerate(folds):
            init_noise = torch.randn((micro_batch, 4, args.height // 8, args.width // 8), device=device)
            text_embed = model.text_encoder_encode([args.prompt] * micro_batch)
            empty_embed = model.text_encoder_encode([''] * micro_batch)
            latents = diffuser.sample(
                model=model.unet_forward, init_noise=init_noise,
                uncond_conditioning=empty_embed,
                model_kwargs=dict(context=text_embed),
                tqdm_kwargs=dict(desc=f'Fold {i}/{len(folds)}', disable=not accelerator.is_main_process),
            )
            samples = model.autoencoder_decode(latents).clamp(-1, 1)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
                    idx += 1

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
