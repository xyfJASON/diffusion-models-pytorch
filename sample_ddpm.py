import os
import math
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
from torchvision.utils import save_image

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


def main():
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
        'var_type': args.var_type or diffusion_params.get('var_type', None),
        'skip_type': None if args.skip_steps is None else args.skip_type,
        'skip_steps': args.skip_steps,
        'device': device,
    })
    diffuser = diffusions.ddpm.DDPM(**diffusion_params)

    # BUILD MODEL
    model = instantiate_from_config(conf.model)

    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    if 'ema' in ckpt:
        model.load_state_dict(ckpt['ema']['shadow'])
    elif 'model' in ckpt:
        model.load_state_dict(ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    logger.info('=' * 19 + ' Model Info ' + '=' * 19)
    logger.info(f'Successfully load model from {args.weights}')
    logger.info('=' * 50)

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model = accelerator.prepare(model)
    model.eval()

    accelerator.wait_for_everyone()

    @torch.no_grad()
    def sample():
        idx = 0
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
        folds = amortize(args.n_samples, micro_batch * accelerator.num_processes)
        for i, bs in enumerate(folds):
            init_noise = torch.randn((micro_batch, *img_shape), device=device)
            samples = diffuser.sample(
                model=accelerator.unwrap_model(model), init_noise=init_noise,
                tqdm_kwargs=dict(desc=f'Fold {i}/{len(folds)}', disable=not accelerator.is_main_process),
            ).clamp(-1, 1)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
                    idx += 1

    @torch.no_grad()
    def sample_denoise():
        idx = 0
        freq = len(diffuser.skip_seq) // args.n_denoise
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
        folds = amortize(args.n_samples, micro_batch * accelerator.num_processes)
        for i, bs in enumerate(folds):
            init_noise = torch.randn((micro_batch, *img_shape), device=device)
            sample_loop = diffuser.sample_loop(
                model=accelerator.unwrap_model(model), init_noise=init_noise,
                tqdm_kwargs=dict(desc=f'Fold {i}/{len(folds)}', disable=not accelerator.is_main_process),
            )
            samples = [
                out['sample'] for timestep, out in enumerate(sample_loop)
                if (len(diffuser.skip_seq) - timestep - 1) % freq == 0
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
        freq = len(diffuser.skip_seq) // args.n_progressive
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
        folds = amortize(args.n_samples, micro_batch * accelerator.num_processes)
        for i, bs in enumerate(folds):
            init_noise = torch.randn((micro_batch, *img_shape), device=device)
            sample_loop = diffuser.sample_loop(
                model=accelerator.unwrap_model(model), init_noise=init_noise,
                tqdm_kwargs=dict(desc=f'Fold {i}/{len(folds)}', disable=not accelerator.is_main_process),
            )
            samples = [
                out['pred_x0'] for timestep, out in enumerate(sample_loop)
                if (len(diffuser.skip_seq) - timestep - 1) % freq == 0
            ]
            samples = torch.stack(samples, dim=1).clamp(-1, 1)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=len(x))
                    idx += 1

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


if __name__ == '__main__':
    main()
