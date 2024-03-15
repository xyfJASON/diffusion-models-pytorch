import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

import diffusions
from datasets import ImageDir
from utils.logger import get_logger
from utils.load import load_weights
from utils.misc import image_norm_to_float, instantiate_from_config, amortize


COMPATIBLE_SAMPLER_MODE = dict(
    ddpm=['sample', 'denoise', 'progressive'],
    ddim=['sample', 'denoise', 'progressive', 'interpolate', 'reconstruction'],
    euler=['sample', 'denoise', 'progressive', 'interpolate'],
)


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to inference configuration file',
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
        help='Number of samples',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--batch_size', type=int, default=500,
        help='Batch size on each process',
    )
    # arguments for all diffusers
    parser.add_argument(
        '--sampler', type=str, choices=['ddpm', 'ddim', 'euler'], default='ddpm',
        help='Type of sampler',
    )
    parser.add_argument(
        '--respace_type', type=str, default='uniform',
        help='Type of respaced timestep sequence',
    )
    parser.add_argument(
        '--respace_steps', type=int, default=None,
        help='Length of respaced timestep sequence',
    )
    # arguments for ddpm
    parser.add_argument(
        '--var_type', type=str, default=None,
        help='Type of variance of the reverse process',
    )
    # arguments for ddim
    parser.add_argument(
        '--ddim_eta', type=float, default=0.0,
        help='Parameter eta in DDIM sampling',
    )
    # sampling mode, see COMPATIBLE_SAMPLER_MODE
    parser.add_argument(
        '--mode', type=str, default='sample', choices=[
            'sample', 'denoise', 'progressive', 'interpolate', 'reconstruction',
        ],
        help='Choose a sample mode',
    )
    # arguments for mode == 'denoise'
    parser.add_argument(
        '--n_denoise', type=int, default=20,
        help='Number of intermediate images when mode is denoise',
    )
    # arguments for mode == 'progressive'
    parser.add_argument(
        '--n_progressive', type=int, default=20,
        help='Number of intermediate images when mode is progressive',
    )
    # arguments for mode == 'interpolate'
    parser.add_argument(
        '--n_interpolate', type=int, default=16,
        help='Number of intermidiate images when mode is interpolate',
    )
    # arguments for mode == 'reconstruction':
    parser.add_argument(
        '--input_dir', type=str, required=False,
        help='Path to the directory containing input images',
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
    if args.sampler == 'ddpm':
        diffuser = diffusions.ddpm.DDPM(
            total_steps=conf.diffusion.params.total_steps,
            beta_schedule=conf.diffusion.params.beta_schedule,
            beta_start=conf.diffusion.params.beta_start,
            beta_end=conf.diffusion.params.beta_end,
            objective=conf.diffusion.params.objective,
            var_type=args.var_type or conf.diffusion.params.get('var_type', None),
            respace_type=None if args.respace_steps is None else args.respace_type,
            respace_steps=args.respace_steps or conf.diffusion.params.total_steps,
            device=device,
        )
    elif args.sampler == 'ddim':
        diffuser = diffusions.ddim.DDIM(
            total_steps=conf.diffusion.params.total_steps,
            beta_schedule=conf.diffusion.params.beta_schedule,
            beta_start=conf.diffusion.params.beta_start,
            beta_end=conf.diffusion.params.beta_end,
            objective=conf.diffusion.params.objective,
            respace_type=None if args.respace_steps is None else args.respace_type,
            respace_steps=args.respace_steps or conf.diffusion.params.total_steps,
            eta=args.ddim_eta,
            device=device,
        )
    elif args.sampler == 'euler':
        diffuser = diffusions.euler.EulerSampler(
            total_steps=conf.diffusion.params.total_steps,
            beta_schedule=conf.diffusion.params.beta_schedule,
            beta_start=conf.diffusion.params.beta_start,
            beta_end=conf.diffusion.params.beta_end,
            objective=conf.diffusion.params.objective,
            respace_type=None if args.respace_steps is None else args.respace_type,
            respace_steps=args.respace_steps or conf.diffusion.params.total_steps,
            device=device,
        )
    else:
        raise ValueError(f'Unknown sampler: {args.sampler}')

    # BUILD MODEL
    model = instantiate_from_config(conf.model)

    # LOAD WEIGHTS
    weights = load_weights(args.weights)
    model.load_state_dict(weights)
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
        bspp = min(args.batch_size, math.ceil(args.n_samples / accelerator.num_processes))
        folds = amortize(args.n_samples, bspp * accelerator.num_processes)
        for i, bs in enumerate(folds):
            init_noise = torch.randn((bspp, *img_shape), device=device)
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
        freq = len(diffuser.respaced_seq) // args.n_denoise
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        bspp = min(args.batch_size, math.ceil(args.n_samples / accelerator.num_processes))
        folds = amortize(args.n_samples, bspp * accelerator.num_processes)
        for i, bs in enumerate(folds):
            init_noise = torch.randn((bspp, *img_shape), device=device)
            sample_loop = diffuser.sample_loop(
                model=accelerator.unwrap_model(model), init_noise=init_noise,
                tqdm_kwargs=dict(desc=f'Fold {i}/{len(folds)}', disable=not accelerator.is_main_process),
            )
            samples = [
                out['sample'] for timestep, out in enumerate(sample_loop)
                if (len(diffuser.respaced_seq) - timestep - 1) % freq == 0
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
        freq = len(diffuser.respaced_seq) // args.n_progressive
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        bspp = min(args.batch_size, math.ceil(args.n_samples / accelerator.num_processes))
        folds = amortize(args.n_samples, bspp * accelerator.num_processes)
        for i, bs in enumerate(folds):
            init_noise = torch.randn((bspp, *img_shape), device=device)
            sample_loop = diffuser.sample_loop(
                model=accelerator.unwrap_model(model), init_noise=init_noise,
                tqdm_kwargs=dict(desc=f'Fold {i}/{len(folds)}', disable=not accelerator.is_main_process),
            )
            samples = [
                out['pred_x0'] for timestep, out in enumerate(sample_loop)
                if (len(diffuser.respaced_seq) - timestep - 1) % freq == 0
            ]
            samples = torch.stack(samples, dim=1).clamp(-1, 1)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=len(x))
                    idx += 1

    @torch.no_grad()
    def sample_interpolate():
        idx = 0
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        bspp = min(args.batch_size, math.ceil(args.n_samples / accelerator.num_processes))

        def slerp(t, z1, z2):  # noqa
            theta = torch.acos(torch.sum(z1 * z2) / (torch.linalg.norm(z1) * torch.linalg.norm(z2)))
            return torch.sin((1 - t) * theta) / torch.sin(theta) * z1 + torch.sin(t * theta) / torch.sin(theta) * z2

        folds = amortize(args.n_samples, bspp * accelerator.num_processes)
        for i, bs in enumerate(folds):
            z1 = torch.randn((bspp, *img_shape), device=device)
            z2 = torch.randn((bspp, *img_shape), device=device)
            samples = torch.stack([
                diffuser.sample(
                    model=accelerator.unwrap_model(model), init_noise=slerp(t, z1, z2),
                    tqdm_kwargs=dict(
                        desc=f'Fold {i}/{len(folds)}, interp {j}/{args.n_interpolate}',
                        disable=not accelerator.is_main_process,
                    ),
                ).clamp(-1, 1)
                for j, t in enumerate(torch.linspace(0, 1, args.n_interpolate))
            ], dim=1)
            samples = accelerator.gather(samples)[:bs]
            if accelerator.is_main_process:
                for x in samples:
                    x = image_norm_to_float(x).cpu()
                    save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=len(x))
                    idx += 1

    @torch.no_grad()
    def sample_reconstruction():
        # build dataset
        transforms = T.Compose([
            T.Resize(conf.data.params.img_size),
            T.CenterCrop(conf.data.params.img_size),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = ImageDir(root=args.input_dir, transform=transforms)
        if args.n_samples < len(dataset):
            dataset = Subset(dataset=dataset, indices=torch.arange(args.n_samples))
        bspp = min(args.batch_size, math.ceil(len(dataset) / accelerator.num_processes))
        dataloader = DataLoader(dataset=dataset, batch_size=bspp, num_workers=4, pin_memory=True, prefetch_factor=2)
        dataloader = accelerator.prepare(dataloader)  # type: ignore
        # sampling
        idx = 0
        for i, X in enumerate(dataloader):
            X = X[0].float() if isinstance(X, (tuple, list)) else X.float()
            noise = diffuser.sample_inversion(
                model=accelerator.unwrap_model(model), img=X,
                tqdm_kwargs=dict(desc=f'img2noise {i}/{len(dataloader)}', disable=not accelerator.is_main_process),
            )
            recX = diffuser.sample(
                model=accelerator.unwrap_model(model), init_noise=noise,
                tqdm_kwargs=dict(desc=f'noise2img {i}/{len(dataloader)}', disable=not accelerator.is_main_process),
            )
            X = accelerator.gather_for_metrics(X)
            recX = accelerator.gather_for_metrics(recX)
            if accelerator.is_main_process:
                for x, r in zip(X, recX):
                    x = image_norm_to_float(x).cpu()
                    r = image_norm_to_float(r).cpu()
                    save_image([x, r], os.path.join(args.save_dir, f'{idx}.png'), nrow=2)
                    idx += 1

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    compatible_mode = COMPATIBLE_SAMPLER_MODE[args.sampler]
    if args.mode not in compatible_mode:
        logger.warning(
            f'`{args.mode}` mode is not designed for `{args.sampler}` sampler, '
            f'unexpected behavior may occur.'
        )

    if args.mode == 'sample':
        sample()
    elif args.mode == 'denoise':
        sample_denoise()
    elif args.mode == 'progressive':
        sample_progressive()
    elif args.mode == 'interpolate':
        sample_interpolate()
    elif args.mode == 'reconstruction':
        if args.input_dir is None:
            raise ValueError('input_dir is required for mode `reconstruction`')
        sample_reconstruction()
    else:
        raise ValueError(f'Unknown mode: {args.mode}')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
