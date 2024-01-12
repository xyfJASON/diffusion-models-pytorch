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
from utils.misc import image_norm_to_float, instantiate_from_config, amortize


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '--seed', type=int, default=2001,
        help='Set random seed',
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='path to model weights',
    )
    parser.add_argument(
        '--ddim_eta', type=float, default=0.0,
        help='Parameter eta in DDIM sampling',
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
        '--mode', type=str, default='sample', choices=['sample', 'interpolate', 'reconstruction'],
        help='Choose a sample mode',
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
    diffusion_params = OmegaConf.to_container(conf.diffusion.params)
    diffusion_params.update({
        'respace_type': None if args.respace_steps is None else args.respace_type,
        'respace_steps': args.respace_steps,
        'eta': args.ddim_eta,
        'device': device,
    })
    diffuser = diffusions.ddim.DDIM(**diffusion_params)

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
    def sample_interpolate():
        idx = 0
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))

        def slerp(t, z1, z2):  # noqa
            theta = torch.acos(torch.sum(z1 * z2) / (torch.linalg.norm(z1) * torch.linalg.norm(z2)))
            return torch.sin((1 - t) * theta) / torch.sin(theta) * z1 + torch.sin(t * theta) / torch.sin(theta) * z2

        folds = amortize(args.n_samples, micro_batch * accelerator.num_processes)
        for i, bs in enumerate(folds):
            z1 = torch.randn((micro_batch, *img_shape), device=device)
            z2 = torch.randn((micro_batch, *img_shape), device=device)
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
        if args.input_dir is None:
            raise ValueError('input_dir must be specified when mode is reconstruction')
        transforms = T.Compose([
            T.Resize(conf.data.params.img_size),
            T.CenterCrop(conf.data.params.img_size),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = ImageDir(root=args.input_dir, transform=transforms)
        if args.n_samples < len(dataset):
            dataset = Subset(dataset=dataset, indices=torch.arange(args.n_samples))
        micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
        dataloader = DataLoader(dataset=dataset, batch_size=micro_batch, **conf.dataloader)
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
    if args.mode == 'sample':
        sample()
    elif args.mode == 'interpolate':
        sample_interpolate()
    elif args.mode == 'reconstruction':
        sample_reconstruction()
    else:
        raise ValueError
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
