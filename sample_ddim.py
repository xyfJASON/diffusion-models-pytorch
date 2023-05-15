import os
import tqdm
import math
import argparse
from yacs.config import CfgNode as CN

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

import accelerate

import models
import diffusions
from datasets import ImageDir
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
        '--seed', type=int, default=2001,
        help='Set random seed',
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='path to model weights',
    )
    parser.add_argument(
        '--load_ema', type=bool, default=True,
        help='Whether to load ema weights',
    )
    parser.add_argument(
        '--ddim_eta', type=float, default=0.0,
        help='Parameter eta in DDIM sampling',
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
    for bs in tqdm.tqdm(
            amortize(args.n_samples, batch_size), desc='Sampling',
            disable=not accelerator.is_main_process,
    ):
        init_noise = torch.randn((micro_batch, *img_shape), device=device)
        samples = diffuser.ddim_sample(model=model, init_noise=init_noise).clamp(-1, 1)
        samples = accelerator.gather(samples)[:bs]
        if accelerator.is_main_process:
            for x in samples:
                x = image_norm_to_float(x).cpu()
                save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=1)
                idx += 1


@torch.no_grad()
def sample_interpolate():
    idx = 0
    img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
    micro_batch = min(args.micro_batch, math.ceil(args.n_samples / accelerator.num_processes))
    batch_size = micro_batch * accelerator.num_processes

    def slerp(t, z1, z2):  # noqa
        theta = torch.acos(torch.sum(z1 * z2) / (torch.linalg.norm(z1) * torch.linalg.norm(z2)))
        return torch.sin((1 - t) * theta) / torch.sin(theta) * z1 + torch.sin(t * theta) / torch.sin(theta) * z2

    for bs in tqdm.tqdm(
            amortize(args.n_samples, batch_size), desc='Sampling',
            disable=not accelerator.is_main_process,
    ):
        z1 = torch.randn((micro_batch, *img_shape), device=device)
        z2 = torch.randn((micro_batch, *img_shape), device=device)
        samples = torch.stack([
            diffuser.ddim_sample(model=model, init_noise=slerp(t, z1, z2)).clamp(-1, 1)
            for t in torch.linspace(0, 1, args.n_interpolate)
        ], dim=1)
        samples = accelerator.gather(samples)[:bs]
        if accelerator.is_main_process:
            for x in samples:
                x = image_norm_to_float(x).cpu()
                save_image(x, os.path.join(args.save_dir, f'{idx}.png'), nrow=len(x))
                idx += 1


@torch.no_grad()
def sample_reconstruction(dataset):
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.micro_batch,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    dataloader = accelerator.prepare(dataloader)  # type: ignore
    idx = 0
    for X in tqdm.tqdm(dataloader, desc='Sampling', disable=not accelerator.is_main_process):
        X = X[0].float() if isinstance(X, (tuple, list)) else X.float()
        noise = diffuser.ddim_sample_inversion(model=model, img=X)
        recX = diffuser.ddim_sample(model=model, init_noise=noise)
        X = accelerator.gather_for_metrics(X)
        recX = accelerator.gather_for_metrics(recX)
        if accelerator.is_main_process:
            for x, r in zip(X, recX):
                x = image_norm_to_float(x).cpu()
                r = image_norm_to_float(r).cpu()
                save_image([x, r], os.path.join(args.save_dir, f'{idx}.png'), nrow=2)
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
        'eta': args.ddim_eta,
        'device': device,
    })
    cfg.diffusion.params.pop('var_type')
    diffuser = diffusions.ddim.DDIM(**cfg.diffusion.params)

    # BUILD MODEL
    model = instantiate_from_config(cfg.model)
    # LOAD WEIGHTS
    ckpt = torch.load(args.weights, map_location='cpu')
    if isinstance(model, (models.UNet, models.UNetCategorialAdaGN)):
        model.load_state_dict(ckpt['ema']['shadow'] if args.load_ema else ckpt['model'])
    else:
        model.load_state_dict(ckpt)
    logger.info(f'Successfully load model from {args.weights}')
    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model = accelerator.prepare(model)
    model.eval()

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    if args.mode == 'sample':
        sample()
    elif args.mode == 'interpolate':
        sample_interpolate()
    elif args.mode == 'reconstruction':
        assert args.input_dir is not None
        transforms = T.Compose([
            T.Resize(cfg.data.img_size),
            T.CenterCrop(cfg.data.img_size),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dset = ImageDir(root=args.input_dir, split='', transform=transforms)
        if args.n_samples < len(dset):
            dset = Subset(dataset=dset, indices=torch.arange(args.n_samples))
        sample_reconstruction(dataset=dset)
    else:
        raise ValueError
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
