import os
import tqdm
import argparse
from functools import partial
from yacs.config import CfgNode as CN

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

import models
import diffusions
from tools import build_model
from utils.data import get_dataset
from utils.logger import get_logger
from utils.mask import DatasetWithMask
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
        '--var_type', type=str, default=None,
        help='Type of variance of the reverse process',
    )
    parser.add_argument(
        '--skip_type', type=str, default=None,
        help='Type of skip sampling',
    )
    parser.add_argument(
        '--skip_steps', type=int, default=None,
        help='Number of timesteps for skip sampling',
    )
    parser.add_argument(
        '--resample', action='store_true',
        help='Use resample strategy proposed in RePaint paper',
    )
    parser.add_argument(
        '--resample_r', type=int, default=10,
        help='Number of resampling, as proposed in RePaint paper',
    )
    parser.add_argument(
        '--resample_j', type=int, default=10,
        help='Jump lengths of resampling, as proposed in RePaint paper',
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
        '--micro_batch', type=int, default=32,
        help='Batch size on each process. Sample by batch is much faster',
    )
    return parser


@torch.no_grad()
def sample():
    test_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='test',
        subset_ids=range(args.n_samples),
    )
    test_set = DatasetWithMask(
        dataset=test_set,
        mask_type='brush',
    )
    test_loader = DataLoader(
        dataset=test_set,
        batch_size=args.micro_batch,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    test_loader = accelerator.prepare(test_loader)  # type: ignore

    sample_fn = diffuser.sample
    if args.resample:
        sample_fn = partial(diffuser.resample, resample_r=args.resample_r, resample_j=args.resample_j)

    idx = 0
    for X, mask in tqdm.tqdm(test_loader, desc='Sampling', disable=not accelerator.is_main_process):
        init_noise = torch.randn_like(X)
        masked_image = X * mask
        diffuser.set_mask_and_image(masked_image, mask.float())
        recX = sample_fn(model=model, init_noise=init_noise).clamp(-1, 1)
        recX = accelerator.gather_for_metrics(recX)
        if accelerator.is_main_process:
            for m, x, r in zip(masked_image, X, recX):
                m = image_norm_to_float(m).cpu()
                x = image_norm_to_float(x).cpu()
                r = image_norm_to_float(r).cpu()
                save_image([m, x, r], os.path.join(args.save_dir, f'{idx}.png'), nrow=3)
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
    diffuser = diffusions.MaskGuided(
        total_steps=cfg.diffusion.total_steps,
        beta_schedule=cfg.diffusion.beta_schedule,
        beta_start=cfg.diffusion.beta_start,
        beta_end=cfg.diffusion.beta_end,
        objective=cfg.diffusion.objective,
        var_type=args.var_type if args.var_type is not None else cfg.diffusion.var_type,
        skip_type=args.skip_type if args.skip_type is not None else None,
        skip_steps=args.skip_steps,
        device=device,
    )

    # BUILD MODEL
    model = build_model(cfg, with_ema=False)

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

    accelerator.wait_for_everyone()

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
