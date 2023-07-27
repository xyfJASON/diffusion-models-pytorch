import os
import argparse
from functools import partial
from yacs.config import CfgNode as CN

import torch
import torchvision.transforms as T
from torchvision.utils import save_image
from torch.utils.data import DataLoader, Subset

import accelerate

import diffusions
from datasets import ImageDir
from utils.logger import get_logger
from utils.mask import DatasetWithMask
from utils.misc import image_norm_to_float, instantiate_from_config


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
        '--input_dir', type=str, required=True,
        help='Path to the directory containing input images',
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
def sample(dataset):
    dataset = DatasetWithMask(
        dataset=dataset,
        mask_type='brush',
    )
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.micro_batch,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    dataloader = accelerator.prepare(dataloader)  # type: ignore

    sample_fn = diffuser.sample
    if args.resample:
        sample_fn = partial(diffuser.resample, resample_r=args.resample_r, resample_j=args.resample_j)

    idx = 0
    for i, (X, mask) in enumerate(dataloader):
        init_noise = torch.randn_like(X)
        masked_image = X * mask
        diffuser.set_mask_and_image(masked_image, mask.float())
        recX = sample_fn(
            model=model, init_noise=init_noise,
            tqdm_kwargs=dict(desc=f'Fold {i}/{len(dataloader)}', disable=not accelerator.is_main_process),
        ).clamp(-1, 1)
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
    cfg.diffusion.params.update({
        'var_type': args.var_type or cfg.diffusion.params.var_type,
        'skip_type': None if args.skip_steps is None else args.skip_type,
        'skip_steps': args.skip_steps,
        'device': device,
    })
    diffuser = diffusions.MaskGuided(**cfg.diffusion.params)

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
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    transforms = T.Compose([
        T.Resize(cfg.data.img_size),
        T.CenterCrop(cfg.data.img_size),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dset = ImageDir(root=args.input_dir, split='', transform=transforms)
    if args.n_samples < len(dset):
        dset = Subset(dataset=dset, indices=torch.arange(args.n_samples))
    sample(dataset=dset)
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
