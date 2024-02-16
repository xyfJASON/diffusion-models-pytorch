import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import math
import argparse
from functools import partial
from omegaconf import OmegaConf

import torch
import accelerate
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import diffusions
from datasets import ImageDir
from utils.logger import get_logger
from utils.load import load_weights
from utils.mask import DatasetWithMask
from utils.misc import image_norm_to_float, instantiate_from_config


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
        '--var_type', type=str, default=None,
        help='Type of variance of the reverse process',
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
        '--input_dir', type=str, required=True,
        help='Path to the directory containing input images',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--batch_size', type=int, default=32,
        help='Batch size on each process',
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
        'respace_type': None if args.respace_steps is None else args.respace_type,
        'respace_steps': args.respace_steps,
        'device': device,
    })
    diffuser = diffusions.MaskGuidance(**diffusion_params)

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
        # build dataset
        if args.input_dir is None:
            raise ValueError('input_dir is required')
        transforms = T.Compose([
            T.Resize(conf.data.params.img_size),
            T.CenterCrop(conf.data.params.img_size),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = ImageDir(root=args.input_dir, transform=transforms)
        dataset = DatasetWithMask(dataset=dataset, mask_type='brush')
        bspp = min(args.batch_size, math.ceil(len(dataset) / accelerator.num_processes))
        dataloader = DataLoader(dataset=dataset, batch_size=bspp, num_workers=4, pin_memory=True, prefetch_factor=2)
        dataloader = accelerator.prepare(dataloader)  # type: ignore
        logger.info(f'Found {len(dataset)} images in {args.input_dir}')
        # sampling
        idx = 0
        sample_fn = diffuser.sample
        if args.resample:
            sample_fn = partial(diffuser.resample, resample_r=args.resample_r, resample_j=args.resample_j)
        for i, (X, mask) in enumerate(dataloader):
            init_noise = torch.randn_like(X)
            masked_image = X * mask
            diffuser.set_mask_and_image(masked_image, mask.float())
            recX = sample_fn(
                model=accelerator.unwrap_model(model), init_noise=init_noise,
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

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
