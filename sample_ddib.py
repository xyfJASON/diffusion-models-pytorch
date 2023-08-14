import os
import tqdm
import argparse
from yacs.config import CfgNode as CN

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
        '--input_dir', type=str, required=True,
        help='Input directory containing images in domain A',
    )
    parser.add_argument(
        '--weights', type=str, required=True,
        help='Path to model weights',
    )
    parser.add_argument(
        '--guidance_scale', type=float, default=2.0,
        help='Classifier-free guidance scale. 0 for unconditional generation, '
             '1 for non-guided generation, >1 for guided generation',
    )
    parser.add_argument(
        '--class_A', type=int, required=True,
        help='Class label of domain A',
    )
    parser.add_argument(
        '--class_B', type=int, required=True,
        help='Class label of domain B',
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
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=500,
        help='Batch size on each process. Sample by batch is much faster',
    )
    return parser


@torch.no_grad()
def translate():
    t = T.Compose([
        T.Resize((cfg.data.img_size, cfg.data.img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    dataset = ImageDir(root=args.input_dir, split='', transform=t)
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
        yA = torch.full((X.shape[0], ), args.class_A, device=device).long()
        yB = torch.full((X.shape[0], ), args.class_B, device=device).long()
        noise = diffuser.ddim_sample_inversion(model=model, img=X, y=yA)
        tX = diffuser.ddim_sample(model=model, init_noise=noise, y=yB)
        X = accelerator.gather_for_metrics(X)
        tX = accelerator.gather_for_metrics(tX)
        if accelerator.is_main_process:
            for x, tx in zip(X, tX):
                x = image_norm_to_float(x).cpu()
                tx = image_norm_to_float(tx).cpu()
                save_image([x, tx], os.path.join(args.save_dir, f'{idx}.png'), nrow=2)
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
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
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
    if isinstance(model, (models.UNet, models.UNetCategorialAdaGN)):
        model.load_state_dict(ckpt['ema']['shadow'])
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
    translate()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
