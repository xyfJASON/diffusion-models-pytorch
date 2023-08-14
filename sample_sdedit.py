import os
import tqdm
import argparse
from yacs.config import CfgNode as CN

import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

import diffusions
from datasets import ImageDir
from utils.logger import get_logger
from utils.misc import instantiate_from_config


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
        '--edit_steps', type=int, required=True,
        help='Number of timesteps to add noise and denoise',
    )
    parser.add_argument(
        '--input_dir', type=str, required=True,
        help='Path to the input directory',
    )
    parser.add_argument(
        '--save_dir', type=str, required=True,
        help='Path to the directory to save samples',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=500,
        help='Batch size on each process. Sample by batch is much faster',
    )
    return parser


@torch.no_grad()
def sample():
    transforms = T.Compose([
        T.Resize((cfg.data.img_size, cfg.data.img_size)),
        T.ToTensor(),
        T.Normalize([0.5] * cfg.data.img_channels, [0.5] * cfg.data.img_channels),
    ])
    dataset = ImageDir(root=args.input_dir, split='', transform=transforms)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=args.micro_batch,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    dataloader = accelerator.prepare(dataloader)  # type: ignore
    logger.info(f'Found {len(dataset)} images in {args.input_dir}')

    idx = 0
    assert 0 <= args.edit_steps < len(diffuser.skip_seq)
    time_seq = diffuser.skip_seq[:args.edit_steps].tolist()
    time_seq_prev = [-1] + time_seq[:-1]
    for i, img in enumerate(dataloader):
        img = img.float()
        noised_img = diffuser.q_sample(x0=img, t=time_seq[-1])
        edited_img = noised_img.clone()
        pbar = tqdm.tqdm(
            total=len(time_seq), desc=f'Fold {i}/{len(dataloader)}',
            disable=not accelerator.is_main_process,
        )
        for t, t_prev in zip(reversed(time_seq), reversed(time_seq_prev)):
            t_batch = torch.full((edited_img.shape[0], ), t, device=device)
            model_output = accelerator.unwrap_model(model)(edited_img, t_batch)
            out = diffuser.p_sample(model_output, edited_img, t, t_prev)
            edited_img = out['sample']
            pbar.update(1)
        pbar.close()
        edited_img.clamp_(-1, 1)
        img = accelerator.gather_for_metrics(img)
        noised_img = accelerator.gather_for_metrics(noised_img)
        edited_img = accelerator.gather_for_metrics(edited_img)
        if accelerator.is_main_process:
            for im, nim, eim in zip(img, noised_img, edited_img):
                save_image(
                    [im, nim, eim], os.path.join(args.save_dir, f'{idx}.png'),
                    nrow=3, normalize=True, value_range=(-1, 1),
                )
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
    diffuser = diffusions.ddpm.DDPM(**cfg.diffusion.params)

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
    sample()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')
