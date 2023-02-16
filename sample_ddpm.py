import os
import math
import argparse
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image

import diffusions.ddpm
import diffusions.schedule
from utils.misc import init_seeds
from utils.logger import get_logger
from engine.tools import build_model
from utils.dist import init_distributed_mode, get_rank, get_world_size


def get_parser():
    parser = argparse.ArgumentParser()
    # load basic configuration from file, including parameter of data, model and diffuser
    parser.add_argument('-c', '--config', metavar='FILE', required=True,
                        help='path to configuration file')
    # arguments for sampling
    parser.add_argument('--model_path', type=str, required=True, help='path to model weights')
    parser.add_argument('--load_ema', action='store_true', help='whether to load ema weights')
    parser.add_argument('--skip_steps', type=int, help='number of timesteps for skip sampling')
    parser.add_argument('--n_samples', type=int, required=True, help='number of samples')
    parser.add_argument('--save_dir', type=str, required=True, help='path to directory saving samples')
    parser.add_argument('--batch_size', type=int, default=1, help='sample by batch is faster')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    parser.add_argument('--mode', type=str, default='sample',
                        choices=['sample', 'denoise', 'progressive'],
                        help='specify a sample mode')
    parser.add_argument('--n_denoise', type=int, default=20, help='only valid when mode is denoise')
    parser.add_argument('--n_progressive', type=int, default=20, help='only valid when mode is progressive')
    # modify basic configuration in yaml file
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER,
                        help="modify config options using the command-line 'KEY VALUE' pairs")
    return parser


@torch.no_grad()
def sample():
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    num_each_device = args.n_samples // get_world_size()

    total_folds = math.ceil(num_each_device / args.batch_size)
    img_shape = (cfg.DATA.IMG_CHANNELS, cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
    for i in range(total_folds):
        n = min(args.batch_size, num_each_device - i * args.batch_size)
        init_noise = torch.randn((n, *img_shape), device=device)
        X = DiffusionModel.sample(model=model, init_noise=init_noise).clamp(-1, 1)
        for j, x in enumerate(X):
            idx = get_rank() * num_each_device + i * args.batch_size + j
            save_image(
                tensor=x.cpu(), fp=os.path.join(args.save_dir, f'{idx}.png'),
                nrow=1, normalize=True, value_range=(-1, 1),
            )
        logger.info(f'Progress {(i+1)/total_folds*100:.2f}%')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


@torch.no_grad()
def sample_denoise():
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    num_each_device = args.n_samples // get_world_size()

    total_folds = math.ceil(num_each_device / args.batch_size)
    img_shape = (cfg.DATA.IMG_CHANNELS, cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
    freq = DiffusionModel.total_steps // args.n_denoise
    for i in range(total_folds):
        n = min(args.batch_size, num_each_device - i * args.batch_size)
        init_noise = torch.randn((n, *img_shape), device=device)
        sample_loop = DiffusionModel.sample_loop(model=model, init_noise=init_noise)
        X = [out['sample'] for timestep, out in enumerate(sample_loop)
             if (DiffusionModel.total_steps - timestep - 1) % freq == 0]
        X = torch.stack(X, dim=1).clamp(-1, 1)
        for j, x in enumerate(X):
            idx = get_rank() * num_each_device + i * args.batch_size + j
            save_image(
                tensor=x.cpu(), fp=os.path.join(args.save_dir, f'{idx}.png'),
                nrow=len(x), normalize=True, value_range=(-1, 1),
            )
        logger.info(f'Progress {(i+1)/total_folds*100:.2f}%')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


@torch.no_grad()
def sample_progressive():
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    num_each_device = args.n_samples // get_world_size()

    total_folds = math.ceil(num_each_device / args.batch_size)
    img_shape = (cfg.DATA.IMG_CHANNELS, cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
    freq = DiffusionModel.total_steps // args.n_progressive
    for i in range(total_folds):
        n = min(args.batch_size, num_each_device - i * args.batch_size)
        init_noise = torch.randn((n, *img_shape), device=device)
        sample_generator = DiffusionModel.sample_loop(model=model, init_noise=init_noise)
        X = [out['pred_X0'] for timestep, out in enumerate(sample_generator)
             if (DiffusionModel.total_steps - timestep - 1) % freq == 0]
        X = torch.stack(X, dim=1).clamp(-1, 1)
        for j, x in enumerate(X):
            idx = get_rank() * num_each_device + i * args.batch_size + j
            save_image(
                tensor=x.cpu(), fp=os.path.join(args.save_dir, f'{idx}.png'),
                nrow=len(x), normalize=True, value_range=(-1, 1),
            )
        logger.info(f'Progress {(i+1)/total_folds*100:.2f}%')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    # PARSE ARGS AND CONFIGS
    args = get_parser().parse_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    # INITIALIZE DISTRIBUTED MODE
    init_distributed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # INITIALIZE SEEDS
    init_seeds(args.seed + get_rank())

    # INITIALIZE LOGGER
    logger = get_logger()
    logger.info(f'Device: {device}')
    logger.info(f"Number of devices: {get_world_size()}")

    # BUILD DIFFUSER
    betas = diffusions.schedule.get_beta_schedule(
        beta_schedule=cfg.DDPM.BETA_SCHEDULE,
        total_steps=cfg.DDPM.TOTAL_STEPS,
        beta_start=cfg.DDPM.BETA_START,
        beta_end=cfg.DDPM.BETA_END,
    )
    if args.skip_steps is None:
        DiffusionModel = diffusions.ddpm.DDPM(
            betas=betas,
            objective=cfg.DDPM.OBJECTIVE,
            var_type=cfg.DDPM.VAR_TYPE,
        )
    else:
        skip = cfg.DDPM.TOTAL_STEPS // args.skip_steps
        timesteps = torch.arange(0, cfg.DDPM.TOTAL_STEPS, skip)
        DiffusionModel = diffusions.ddpm.DDPMSkip(
            timesteps=timesteps,
            betas=betas,
            objective=cfg.DDPM.OBJECTIVE,
            var_type=cfg.DDPM.VAR_TYPE,
        )

    # BUILD MODEL
    model = build_model(cfg)
    model.to(device=device)
    model.eval()

    # LOAD WEIGHTS
    ckpt = torch.load(args.model_path, map_location='cpu')
    model.load_state_dict(ckpt['ema']['shadow'] if args.load_ema else ckpt['model'])
    logger.info(f'Successfully load model from {args.model_path}')

    # SAMPLE
    if args.mode == 'sample':
        sample()
    elif args.mode == 'denoise':
        sample_denoise()
    elif args.mode == 'progressive':
        sample_progressive()
    else:
        raise ValueError
