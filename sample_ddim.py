import os
import math
import argparse
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image

import diffusions.ddim
import diffusions.schedule
from utils.misc import init_seeds
from utils.logger import get_logger
from engine.tools import build_model
from utils.data import get_dataset, get_dataloader
from utils.dist import init_distributed_mode, get_rank, get_world_size


def get_parser():
    parser = argparse.ArgumentParser()
    # load basic configuration from yaml file, including parameter of data, model and diffuser
    parser.add_argument('-c', '--config', metavar='FILE', required=True,
                        help='path to configuration file')
    # arguments for sampling
    parser.add_argument('--model_path', type=str, required=True, help='path to model weights')
    parser.add_argument('--load_ema', action='store_true', help='whether to load ema weights')
    parser.add_argument('--skip_steps', type=int, help='number of timesteps for skip sampling')
    parser.add_argument('--skip_type', type=str, default='uniform', choices=['uniform', 'quad'],
                        help='type of skip sequence')
    parser.add_argument('--n_samples', type=int, required=True, help='number of samples')
    parser.add_argument('--save_dir', type=str, required=True, help='path to directory saving samples')
    parser.add_argument('--batch_size', type=int, default=1, help='sample by batch is faster')
    parser.add_argument('--seed', type=int, default=2001, help='random seed')
    parser.add_argument('--mode', type=str, default='sample',
                        choices=['sample', 'interpolate', 'reconstruction'],
                        help='specify a sample mode')
    parser.add_argument('--n_interpolate', type=int, default=16, help='only valid when mode is interpolate')
    # modify basic configurations in yaml file
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
def sample_interpolate():
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    num_each_device = args.n_samples // get_world_size()

    def slerp(t, z1, z2):  # noqa
        theta = torch.acos(torch.sum(z1 * z2) / (torch.linalg.norm(z1) * torch.linalg.norm(z2)))
        return torch.sin((1 - t) * theta) / torch.sin(theta) * z1 + torch.sin(t * theta) / torch.sin(theta) * z2

    total_folds = math.ceil(num_each_device / args.batch_size)
    img_shape = (cfg.DATA.IMG_CHANNELS, cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)
    for i in range(total_folds):
        n = min(args.batch_size, num_each_device - i * args.batch_size)
        z1 = torch.randn((n, *img_shape), device=device)
        z2 = torch.randn((n, *img_shape), device=device)
        results = torch.stack([
            DiffusionModel.sample(model=model, init_noise=slerp(t, z1, z2)).clamp(-1, 1)
            for t in torch.linspace(0, 1, args.n_interpolate)
        ], dim=1)
        for j, x in enumerate(results):
            idx = get_rank() * num_each_device + i * args.batch_size + j
            save_image(
                tensor=x.cpu(), fp=os.path.join(args.save_dir, f'{idx}.png'),
                nrow=len(x), normalize=True, value_range=(-1, 1),
            )
        logger.info(f'Progress {(i+1)/total_folds*100:.2f}%')
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


@torch.no_grad()
def sample_reconstruction():
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)

    for i, X in enumerate(test_loader):
        X = X[0] if isinstance(X, (tuple, list)) else X
        X = X.to(device=device, dtype=torch.float32)
        noise = DiffusionModel.sample_inversion(model=model, img=X)
        recX = DiffusionModel.sample(model=model, init_noise=noise)
        for j, (x, r) in enumerate(zip(X, recX)):
            filename = f'rank{get_rank()}-{i*args.batch_size+j}.png'
            save_image(
                tensor=[x, r], fp=os.path.join(args.save_dir, filename),
                nrow=2, normalize=True, value_range=(-1, 1),
            )
        logger.info(f'Progress {(i+1)/len(test_loader)*100:.2f}%')
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
        beta_schedule=cfg.DDIM.BETA_SCHEDULE,
        total_steps=cfg.DDIM.TOTAL_STEPS,
        beta_start=cfg.DDIM.BETA_START,
        beta_end=cfg.DDIM.BETA_END,
    )
    if args.skip_steps is None:
        DiffusionModel = diffusions.ddim.DDIM(
            betas=betas,
            objective=cfg.DDIM.OBJECTIVE,
            eta=cfg.DDIM.ETA,
        )
    else:
        skip = cfg.DDIM.TOTAL_STEPS // args.skip_steps
        timesteps = torch.arange(0, cfg.DDIM.TOTAL_STEPS, skip)
        DiffusionModel = diffusions.ddim.DDIMSkip(
            timesteps=timesteps,
            betas=betas,
            objective=cfg.DDIM.OBJECTIVE,
            eta=cfg.DDIM.ETA,
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
    elif args.mode == 'interpolate':
        sample_interpolate()
    elif args.mode == 'reconstruction':
        test_set = get_dataset(
            name=cfg.DATA.NAME,
            dataroot=cfg.DATA.DATAROOT,
            img_size=cfg.DATA.IMG_SIZE,
            split='test',
            subset_ids=range(args.n_samples),
        )
        test_loader = get_dataloader(
            dataset=test_set,
            batch_size=args.batch_size,
        )
        sample_reconstruction()
    else:
        raise ValueError
