import os
import math
import argparse
from functools import partial
from yacs.config import CfgNode as CN

import torch
from torchvision.utils import save_image

import diffusions.classifier_free
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
    parser.add_argument('--n_samples_each_class', type=int, required=True, help='number of samples')
    parser.add_argument('--guidance_scale', type=float, required=True,
                        help='guidance scale. 0 for unconditional generation, '
                             '1 for non-guided generation, >1 for guided generation')
    parser.add_argument('--skip_steps', type=int, help='number of timesteps for skip sampling')
    parser.add_argument('--ddim', action='store_true', help='use DDIM deterministic sampling')
    parser.add_argument('--ddim_eta', type=float, default=0.0, help='eta in DDIM')
    parser.add_argument('--save_dir', type=str, required=True, help='path to directory saving samples')
    parser.add_argument('--batch_size', type=int, default=1, help='sample by batch is faster')
    parser.add_argument('--seed', type=int, default=2022, help='random seed')
    # modify basic configuration in yaml file
    parser.add_argument('--opts', default=[], nargs=argparse.REMAINDER,
                        help="modify config options using the command-line 'KEY VALUE' pairs")
    return parser


@torch.no_grad()
def sample():
    logger.info('Start sampling...')
    logger.info(f'Samples will be saved to {args.save_dir}')
    os.makedirs(args.save_dir, exist_ok=True)
    num_each_device = args.n_samples_each_class // get_world_size()

    total_folds = math.ceil(num_each_device / args.batch_size)
    img_shape = (cfg.DATA.IMG_CHANNELS, cfg.DATA.IMG_SIZE, cfg.DATA.IMG_SIZE)

    if args.ddim:
        sample_fn = partial(DiffusionModel.ddim_sample, eta=args.ddim_eta)
    else:
        sample_fn = DiffusionModel.sample

    for c in range(cfg.DATA.NUM_CLASSES):
        logger.info(f'Sampling class {c}')
        for i in range(total_folds):
            n = min(args.batch_size, num_each_device - i * args.batch_size)
            init_noise = torch.randn((n, *img_shape), device=device)
            X = sample_fn(
                model=model,
                class_label=c,
                init_noise=init_noise,
                guidance_scale=args.guidance_scale,
            ).clamp(-1, 1)
            for j, x in enumerate(X):
                idx = get_rank() * num_each_device + i * args.batch_size + j
                save_image(
                    tensor=x.cpu(), fp=os.path.join(args.save_dir, f'class{c}-{idx}.png'),
                    nrow=1, normalize=True, value_range=(-1, 1),
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
        beta_schedule=cfg.CLASSIFIER_FREE.BETA_SCHEDULE,
        total_steps=cfg.CLASSIFIER_FREE.TOTAL_STEPS,
        beta_start=cfg.CLASSIFIER_FREE.BETA_START,
        beta_end=cfg.CLASSIFIER_FREE.BETA_END,
    )
    if args.skip_steps is None:
        DiffusionModel = diffusions.classifier_free.ClassifierFree(
            betas=betas,
            objective=cfg.CLASSIFIER_FREE.OBJECTIVE,
            var_type=cfg.CLASSIFIER_FREE.VAR_TYPE,
        )
    else:
        skip = cfg.CLASSIFIER_FREE.TOTAL_STEPS // args.skip_steps
        timesteps = torch.arange(0, cfg.CLASSIFIER_FREE.TOTAL_STEPS, skip)
        DiffusionModel = diffusions.classifier_free.ClassifierFreeSkip(
            timesteps=timesteps,
            betas=betas,
            objective=cfg.CLASSIFIER_FREE.OBJECTIVE,
            var_type=cfg.CLASSIFIER_FREE.VAR_TYPE,
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
    sample()
