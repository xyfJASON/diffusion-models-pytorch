import os
import argparse
from omegaconf import OmegaConf

import torch
import accelerate
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.utils import save_image

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
        '--class_A', type=int, required=True,
        help='Class label of domain A',
    )
    parser.add_argument(
        '--class_B', type=int, required=True,
        help='Class label of domain B',
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
        '--save_dir', type=str, required=True,
        help='Path to directory saving samples',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=500,
        help='Batch size on each process. Sample by batch is much faster',
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
    def translate():
        # build dataset
        if args.input_dir is None:
            raise ValueError('input_dir must be specified when mode is reconstruction')
        transforms = T.Compose([
            T.Resize((conf.data.params.img_size, conf.data.params.img_size)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
        dataset = ImageDir(root=args.input_dir, transform=transforms)
        dataloader = DataLoader(dataset=dataset, batch_size=args.micro_batch, **conf.dataloader)
        dataloader = accelerator.prepare(dataloader)  # type: ignore
        # sampling
        idx = 0
        for i, X in enumerate(dataloader):
            X = X[0].float() if isinstance(X, (tuple, list)) else X.float()
            yA = torch.full((X.shape[0], ), args.class_A, device=device).long()
            yB = torch.full((X.shape[0], ), args.class_B, device=device).long()
            noise = diffuser.sample_inversion(
                model=accelerator.unwrap_model(model), img=X, model_kwargs=dict(y=yA),
                tqdm_kwargs=dict(desc=f'img2noise {i}/{len(dataloader)}', disable=not accelerator.is_main_process),
            )
            tX = diffuser.sample(
                model=model, init_noise=noise, model_kwargs=dict(y=yB),
                tqdm_kwargs=dict(desc=f'noise2img {i}/{len(dataloader)}', disable=not accelerator.is_main_process),
            )
            X = accelerator.gather_for_metrics(X)
            tX = accelerator.gather_for_metrics(tX)
            if accelerator.is_main_process:
                for x, tx in zip(X, tX):
                    x = image_norm_to_float(x).cpu()
                    tx = image_norm_to_float(tx).cpu()
                    save_image([x, tx], os.path.join(args.save_dir, f'{idx}.png'), nrow=2)
                    idx += 1

    # START SAMPLING
    logger.info('Start sampling...')
    os.makedirs(args.save_dir, exist_ok=True)
    logger.info(f'Samples will be saved to {args.save_dir}')
    translate()
    logger.info(f'Sampled images are saved to {args.save_dir}')
    logger.info('End of sampling')


if __name__ == '__main__':
    main()
