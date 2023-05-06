import os
import tqdm
import math
import argparse
from contextlib import nullcontext
from yacs.config import CfgNode as CN

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image

import accelerate

from models import EMA
from metrics import AverageMeter
from utils.logger import StatusTracker, get_logger
from utils.data import get_dataset, get_data_generator
from utils.misc import get_time_str, create_exp_dir, check_freq, find_resume_checkpoint, instantiate_from_config


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config', type=str, required=True,
        help='Path to training configuration file',
    )
    parser.add_argument(
        '-e', '--exp_dir', type=str,
        help='Path to the experiment directory. Default to be ./runs/exp-{current time}/',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def train(args, cfg):
    # INITIALIZE ACCELERATOR
    accelerator = accelerate.Accelerator()
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()
    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            cfg_dump=cfg.dump(sort_keys=False),
            exist_ok=cfg.train.resume is not None,
            time_str=args.time_str,
            no_interaction=args.no_interaction,
        )
    # INITIALIZE LOGGER
    logger = get_logger(
        log_file=os.path.join(exp_dir, f'output-{args.time_str}.log'),
        use_tqdm_handler=True,
        is_main_process=accelerator.is_main_process,
    )
    # INITIALIZE STATUS TRACKER
    status_tracker = StatusTracker(
        logger=logger,
        exp_dir=exp_dir,
        print_freq=cfg.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )
    # SET SEED
    accelerate.utils.set_seed(cfg.seed, device_specific=True)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')
    logger.info('=' * 30)

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    assert cfg.train.batch_size % accelerator.num_processes == 0
    batch_size_per_process = cfg.train.batch_size // accelerator.num_processes
    micro_batch = cfg.dataloader.micro_batch or batch_size_per_process
    train_set = get_dataset(
        name=cfg.data.name,
        dataroot=cfg.data.dataroot,
        img_size=cfg.data.img_size,
        split='train',
    )
    train_loader = DataLoader(
        dataset=train_set,
        shuffle=True,
        drop_last=True,
        batch_size=batch_size_per_process,
        num_workers=cfg.dataloader.num_workers,
        pin_memory=cfg.dataloader.pin_memory,
        prefetch_factor=cfg.dataloader.prefetch_factor,
    )
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {cfg.train.batch_size}')
    logger.info('=' * 30)

    # BUILD DIFFUSER
    cfg.diffusion.params.update({'device': device})
    diffuser = instantiate_from_config(cfg.diffusion)

    # BUILD MODEL AND OPTIMIZERS
    model = instantiate_from_config(cfg.model)
    ema = EMA(model=model, decay=cfg.model.ema_decay, gradual=cfg.model.get('ema_gradual', True))
    cfg.train.optim.params.update({'params': model.parameters()})
    optimizer = instantiate_from_config(cfg.train.optim)
    step = 0

    def load_ckpt(ckpt_path: str):
        nonlocal step
        # load model
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {ckpt_path}')
        ema.load_state_dict(ckpt_model['ema'], device=device)
        logger.info(f'Successfully load ema from {ckpt_path}')
        # load optimizer
        ckpt_optimizer = torch.load(os.path.join(ckpt_path, 'optimizer.pt'), map_location='cpu')
        optimizer.load_state_dict(ckpt_optimizer['optimizer'])
        logger.info(f'Successfully load optimizer from {ckpt_path}')
        # load meta information
        ckpt_meta = torch.load(os.path.join(ckpt_path, 'meta.pt'), map_location='cpu')
        step = ckpt_meta['step'] + 1

    @accelerator.on_main_process
    def save_ckpt(save_path: str):
        os.makedirs(save_path, exist_ok=True)
        # save model
        accelerator.save(dict(
            model=accelerator.unwrap_model(model).state_dict(),
            ema=ema.state_dict(),
        ), os.path.join(save_path, 'model.pt'))
        # save optimizer
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if cfg.train.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, cfg.train.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)  # type: ignore
    ema.to(device)

    accelerator.wait_for_everyone()

    def run_step(_batch):
        optimizer.zero_grad()
        if isinstance(_batch, (tuple, list)):
            _batch = _batch[0]
        batch_size = _batch.shape[0]
        loss_meter = AverageMeter()
        for i in range(0, batch_size, micro_batch):
            X = _batch[i:i+micro_batch].float()
            t = torch.randint(cfg.diffusion.params.total_steps, (X.shape[0], ), device=device).long()
            loss_scale = X.shape[0] / batch_size
            no_sync = (i + micro_batch) < batch_size
            cm = accelerator.no_sync(model) if no_sync else nullcontext()
            with cm:
                loss = diffuser.loss_func(model, x0=X, t=t)
                accelerator.backward(loss * loss_scale)
            loss_meter.update(loss.item(), X.shape[0])
        accelerator.clip_grad_norm_(model.parameters(), max_norm=cfg.train.clip_grad_norm)
        optimizer.step()
        ema.update()
        return dict(
            loss=loss_meter.avg,
            lr=optimizer.param_groups[0]['lr'],
        )

    @torch.no_grad()
    def sample(savepath: str):

        def amortize(n_samples: int, _batch_size: int):
            k = n_samples // _batch_size
            r = n_samples % _batch_size
            return k * [_batch_size] if r == 0 else k * [_batch_size] + [r]

        unwrapped_model = accelerator.unwrap_model(model)
        ema.apply_shadow()
        all_samples = []
        img_shape = (cfg.data.img_channels, cfg.data.img_size, cfg.data.img_size)
        mb = min(micro_batch, math.ceil(cfg.train.n_samples / accelerator.num_processes))
        batch_size = mb * accelerator.num_processes
        for bs in tqdm.tqdm(amortize(cfg.train.n_samples, batch_size), desc='Sampling',
                            leave=False, disable=not accelerator.is_main_process):
            init_noise = torch.randn((mb, *img_shape), device=device)
            samples = diffuser.sample(model=unwrapped_model, init_noise=init_noise).clamp(-1, 1)
            samples = accelerator.gather(samples)[:bs]
            all_samples.append(samples)
        all_samples = torch.cat(all_samples, dim=0)
        if accelerator.is_main_process:
            nrow = math.ceil(math.sqrt(cfg.train.n_samples))
            save_image(all_samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))
        ema.restore()

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        is_main_process=accelerator.is_main_process,
        with_tqdm=True,
    )
    while step < cfg.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        model.eval()
        # save checkpoint
        if check_freq(cfg.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(cfg.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(cfg.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


def main():
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    cfg.merge_from_list(unknown_args)
    cfg.freeze()

    train(args, cfg)


if __name__ == '__main__':
    main()
