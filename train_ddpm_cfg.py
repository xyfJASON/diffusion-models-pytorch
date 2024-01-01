import os
import math
import argparse
from omegaconf import OmegaConf
from contextlib import nullcontext

import torch
import accelerate
from torch.utils.data import DataLoader
from torchvision.utils import save_image

from models import EMA
from utils.logger import StatusTracker, get_logger
from utils.misc import create_exp_dir, find_resume_checkpoint, instantiate_from_config
from utils.misc import get_time_str, check_freq, amortize, get_data_generator, AverageMeter


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
        '-r', '--resume', type=str,
        help='Resume from a checkpoint. Could be a path or `best` or `latest`',
    )
    parser.add_argument(
        '-ni', '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    return parser


def main():
    # ARGS & CONF
    args, unknown_args = get_parser().parse_known_args()
    args.time_str = get_time_str()
    if args.exp_dir is None:
        args.exp_dir = os.path.join('runs', f'exp-{args.time_str}')
    unknown_args = [(a[2:] if a.startswith('--') else a) for a in unknown_args]
    unknown_args = [f'{k}={v}' for k, v in zip(unknown_args[::2], unknown_args[1::2])]
    conf = OmegaConf.load(args.config)
    conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(unknown_args))

    # INITIALIZE ACCELERATOR
    ddp_kwargs = accelerate.DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = accelerate.Accelerator(kwargs_handlers=[ddp_kwargs])
    device = accelerator.device
    print(f'Process {accelerator.process_index} using device: {device}')
    accelerator.wait_for_everyone()

    # CREATE EXPERIMENT DIRECTORY
    exp_dir = args.exp_dir
    if accelerator.is_main_process:
        create_exp_dir(
            exp_dir=exp_dir,
            conf_yaml=OmegaConf.to_yaml(conf),
            exist_ok=args.resume is not None,
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
        print_freq=conf.train.print_freq,
        is_main_process=accelerator.is_main_process,
    )

    # SET SEED
    accelerate.utils.set_seed(conf.seed, device_specific=True)
    logger.info('=' * 19 + ' System Info ' + '=' * 18)
    logger.info(f'Experiment directory: {exp_dir}')
    logger.info(f'Number of processes: {accelerator.num_processes}')
    logger.info(f'Distributed type: {accelerator.distributed_type}')
    logger.info(f'Mixed precision: {accelerator.mixed_precision}')

    accelerator.wait_for_everyone()

    # BUILD DATASET & DATALOADER
    if conf.train.batch_size % accelerator.num_processes != 0:
        raise ValueError(
            f'Batch size should be divisible by number of processes, '
            f'get {conf.train.batch_size} % {accelerator.num_processes} != 0'
        )
    batch_size_per_process = conf.train.batch_size // accelerator.num_processes
    micro_batch = conf.train.micro_batch or batch_size_per_process
    train_set = instantiate_from_config(conf.data)
    train_loader = DataLoader(
        dataset=train_set, batch_size=batch_size_per_process,
        shuffle=True, drop_last=True, **conf.dataloader,
    )
    logger.info('=' * 19 + ' Data Info ' + '=' * 20)
    logger.info(f'Size of training set: {len(train_set)}')
    logger.info(f'Batch size per process: {batch_size_per_process}')
    logger.info(f'Total batch size: {conf.train.batch_size}')
    logger.info('=' * 50)

    # BUILD DIFFUSER
    diffuser = instantiate_from_config(conf.diffusion, device=device)

    # BUILD MODEL AND OPTIMIZERS
    model = instantiate_from_config(conf.model)
    ema = EMA(model.parameters(), decay=conf.train.ema_decay, gradual=conf.train.ema_gradual)
    optimizer = instantiate_from_config(conf.train.optim, params=model.parameters())
    step = 0

    def load_ckpt(ckpt_path: str):
        nonlocal step
        # load model
        ckpt_model = torch.load(os.path.join(ckpt_path, 'model.pt'), map_location='cpu')
        model.load_state_dict(ckpt_model['model'])
        logger.info(f'Successfully load model from {ckpt_path}')
        # load ema
        ckpt_ema = torch.load(os.path.join(ckpt_path, 'ema.pt'), map_location='cpu')
        ema.load_state_dict(ckpt_ema['ema'])
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
        unwrapped_model = accelerator.unwrap_model(model)
        # save model
        accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'model.pt'))
        # save ema
        accelerator.save(dict(ema=ema.state_dict()), os.path.join(save_path, 'ema.pt'))
        # save ema model
        ema.apply_shadow(model.parameters())
        accelerator.save(dict(model=unwrapped_model.state_dict()), os.path.join(save_path, 'ema_model.pt'))
        ema.restore(model.parameters())
        # save optimizer
        accelerator.save(dict(optimizer=optimizer.state_dict()), os.path.join(save_path, 'optimizer.pt'))
        # save meta information
        accelerator.save(dict(step=step), os.path.join(save_path, 'meta.pt'))

    # RESUME TRAINING
    if args.resume is not None:
        resume_path = find_resume_checkpoint(exp_dir, args.resume)
        logger.info(f'Resume from {resume_path}')
        load_ckpt(resume_path)
        logger.info(f'Restart training at step {step}')

    # PREPARE FOR DISTRIBUTED MODE AND MIXED PRECISION
    model, optimizer, train_loader = accelerator.prepare(model, optimizer, train_loader)  # type: ignore
    ema.to(device)

    accelerator.wait_for_everyone()

    def run_step(_batch):
        optimizer.zero_grad()
        batchX, batchy = _batch
        batch_size = batchX.shape[0]
        loss_meter = AverageMeter()
        for i in range(0, batch_size, micro_batch):
            X = batchX[i:i+micro_batch].float()
            y = batchy[i:i+micro_batch].long()
            if torch.rand(1) < conf.train.p_uncond:
                y = None
            t = torch.randint(conf.diffusion.params.total_steps, (X.shape[0], ), device=device).long()
            loss_scale = X.shape[0] / batch_size
            no_sync = (i + micro_batch) < batch_size
            cm = accelerator.no_sync(model) if no_sync else nullcontext()
            with cm:
                loss = diffuser.loss_func(model, x0=X, t=t, model_kwargs=dict(y=y))
                accelerator.backward(loss * loss_scale)
            loss_meter.update(loss.item(), X.shape[0])
        accelerator.clip_grad_norm_(model.parameters(), max_norm=conf.train.clip_grad_norm)
        optimizer.step()
        ema.update(model.parameters())
        return dict(
            loss=loss_meter.avg,
            lr=optimizer.param_groups[0]['lr'],
        )

    @torch.no_grad()
    def sample(savepath: str):
        unwrapped_model = accelerator.unwrap_model(model)
        ema.apply_shadow(model.parameters())
        # use respaced sampling to save time during training
        diffuser.set_respaced_seq('uniform', diffuser.total_steps // 20)

        all_samples = []
        img_shape = (conf.data.img_channels, conf.data.params.img_size, conf.data.params.img_size)
        mb = min(micro_batch, math.ceil(conf.train.n_samples_each_class / accelerator.num_processes))
        for c in range(min(10, conf.data.num_classes)):
            samples_c = []
            folds = amortize(conf.train.n_samples_each_class, mb * accelerator.num_processes)
            for i, bs in enumerate(folds):
                init_noise = torch.randn((mb, *img_shape), device=device)
                labels = torch.full((mb, ), fill_value=c, device=device)
                samples = diffuser.sample(
                    model=unwrapped_model,
                    init_noise=init_noise,
                    var_type='fixed_small',
                    model_kwargs=dict(y=labels),
                    tqdm_kwargs=dict(
                        desc=f'Sampling fold {i}/{len(folds)}',
                        leave=False, disable=not accelerator.is_main_process,
                    ),
                ).clamp(-1, 1)
                samples = accelerator.gather(samples)[:bs]
                samples_c.append(samples)
            samples_c = torch.cat(samples_c, dim=0)
            all_samples.append(samples_c)
        all_samples = torch.cat(all_samples, dim=0).view(-1, *img_shape)
        if accelerator.is_main_process:
            nrow = conf.train.n_samples_each_class
            save_image(all_samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))

        diffuser.set_respaced_seq('none', diffuser.total_steps)
        ema.restore(model.parameters())

    # START TRAINING
    logger.info('Start training...')
    train_data_generator = get_data_generator(
        dataloader=train_loader,
        tqdm_kwargs=dict(desc='Epoch', leave=False, disable=not accelerator.is_main_process),
    )
    while step < conf.train.n_steps:
        # get a batch of data
        batch = next(train_data_generator)
        # run a step
        model.train()
        train_status = run_step(batch)
        status_tracker.track_status('Train', train_status, step)
        accelerator.wait_for_everyone()

        model.eval()
        # save checkpoint
        if check_freq(conf.train.save_freq, step):
            save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step:0>6d}'))
            accelerator.wait_for_everyone()
        # sample from current model
        if check_freq(conf.train.sample_freq, step):
            sample(os.path.join(exp_dir, 'samples', f'step{step:0>6d}.png'))
            accelerator.wait_for_everyone()
        step += 1
    # save the last checkpoint if not saved
    if not check_freq(conf.train.save_freq, step - 1):
        save_ckpt(os.path.join(exp_dir, 'ckpt', f'step{step-1:0>6d}'))
    accelerator.wait_for_everyone()
    status_tracker.close()
    logger.info('End of training')


if __name__ == '__main__':
    main()
