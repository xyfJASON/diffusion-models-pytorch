import os
import math
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import models
import diffusions.ddpm
from metrics import AverageMeter
from engine.tools import build_model, build_optimizer
from utils.optimizer import optimizer_to_device
from utils.logger import get_logger, StatusTracker
from utils.data import get_dataset, get_dataloader, get_data_generator
from utils.dist import init_distributed_mode, broadcast_objects, main_process_only
from utils.dist import get_rank, get_world_size, get_local_rank, is_dist_avail_and_initialized, is_main_process
from utils.misc import get_time_str, init_seeds, create_exp_dir, check_freq, get_bare_model, find_resume_checkpoint


class DDPMTrainer:
    def __init__(self, cfg, args):
        self.cfg, self.args = cfg, args
        self.time_str = get_time_str()

        # INITIALIZE DISTRIBUTED MODE
        init_distributed_mode()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # INITIALIZE SEEDS
        init_seeds(self.cfg.SEED + get_rank())

        # CREATE EXPERIMENT DIRECTORY
        self.exp_dir = create_exp_dir(
            cfg_dump=self.cfg.dump(),
            resume=self.cfg.TRAIN.RESUME is not None,
            time_str=self.time_str,
            name=self.args.name,
            no_interaction=self.args.no_interaction,
        )
        self.exp_dir = broadcast_objects(self.exp_dir)

        # INITIALIZE LOGGER
        self.logger = get_logger(log_file=os.path.join(self.exp_dir, f'output-{self.time_str}.log'))
        self.logger.info(f'Experiment directory: {self.exp_dir}')
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f"Number of devices: {get_world_size()}")

        # BUILD DATASET & DATALOADER & DATA GENERATOR
        train_set = get_dataset(
            name=self.cfg.DATA.NAME,
            dataroot=self.cfg.DATA.DATAROOT,
            img_size=self.cfg.DATA.IMG_SIZE,
            split='train',
        )
        self.train_loader = get_dataloader(
            dataset=train_set,
            shuffle=True,
            drop_last=True,
            batch_size=self.cfg.DATALOADER.BATCH_SIZE,
            num_workers=self.cfg.DATALOADER.NUM_WORKERS,
            pin_memory=self.cfg.DATALOADER.PIN_MEMORY,
            prefetch_factor=self.cfg.DATALOADER.PREFETCH_FACTOR,
        )
        self.micro_batch = self.cfg.DATALOADER.MICRO_BATCH
        if self.micro_batch == 0:
            self.micro_batch = self.cfg.DATALOADER.BATCH_SIZE
        effective_batch = self.cfg.DATALOADER.BATCH_SIZE * get_world_size()
        self.logger.info(f'Size of training set: {len(train_set)}')
        self.logger.info(f'Batch size per device: {self.cfg.DATALOADER.BATCH_SIZE}')
        self.logger.info(f'Effective batch size: {effective_batch}')

        # BUILD DIFFUSER, MODEL AND OPTIMIZERS
        self.DiffusionModel = diffusions.ddpm.DDPM(
            total_steps=self.cfg.DDPM.TOTAL_STEPS,
            beta_schedule=self.cfg.DDPM.BETA_SCHEDULE,
            beta_start=self.cfg.DDPM.BETA_START,
            beta_end=self.cfg.DDPM.BETA_END,
            objective=self.cfg.DDPM.OBJECTIVE,
            var_type=self.cfg.DDPM.VAR_TYPE,
        )
        self.model = build_model(cfg)
        self.model.to(device=self.device)
        self.ema = models.EMA(self.model, decay=self.cfg.MODEL.EMA_DECAY)
        self.optimizer = build_optimizer(self.model.parameters(), self.cfg)

        # LOAD PRETRAINED WEIGHTS
        if self.cfg.MODEL.WEIGHTS is not None:
            self.load_model(self.cfg.MODEL.WEIGHTS)

        # RESUME
        self.cur_step = 0
        if self.cfg.TRAIN.RESUME is not None:
            resume_path = find_resume_checkpoint(self.exp_dir, self.cfg.TRAIN.RESUME)
            self.logger.info(f'Resume from {resume_path}')
            self.load_ckpt(resume_path)

        # DISTRIBUTED MODELS
        if is_dist_avail_and_initialized():
            self.model = DDP(
                module=self.model,
                device_ids=[get_local_rank()],
                output_device=get_local_rank(),
            )

        # DEFINE LOSSES
        self.loss_meter = AverageMeter().to(self.device)

        # DEFINE STATUS TRACKER
        self.status_tracker = StatusTracker(
            logger=self.logger,
            exp_dir=self.exp_dir,
            print_freq=cfg.TRAIN.PRINT_FREQ,
        )

    def load_model(self, model_path: str, load_ema: bool = False):
        ckpt = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(ckpt['ema']['shadow'] if load_ema else ckpt['model'])
        self.model.to(device=self.device)
        self.logger.info(f'Successfully load model from {model_path}')
        self.ema.load_state_dict(ckpt['ema'], device=self.device)
        self.logger.info(f'Successfully load ema from {model_path}')

    def load_ckpt(self, ckpt_path: str):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        # load model
        self.load_model(ckpt_path)
        # load optimizer
        self.optimizer.load_state_dict(ckpt['optimizer'])
        optimizer_to_device(self.optimizer, self.device)
        self.logger.info(f'Successfully load optimizer from {ckpt_path}')
        # load meta informations
        self.cur_step = ckpt['step'] + 1
        self.logger.info(f'Restart training at step {self.cur_step}')

    @main_process_only
    def save_ckpt(self, save_path: str):
        state_dicts = dict(
            model=get_bare_model(self.model).state_dict(),
            ema=self.ema.state_dict(),
            optimizer=self.optimizer.state_dict(),
            step=self.cur_step,
        )
        torch.save(state_dicts, save_path)

    def run_loop(self):
        self.logger.info('Start training...')
        train_data_generator = get_data_generator(
            dataloader=self.train_loader,
            start_epoch=self.cur_step,
        )
        while self.cur_step < self.cfg.TRAIN.TRAIN_STEPS:
            # get a batch of data
            batch = next(train_data_generator)
            # run a step
            train_status = self.run_step(batch)
            self.status_tracker.track_status('Train', train_status, self.cur_step)
            # save checkpoint
            if check_freq(self.cfg.TRAIN.SAVE_FREQ, self.cur_step):
                self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step:0>6d}.pt'))
            # sample from current model
            if check_freq(self.cfg.TRAIN.SAMPLE_FREQ, self.cur_step):
                self.sample(os.path.join(self.exp_dir, 'samples', f'step{self.cur_step:0>6d}.png'))
            # synchronizes all processes
            if is_dist_avail_and_initialized():
                dist.barrier()
            self.cur_step += 1
        # save the last checkpoint if not saved
        if not check_freq(self.cfg.TRAIN.SAVE_FREQ, self.cur_step - 1):
            self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step-1:0>6d}.pt'))
        self.status_tracker.close()
        self.logger.info('End of training')

    def run_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()
        self.loss_meter.reset()
        batch = batch[0] if isinstance(batch, (tuple, list)) else batch
        batch_size = batch.shape[0]
        for i in range(0, batch_size, self.micro_batch):
            X = batch[i:i+self.micro_batch].to(device=self.device, dtype=torch.float32)
            t = torch.randint(self.cfg.DDPM.TOTAL_STEPS, (X.shape[0], ), device=self.device).long()
            # no need to synchronize gradient before the last micro batch
            no_sync = is_dist_avail_and_initialized() and (i + self.micro_batch) < batch_size
            cm = self.model.no_sync() if no_sync else nullcontext()
            with cm:
                loss = self.DiffusionModel.loss_func(self.model, X0=X, t=t)
                loss.backward()
            self.loss_meter.update(loss.detach(), X.shape[0])
        train_status = dict(
            loss=self.loss_meter.compute(),
            lr=self.optimizer.param_groups[0]['lr'],
        )
        nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
        self.optimizer.step()
        self.ema.update()
        return train_status

    @torch.no_grad()
    def sample(self, savepath: str):
        num_each_device = 64 // get_world_size()
        model = get_bare_model(self.model).eval()
        samples = []
        total_folds = math.ceil(num_each_device / self.micro_batch)
        img_shape = (self.cfg.DATA.IMG_CHANNELS, self.cfg.DATA.IMG_SIZE, self.cfg.DATA.IMG_SIZE)
        for i in range(total_folds):
            n = min(self.micro_batch, num_each_device - i * self.micro_batch)
            init_noise = torch.randn((n, *img_shape), device=self.device)
            X = self.DiffusionModel.sample(
                model=model,
                init_noise=init_noise,
            ).clamp(-1, 1)
            samples.append(X)
        samples = torch.cat(samples, dim=0)
        if is_dist_avail_and_initialized():
            sample_list = [torch.Tensor() for _ in range(get_world_size())]
            dist.all_gather_object(sample_list, samples)
            samples = torch.cat(sample_list, dim=0)
        if is_main_process():
            save_image(samples, savepath, nrow=8, normalize=True, value_range=(-1, 1))

    # @torch.no_grad()
    # def sample(self):
    #     cfg = self.config.sample
    #     self.logger.info('Start sampling...')
    #     self.logger.info(f'Samples will be saved to {cfg.save_dir}')
    #     os.makedirs(cfg.save_dir, exist_ok=True)
    #     num_each_device = cfg.n_samples // self.dist_info.world_size
    #     model = get_bare_model(self.model).eval()
    #
    #     total = math.ceil(num_each_device / cfg.batch_size)
    #     img_shape = (self.config.data.img_channels, self.config.data.img_size, self.config.data.img_size)
    #     for i in range(total):
    #         n = min(cfg.batch_size, num_each_device - i * cfg.batch_size)
    #         init_noise = torch.randn((n, *img_shape), device=self.device)
    #         X = self.DiffusionModel.sample(
    #             model=model,
    #             init_noise=init_noise,
    #             with_tqdm=self.dist_info.is_master,
    #             desc=f'Sampling({i+1}/{total})',
    #             ncols=120,
    #         ).clamp(-1, 1)
    #         for j, x in enumerate(X):
    #             idx = self.dist_info.global_rank * num_each_device + i * cfg.batch_size + j
    #             save_image(
    #                 tensor=x.cpu(), fp=os.path.join(cfg.save_dir, f'{idx}.png'),
    #                 nrow=1, normalize=True, value_range=(-1, 1),
    #             )
    #     self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
    #     self.logger.info('End of sampling')
    #
    # @torch.no_grad()
    # def sample_denoise(self):
    #     cfg = self.config.sample_denoise
    #     self.logger.info('Start sampling...')
    #     self.logger.info(f'Samples will be saved to {cfg.save_dir}')
    #     os.makedirs(cfg.save_dir, exist_ok=True)
    #     num_each_device = cfg.n_samples // self.dist_info.world_size
    #     model = get_bare_model(self.model).eval()
    #
    #     total = math.ceil(num_each_device / cfg.batch_size)
    #     img_shape = (self.config.data.img_channels, self.config.data.img_size, self.config.data.img_size)
    #     freq = self.config.ddpm.total_steps // cfg.n_denoise
    #     for i in range(total):
    #         n = min(cfg.batch_size, num_each_device - i * cfg.batch_size)
    #         init_noise = torch.randn((n, *img_shape), device=self.device)
    #         sample_generator = self.DiffusionModel.sample_loop(
    #             model=model,
    #             init_noise=init_noise,
    #             with_tqdm=self.dist_info.is_master,
    #             desc=f'Sampling({i+1}/{total})',
    #             ncols=120,
    #         )
    #         X = []
    #         for timestep, out in enumerate(sample_generator):
    #             if (self.config.ddpm.total_steps - timestep - 1) % freq == 0:
    #                 X.append(out['sample'])
    #         X = torch.stack(X, dim=1).clamp(-1, 1)
    #         for j, x in enumerate(X):
    #             idx = self.dist_info.global_rank * num_each_device + i * cfg.batch_size + j
    #             save_image(
    #                 tensor=x.cpu(), fp=os.path.join(cfg.save_dir, f'{idx}.png'),
    #                 nrow=len(x), normalize=True, value_range=(-1, 1),
    #             )
    #     self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
    #     self.logger.info('End of sampling')
    #
    # @torch.no_grad()
    # def sample_progressive(self):
    #     cfg = self.config.sample_progressive
    #     self.logger.info('Start sampling...')
    #     self.logger.info(f'Samples will be saved to {cfg.save_dir}')
    #     os.makedirs(cfg.save_dir, exist_ok=True)
    #     num_each_device = cfg.n_samples // self.dist_info.world_size
    #     model = get_bare_model(self.model).eval()
    #
    #     total = math.ceil(num_each_device / cfg.batch_size)
    #     img_shape = (self.config.data.img_channels, self.config.data.img_size, self.config.data.img_size)
    #     freq = self.config.ddpm.total_steps // cfg.n_progressive
    #     for i in range(total):
    #         n = min(cfg.batch_size, num_each_device - i * cfg.batch_size)
    #         init_noise = torch.randn((n, *img_shape), device=self.device)
    #         sample_generator = self.DiffusionModel.sample_loop(
    #             model=model,
    #             init_noise=init_noise,
    #             with_tqdm=self.dist_info.is_master,
    #             desc=f'Sampling({i+1}/{total})',
    #             ncols=120,
    #         )
    #         X = []
    #         for timestep, out in enumerate(sample_generator):
    #             if (self.config.ddpm.total_steps - timestep - 1) % freq == 0:
    #                 X.append(out['pred_X0'])
    #         X = torch.stack(X, dim=1).clamp(-1, 1)
    #         for j, x in enumerate(X):
    #             idx = self.dist_info.global_rank * num_each_device + i * cfg.batch_size + j
    #             save_image(
    #                 tensor=x.cpu(), fp=os.path.join(cfg.save_dir, f'{idx}.png'),
    #                 nrow=len(x), normalize=True, value_range=(-1, 1),
    #             )
    #     self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
    #     self.logger.info('End of sampling')
    #
    # @torch.no_grad()
    # def sample_skip(self):
    #     cfg = self.config.sample_skip
    #     self.logger.info('Start sampling...')
    #     self.logger.info(f'Samples will be saved to {cfg.save_dir}')
    #     os.makedirs(cfg.save_dir, exist_ok=True)
    #     num_each_device = cfg.n_samples // self.dist_info.world_size
    #     model = get_bare_model(self.model).eval()
    #
    #     skip = self.config.ddpm.total_steps // cfg.n_timesteps
    #     timesteps = torch.arange(0, self.config.ddpm.total_steps, skip)
    #     DiffusionModel = diffusions.DDPMSkip(
    #         timesteps=timesteps,
    #         total_steps=self.config.ddpm.total_steps,
    #         beta_schedule=self.config.ddpm.beta_schedule,
    #         beta_start=self.config.ddpm.beta_start,
    #         beta_end=self.config.ddpm.beta_end,
    #         objective=self.config.ddpm.objective,
    #         var_type=self.config.ddpm.var_type,
    #     )
    #
    #     total = math.ceil(num_each_device / cfg.batch_size)
    #     img_shape = (self.config.data.img_channels, self.config.data.img_size, self.config.data.img_size)
    #     for i in range(total):
    #         n = min(cfg.batch_size, num_each_device - i * cfg.batch_size)
    #         init_noise = torch.randn((n, *img_shape), device=self.device)
    #         X = DiffusionModel.sample(
    #             model=model,
    #             init_noise=init_noise,
    #             with_tqdm=self.dist_info.is_master,
    #             desc=f'Sampling({i+1}/{total})',
    #             ncols=120,
    #         ).clamp(-1, 1)
    #
    #         for j, x in enumerate(X):
    #             idx = self.dist_info.global_rank * num_each_device + i * cfg.batch_size + j
    #             save_image(
    #                 tensor=x.cpu(),
    #                 fp=os.path.join(cfg.save_dir, f'{idx}.png'),
    #                 nrow=1, normalize=True, value_range=(-1, 1),
    #             )
    #     self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
    #     self.logger.info('End of sampling')
