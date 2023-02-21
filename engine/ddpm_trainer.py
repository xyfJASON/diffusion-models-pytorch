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
import diffusions.schedule
from metrics import AverageMeter
from engine.tools import build_model, build_optimizer
from utils.optimizer import optimizer_to_device
from utils.logger import get_logger, StatusTracker
from utils.data import get_dataset, get_dataloader, get_data_generator
from utils.dist import init_distributed_mode, broadcast_objects, main_process_only
from utils.dist import get_rank, get_world_size, get_local_rank, is_dist_avail_and_initialized, is_main_process
from utils.misc import get_time_str, init_seeds, create_exp_dir, check_freq, get_bare_model, find_resume_checkpoint


class DDPMTrainer:
    def __init__(self, args):
        self.args = args
        self.time_str = get_time_str()

        # INITIALIZE DISTRIBUTED MODE
        init_distributed_mode()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # INITIALIZE SEEDS
        init_seeds(self.args.seed + get_rank())

        # CREATE EXPERIMENT DIRECTORY
        self.exp_dir = create_exp_dir(
            cfg_dict=self.args.__dict__,
            resume=self.args.resume is not None,
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
            name=self.args.data_name,
            dataroot=self.args.data_dataroot,
            img_size=self.args.data_img_size,
            split='train',
        )
        self.train_loader = get_dataloader(
            dataset=train_set,
            shuffle=True,
            drop_last=True,
            batch_size=self.args.batch_size,
            num_workers=self.args.num_workers,
            pin_memory=self.args.pin_memory,
            prefetch_factor=self.args.prefetch_factor,
        )
        self.micro_batch = self.args.micro_batch
        if self.micro_batch == 0:
            self.micro_batch = self.args.batch_size
        effective_batch = self.args.batch_size * get_world_size()
        self.logger.info(f'Size of training set: {len(train_set)}')
        self.logger.info(f'Batch size per device: {self.args.batch_size}')
        self.logger.info(f'Effective batch size: {effective_batch}')

        # BUILD DIFFUSER, MODEL AND OPTIMIZERS
        betas = diffusions.schedule.get_beta_schedule(
            beta_schedule=self.args.diffusion_beta_schedule,
            total_steps=self.args.diffusion_total_steps,
            beta_start=self.args.diffusion_beta_start,
            beta_end=self.args.diffusion_beta_end,
        )
        self.DiffusionModel = diffusions.ddpm.DDPM(
            betas=betas,
            objective=self.args.diffusion_objective,
            var_type=self.args.diffusion_var_type,
        )
        self.model = build_model(self.args)
        self.model.to(device=self.device)
        self.ema = models.EMA(
            model=self.model,
            decay=self.args.ema_decay,
            gradual=self.args.ema_gradual,
        )
        self.optimizer = build_optimizer(self.model.parameters(), self.args)

        # LOAD PRETRAINED WEIGHTS
        if self.args.weights is not None:
            self.load_model(self.args.weights)

        # RESUME
        self.cur_step = 0
        if self.args.resume is not None:
            resume_path = find_resume_checkpoint(self.exp_dir, self.args.resume)
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
            print_freq=self.args.print_freq,
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
        while self.cur_step < self.args.train_steps:
            # get a batch of data
            batch = next(train_data_generator)
            # run a step
            train_status = self.run_step(batch)
            self.status_tracker.track_status('Train', train_status, self.cur_step)
            # save checkpoint
            if check_freq(self.args.save_freq, self.cur_step):
                self.save_ckpt(os.path.join(self.exp_dir, 'ckpt', f'step{self.cur_step:0>6d}.pt'))
            # sample from current model
            if check_freq(self.args.sample_freq, self.cur_step):
                self.sample(os.path.join(self.exp_dir, 'samples', f'step{self.cur_step:0>6d}.png'))
            # synchronizes all processes
            if is_dist_avail_and_initialized():
                dist.barrier()
            self.cur_step += 1
        # save the last checkpoint if not saved
        if not check_freq(self.args.save_freq, self.cur_step - 1):
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
            t = torch.randint(self.args.diffusion_total_steps, (X.shape[0], ), device=self.device).long()
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
        img_shape = (self.args.data_img_channels, self.args.data_img_size, self.args.data_img_size)
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
