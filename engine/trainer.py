import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import yaml
import tqdm
import math

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import models
from utils.logger import MessageLogger, AverageMeter
from utils.data import build_dataset, build_dataloader
from utils.optimizer import build_optimizer, optimizer_to_device
from utils.dist import get_dist_info, master_only, init_dist, reduce_tensor
from utils.misc import get_device, init_seeds, create_log_directory, check_freq, get_bare_model


class Trainer:
    def __init__(self, config_path: str):

        # READ CONFIGURATION FILE
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # INITIALIZE DISTRIBUTED MODE
        init_dist()
        self.dist_info = get_dist_info()
        self.device = get_device(self.dist_info)

        # INITIALIZE SEEDS
        init_seeds(self.config.get('seed', 2022) + self.dist_info['global_rank'])

        # CREATE LOG DIRECTORY
        self.log_root = create_log_directory(self.config['train'], config_path)
        if self.dist_info['is_dist']:
            object_list = [self.log_root]
            dist.broadcast_object_list(object_list, src=0)
            self.log_root = object_list[0]

        # INITIALIZE LOGGER
        self.logger = MessageLogger(
            log_root=self.log_root,
            print_freq=self.config['train']['print_freq'].get('by_step', 0),
            use_tensorboard=self.config['train'].get('use_tensorboard', False),
        )
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Log directory: {self.log_root}')

        # DATA
        self.train_set, self.img_channels = build_dataset(
            name=self.config['data']['dataset'],
            dataroot=self.config['data']['dataroot'],
            img_size=self.config['data']['img_size'],
            split='train',
        )
        self.img_shape = (self.img_channels, self.config['data']['img_size'], self.config['data']['img_size'])
        self.train_loader = build_dataloader(
            dataset=self.train_set,
            batch_size=self.config['train']['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.logger.info(f'Size of training set: {len(self.train_set)}')
        self.logger.info(f'Each epoch has {len(self.train_loader)} iterations')

        # BUILD MODELS, OPTIMIZERS AND SCHEDULERS
        self.DiffusionModel = models.DDPM(
            total_steps=self.config['model']['total_steps'],
            beta_schedule=self.config['model']['beta_schedule'],
            beta_start=self.config['model']['beta_start'],
            beta_end=self.config['model']['beta_end'],
            objective=self.config['model']['objective'],
            var_type=self.config['model']['var_type'],
        )
        self.model = models.UNet(
            img_channels=self.img_channels,
            dim=self.config['model']['dim'],
            dim_mults=self.config['model']['dim_mults'],
            use_attn=self.config['model']['use_attn'],
            num_res_blocks=self.config['model']['num_res_blocks'],
            resblock_groups=self.config['model']['resblock_groups'],
            attn_groups=self.config['model']['attn_groups'],
            attn_heads=self.config['model']['attn_heads'],
            dropout=self.config['model']['dropout'],
        )
        self.model.to(device=self.device)
        self.ema = models.EMA(self.model, decay=self.config['model']['ema_decay'])
        self.optimizer = build_optimizer(self.model.parameters(), cfg=self.config['train']['optimizer'])

        # pretrained
        if self.config['train'].get('pretrained'):
            self.load_model(self.config['train']['pretrained'])

        # resume
        self.start_epoch = 0
        self.global_step = -1
        if self.config['train'].get('resume'):
            self.load_ckpt(self.config['train']['resume'])

        # distributed
        if self.dist_info['is_dist']:
            self.model = DDP(
                module=self.model,
                device_ids=[self.dist_info['local_rank']],
                output_device=self.dist_info['local_rank'],
            )

        # TEST SAMPLES
        if self.config['train']['n_samples'] % self.dist_info['world_size'] != 0:
            raise ValueError('Number of samples should be divisible by WORLD_SIZE!')

    def load_model(self, model_path: str):
        ckpt = torch.load(model_path, map_location='cpu')
        if ckpt.get('model'):
            self.model.load_state_dict(ckpt['model'])
            self.model.to(device=self.device)
            self.logger.info(f'Successfully load model from {model_path}')
        else:
            self.logger.warning(f'Fail to load model from {model_path}')
        if ckpt.get('ema'):
            self.ema.load_state_dict(ckpt['ema'], device=self.device)
            self.logger.info(f'Successfully load ema from {model_path}')
        else:
            self.logger.warning(f'Fail to load ema from {model_path}')

    @master_only
    def save_model(self, save_path: str):
        state_dicts = dict(
            model=get_bare_model(self.model).state_dict(),
            ema=self.ema.state_dict(),
        )
        torch.save(state_dicts, save_path)

    def load_ckpt(self, ckpt_path: str):
        self.logger.info(f'Resuming from {ckpt_path}...')
        self.load_model(ckpt_path)
        ckpt = torch.load(ckpt_path, map_location='cpu')
        self.optimizer.load_state_dict(ckpt['optimizer'])
        optimizer_to_device(self.optimizer, self.device)
        self.start_epoch = ckpt['epoch'] + 1
        self.global_step = ckpt['global_step']
        self.logger.info(f'Restart at epoch {self.start_epoch}')

    @master_only
    def save_ckpt(self, save_path: str, epoch: int):
        state_dicts = dict(
            model=get_bare_model(self.model).state_dict(),
            ema=self.ema.state_dict(),
            optimizer=self.optimizer.state_dict(),
            epoch=epoch,
            global_step=self.global_step,
        )
        torch.save(state_dicts, save_path)

    def run(self):
        self.logger.info('Start training...')

        # start of epoch
        for ep in range(self.start_epoch, self.config['train']['epochs']):
            self.model.train()
            if self.dist_info['is_dist']:
                self.train_loader.sampler.set_epoch(ep)

            # start of iter
            loss_meter = AverageMeter()
            pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120, disable=not self.dist_info['is_master'])
            for it, X in enumerate(pbar):
                self.model.train()

                X = X[0] if isinstance(X, (tuple, list)) else X
                X = X.to(device=self.device, dtype=torch.float32)

                t = torch.randint(self.config['model']['total_steps'], (X.shape[0], ), device=self.device).long()
                loss = self.DiffusionModel.loss_func(self.model, X0=X, t=t)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.)
                self.optimizer.step()
                self.ema.update()

                self.global_step += 1
                loss = reduce_tensor(loss.detach())
                loss_meter.update(loss.item())
                pbar.set_postfix({'loss': loss_meter.avg})
                train_status = dict(loss=loss.item())
                self.logger.track_status('Train', train_status, global_step=self.global_step, epoch=ep, iteration=it)
                if check_freq(self.config['train']['save_freq'].get('by_step'), self.global_step):
                    self.save_ckpt(os.path.join(self.log_root, 'ckpt', f'epoch{ep}_iter{it}.pt'), epoch=ep)
                if check_freq(self.config['train']['sample_freq'].get('by_step'), self.global_step):
                    self.ema.apply_shadow()
                    self.sample(os.path.join(self.log_root, 'samples', f'epoch{ep}_iter{it}.png'))
                    self.ema.restore()
                if self.dist_info['is_dist']:
                    dist.barrier()
            pbar.close()
            # end of iter

            if check_freq(self.config['train']['save_freq'].get('by_epoch'), ep):
                self.save_ckpt(os.path.join(self.log_root, 'ckpt', f'epoch{ep}.pt'), epoch=ep)
            if check_freq(self.config['train']['sample_freq'].get('by_epoch'), ep):
                self.ema.apply_shadow()
                self.sample(os.path.join(self.log_root, 'samples', f'epoch{ep}.png'))
                self.ema.restore()
            if self.dist_info['is_dist']:
                dist.barrier()
        # end of epoch

        self.save_model(os.path.join(self.log_root, 'model.pt'))
        self.logger.info('End of traning')
        self.logger.close()

    @torch.no_grad()
    def sample_single_device(self, num: int):
        model = get_bare_model(self.model)
        model.eval()
        samples = []
        total = math.ceil(num / self.config['train']['batch_size'])
        for i in range(total):
            n = min(self.config['train']['batch_size'], num - i * self.config['train']['batch_size'])
            init_noise = torch.randn((n, *self.img_shape), device=self.device)
            X = self.DiffusionModel.sample(
                model=model,
                init_noise=init_noise,
                with_tqdm=self.dist_info['is_master'],
                desc=f'Sampling({i+1}/{total})',
                ncols=120,
            ).clamp_(-1, 1)
            samples.append(X)
        samples = torch.cat(samples, dim=0)
        return samples

    @torch.no_grad()
    def sample(self, savepath: str):
        num_each_device = self.config['train']['n_samples'] // self.dist_info['world_size']
        samples = self.sample_single_device(num_each_device)
        if self.dist_info['is_dist']:
            sample_list = [torch.Tensor() for _ in range(self.dist_info['world_size'])]
            dist.all_gather_object(sample_list, samples)
            samples = torch.cat(sample_list, dim=0)
        samples = samples.cpu()
        if self.dist_info['is_master']:
            nrow = math.floor(math.sqrt(self.config['train']['n_samples']))
            save_image(samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))
