import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import yaml
import tqdm
import math

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import models
from utils.logger import MessageLogger
from utils.data import build_dataset, build_dataloader
from utils.optimizer import build_optimizer, optimizer_to_device
from utils.dist import get_dist_info, master_only, init_dist, reduce_tensor
from utils.misc import get_device, init_seeds, create_log_directory, check_freq_epoch, check_freq_iteration, get_bare_model


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
        init_seeds(self.config.get('seed', 2001) + self.dist_info['global_rank'])

        # CREATE LOG DIRECTORY
        self.log_root = create_log_directory(self.config, config_path)
        if self.dist_info['is_dist']:
            object_list = [self.log_root]
            dist.broadcast_object_list(object_list, src=0)
            self.log_root = object_list[0]

        # INITIALIZE LOGGER
        self.logger = MessageLogger(
            log_root=self.log_root,
            use_tensorboard=self.config.get('use_tensorboard', False)
        )
        self.logger.info(f'Device: {self.device}')
        self.logger.info(f'Log directory: {self.log_root}')

        # DATA
        self.train_set, self.img_channels = build_dataset(
            name=self.config['dataset'],
            dataroot=self.config['dataroot'],
            img_size=self.config['img_size'],
            split='train',
        )
        self.train_loader = build_dataloader(
            dataset=self.train_set,
            batch_size=self.config['batch_size'],
            shuffle=True,
            num_workers=4,
            pin_memory=True,
        )
        self.logger.info(f'Size of training set: {len(self.train_set)}')
        self.logger.info(f'Each epoch has {len(self.train_loader)} iterations')

        # BUILD MODELS, OPTIMIZERS AND SCHEDULERS
        self.DiffusionModel = models.DDPM(
            total_steps=self.config['total_steps'],
            beta_schedule=self.config['beta_schedule'],
            beta_start=self.config['beta_start'],
            beta_end=self.config['beta_end'],
            objective=self.config['objective']
        )
        self.model = models.UNet(
            img_channels=self.img_channels,
            dim=self.config['dim'],
            dim_mults=self.config['dim_mults'],
            use_attn=self.config['use_attn'],
            num_res_blocks=self.config['num_res_blocks'],
            resblock_groups=self.config['resblock_groups'],
            attn_groups=self.config['attn_groups'],
            attn_heads=self.config['attn_heads'],
            dropout=self.config['dropout'],
        )
        self.model.to(device=self.device)
        self.ema = models.EMA(self.model, decay=self.config['ema_decay'])
        self.optimizer = build_optimizer(self.model.parameters(), cfg=self.config['optimizer'])

        # pretrained
        if self.config.get('pretrained'):
            self.load_model(self.config['pretrained'])

        # resume
        self.start_epoch = 0
        if self.config.get('resume'):
            self.load_ckpt(self.config['resume'])

        # distributed
        if self.dist_info['is_dist']:
            self.model = DDP(self.model, device_ids=[self.dist_info['local_rank']], output_device=self.dist_info['local_rank'])

        # TEST SAMPLES
        assert self.config['n_samples'] % self.dist_info['world_size'] == 0

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
        self.logger.info(f'Restart at epoch {self.start_epoch}')

    @master_only
    def save_ckpt(self, save_path: str, epoch: int):
        state_dicts = dict(
            model=get_bare_model(self.model).state_dict(),
            ema=self.ema.state_dict(),
            optimizer=self.optimizer.state_dict(),
            epoch=epoch,
        )
        torch.save(state_dicts, save_path)

    def run(self):
        self.logger.info('Start training...')

        # start of epoch
        for ep in range(self.start_epoch, self.config['epochs']):
            self.model.train()
            if self.dist_info['is_dist']:
                self.train_loader.sampler.set_epoch(ep)

            # start of iter
            pbar = tqdm.tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120, disable=not self.dist_info['is_master'])
            for it, X in enumerate(pbar):
                X = X[0] if isinstance(X, (tuple, list)) else X
                X = X.to(device=self.device, dtype=torch.float32)

                t = torch.randint(self.config['total_steps'], (X.shape[0], ), device=self.device).long()
                loss = self.DiffusionModel.loss_func(self.model, X0=X, t=t)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                self.ema.update()

                torch.cuda.synchronize()
                loss = reduce_tensor(loss.detach())
                pbar.set_postfix({'loss': loss.item()})
                if check_freq_iteration(self.config.get('log_freq'), it, len(self.train_loader)):
                    train_status = dict(loss=loss.item())
                    self.logger.track_status('Train', train_status, epoch=ep, iteration=it, n_iters_per_epoch=len(self.train_loader))
            pbar.close()
            # end of iter

            if check_freq_epoch(self.config.get('save_freq'), ep):
                self.save_ckpt(os.path.join(self.log_root, 'ckpt', f'epoch{ep}.pt'), epoch=ep)
            if check_freq_epoch(self.config.get('sample_freq'), ep):
                self.ema.apply_shadow()
                self.sample(os.path.join(self.log_root, 'samples', f'epoch{ep}.png'))
                self.ema.restore()
            if self.dist_info['is_dist']:
                dist.barrier()
        # end of epoch

        self.logger.info('End of traning')
        self.logger.close()

    @torch.no_grad()
    def sample_single_device(self, num: int):
        model = get_bare_model(self.model)
        model.eval()
        samples = []
        for i in range(math.ceil(num / self.config['batch_size'])):
            n = min(self.config['batch_size'], num - i * self.config['batch_size'])
            X = self.DiffusionModel.sample(
                model=model,
                shape=(n, self.img_channels, self.config['img_size'], self.config['img_size']),
                with_tqdm=self.dist_info['is_master'],
            ).clamp_(-1, 1).to(device=self.device)
            samples.append(X)
        samples = torch.cat(samples, dim=0)
        return samples

    @torch.no_grad()
    def sample(self, savepath: str):
        num_each_device = self.config['n_samples'] // self.dist_info['world_size']
        shape = (num_each_device, self.img_channels, self.config['img_size'], self.config['img_size'])
        samples = self.sample_single_device(num_each_device)
        assert samples.shape == shape
        if self.dist_info['is_dist']:
            sample_list = [torch.zeros(shape, device=self.device) for _ in range(self.dist_info['world_size'])]
            dist.all_gather(sample_list, samples)
            samples = torch.cat(sample_list, dim=0)
        if self.dist_info['is_master']:
            nrow = math.floor(math.sqrt(self.config['n_samples']))
            save_image(samples, savepath, nrow=nrow, normalize=True, value_range=(-1, 1))
