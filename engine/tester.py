import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import yaml
import tqdm
import math
from typing import Dict

import torch
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T
from torchvision.utils import save_image
from torchmetrics.image import FrechetInceptionDistance, InceptionScore

import models
from datasets import ImageDir
from utils.logger import get_logger
from utils.data import build_dataset
from utils.dist import get_dist_info, init_dist
from utils.misc import get_device, init_seeds, get_bare_model


class Tester:
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

        # INITIALIZE LOGGER
        self.logger = get_logger()
        self.logger.info(f'Device: {self.device}')

        # DATA
        self.test_set, self.img_channels = build_dataset(
            name=self.config['data']['dataset'],
            dataroot=self.config['data']['dataroot'],
            img_size=self.config['data']['img_size'],
            split='train',
        )
        self.img_shape = (self.img_channels, self.config['data']['img_size'], self.config['data']['img_size'])
        self.logger.info(f'Size of test set: {len(self.test_set)}')

        # BUILD MODELS
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
        self.model.eval()

        # pretrained
        self.load_model(self.config['test']['pretrained'], load_ema=self.config['test']['load_ema'])

        # distributed
        if self.dist_info['is_dist']:
            self.model = DDP(self.model, device_ids=[self.dist_info['local_rank']], output_device=self.dist_info['local_rank'])

    def load_model(self, model_path: str, load_ema: bool = True):
        ckpt = torch.load(model_path, map_location='cpu')
        state_dict = ckpt.get('ema')['shadow'] if load_ema else ckpt.get('model')
        if state_dict is None:
            self.logger.warning(f'Fail to load model from {model_path}')
            sys.exit(1)
        else:
            self.model.load_state_dict(state_dict)
            self.logger.info(f'Successfully load model from {model_path}')

    @torch.no_grad()
    def evaluate(self):
        self.logger.info('Start evaluating...')
        cfg = self.config['test']['evaluate']

        # ground-truth data
        test_set = self.test_set
        if cfg.get('num_real') is not None:
            if cfg['num_real'] < len(self.test_set):
                test_set = Subset(self.test_set, torch.randperm(len(self.test_set))[:cfg['num_real']])
                self.logger.info(f"Use a subset of test set, {cfg['num_real']}/{len(self.test_set)}")
            else:
                self.logger.warning(f'Size of test set <= num_real, ignore num_real')

        # generated data
        gen_set = ImageDir(
            root=cfg['generated_samples'],
            split='',
            transform=T.Compose([T.Resize(self.img_shape[1:]), T.ToTensor()]),
        )

        metric_fid = FrechetInceptionDistance().to(self.device)
        metric_iscore = InceptionScore().to(self.device)

        self.logger.info('Sample real images for FID')
        for real_img in tqdm.tqdm(test_set, desc='Sampling', ncols=120):
            real_img = real_img[0] if isinstance(real_img, (tuple, list)) else real_img
            real_img = real_img.unsqueeze(0).to(self.device)
            real_img = ((real_img + 1) / 2 * 255).to(dtype=torch.uint8)
            metric_fid.update(real_img, real=True)

        self.logger.info('Sample fake images for FID and IS')
        for fake_img in tqdm.tqdm(gen_set, desc='Sampling', ncols=120):
            fake_img = fake_img.unsqueeze(0).to(self.device)
            fake_img = (fake_img * 255).to(dtype=torch.uint8)
            metric_fid.update(fake_img, real=False)
            metric_iscore.update(fake_img)

        fid = metric_fid.compute().item()
        iscore = metric_iscore.compute()
        iscore = (iscore[0].item(), iscore[1].item())
        self.logger.info(f'fid: {fid}')
        self.logger.info(f'iscore: {iscore[0]} ({iscore[1]})')
        self.logger.info('End of evaluation')

    @torch.no_grad()
    def sample_single_device(self, num: int, return_freq: int, cfg: Dict):
        model = get_bare_model(self.model)
        model.eval()
        total = math.ceil(num / cfg['batch_size'])
        for i in range(total):
            n = min(cfg['batch_size'], num - i * cfg['batch_size'])
            init_noise = torch.randn((n, *self.img_shape), device=self.device)
            X = self.DiffusionModel.sample(
                model=model,
                init_noise=init_noise,
                return_freq=return_freq,
                with_tqdm=self.dist_info['is_master'],
                desc=f'Sampling({i+1}/{total})',
                ncols=120,
            )
            if return_freq:
                X = torch.stack(X, dim=1)
            for j, x in enumerate(X):
                idx = self.dist_info['global_rank'] * num + i * cfg['batch_size'] + j
                save_image(x.clamp_(-1, 1).cpu(),
                           os.path.join(cfg['save_dir'], f'{idx}.png'),
                           nrow=len(x) if return_freq else 1, normalize=True, value_range=(-1, 1))

    @torch.no_grad()
    def sample(self):
        self.logger.info('Start sampling...')
        cfg = self.config['test']['sample']
        os.makedirs(cfg['save_dir'], exist_ok=True)
        num_each_device = cfg['n_samples'] // self.dist_info['world_size']
        self.sample_single_device(num_each_device, return_freq=0, cfg=cfg)
        self.logger.info(f"Sampled images are saved to {cfg['save_dir']}")
        self.logger.info('End of sampling')

    @torch.no_grad()
    def sample_denoise(self):
        self.logger.info('Start sampling...')
        cfg = self.config['test']['sample_denoise']
        os.makedirs(cfg['save_dir'], exist_ok=True)
        num_each_device = cfg['n_samples'] // self.dist_info['world_size']
        self.sample_single_device(num_each_device, return_freq=cfg['return_freq'], cfg=cfg)
        self.logger.info(f"Sampled images are saved to {cfg['save_dir']}")
        self.logger.info('End of sampling')
