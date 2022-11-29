import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import yaml
import tqdm
import math

import torch
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image
from torchmetrics.image import FrechetInceptionDistance, InceptionScore

import models
from utils.logger import get_logger
from utils.data import build_dataset
from utils.dist import get_dist_info, master_only, init_dist
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
        init_seeds(self.config.get('seed', 2001) + self.dist_info['global_rank'])

        # INITIALIZE LOGGER
        self.logger = get_logger()
        self.logger.info(f'Device: {self.device}')

        # DATA
        self.test_set, self.img_channels = build_dataset(
            name=self.config['dataset'],
            dataroot=self.config['dataroot'],
            img_size=self.config['img_size'],
            split='test',
        )
        self.logger.info(f'Size of test set: {len(self.test_set)}')

        # BUILD MODELS
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
            resblock_groups=self.config['resblock_groups'],
            attn_groups=self.config['attn_groups'],
            attn_heads=self.config['attn_heads'],
        )
        self.model.to(device=self.device)
        self.model.eval()

        # pretrained
        self.load_model(self.config['pretrained'], load_ema=self.config.get('load_ema', True))

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
        cfg = self.config['evaluate']

        test_set = self.test_set
        if cfg.get('n_eval') is not None:
            if cfg['n_eval'] < len(self.test_set):
                test_set = Subset(self.test_set, torch.randperm(len(self.test_set))[:cfg['n_eval']])
                self.logger.info(f"Use a subset of test set, {cfg['n_eval']}/{len(self.test_set)}")
            else:
                self.logger.warning(f'Size of test set <= n_eval, ignore n_eval')

        metric_fid = FrechetInceptionDistance().to(self.device)
        metric_iscore = InceptionScore().to(self.device)
        self.logger.info('Sample real images for FID')
        for real_img in tqdm.tqdm(test_set, desc='Sampling', ncols=120):
            real_img = real_img[0] if isinstance(real_img, (tuple, list)) else real_img
            real_img = real_img.unsqueeze(0).to(self.device)
            real_img = ((real_img + 1) / 2 * 255).to(dtype=torch.uint8, device=self.device)
            metric_fid.update(real_img, real=True)

        self.logger.info('Sample fake images for FID and IS')
        for i in tqdm.tqdm(range(math.ceil(len(test_set) / cfg['batch_size'])), desc='Sampling', ncols=120):
            num = min(cfg['batch_size'], len(test_set) - i * cfg['batch_size'])
            fake_img = self.DiffusionModel.sample(
                model=self.model,
                shape=(num, self.img_channels, self.config['img_size'], self.config['img_size']),
                same_XT=cfg.get('same_XT', False),
            ).clamp_(-1, 1)
            fake_img = ((fake_img + 1) / 2 * 255).to(dtype=torch.uint8, device=self.device)
            metric_fid.update(fake_img, real=False)
            metric_iscore.update(fake_img)

        fid = metric_fid.compute().item()
        iscore = metric_iscore.compute()
        iscore = (iscore[0].item(), iscore[1].item())
        self.logger.info(f'fid: {fid}')
        self.logger.info(f'iscore: {iscore[0]} ({iscore[1]})')
        self.logger.info('End of evaluation')

    @master_only
    @torch.no_grad()
    def sample(self):
        self.logger.info('Start sampling...')
        model = get_bare_model(self.model)
        model.eval()

        cfg = self.config['sample']
        os.makedirs(cfg['save_dir'], exist_ok=True)

        for i in tqdm.tqdm(range(math.ceil(cfg['n_samples'] / cfg['batch_size'])), desc='Sampling', ncols=120):
            num = min(cfg['batch_size'], cfg['n_samples'] - i * cfg['batch_size'])
            X = self.DiffusionModel.sample(
                model=model,
                shape=(num, self.img_channels, self.config['img_size'], self.config['img_size']),
                same_XT=cfg.get('same_XT', False),
            ).clamp_(-1, 1)
            for j, x in enumerate(X):
                save_image([x.squeeze(0).cpu()],
                           os.path.join(cfg['save_dir'], str(i * cfg['batch_size'] + j) + '.png'),
                           nrow=1, normalize=True, value_range=(-1, 1))
        self.logger.info(f"Sampled images are saved to {cfg['save_dir']}")
        self.logger.info('End of sampling')

    @master_only
    @torch.no_grad()
    def sample_denoise(self):
        self.logger.info('Start sampling...')
        model = get_bare_model(self.model)
        model.eval()

        cfg = self.config['sample_denoise']
        os.makedirs(cfg['save_dir'], exist_ok=True)

        for i in tqdm.tqdm(range(math.ceil(cfg['n_samples'] / cfg['batch_size'])), desc='Sampling', ncols=120):
            num = min(cfg['batch_size'], cfg['n_samples'] - i * cfg['batch_size'])
            X = self.DiffusionModel.sample(
                model=model,
                shape=(num, self.img_channels, self.config['img_size'], self.config['img_size']),
                same_XT=cfg.get('same_XT', False),
                return_all=True,
            )
            result = [X[k].cpu() for k in range(0, len(X) - 1, (len(X) - 1 + 18) // 19)] + [X[-1].cpu()]
            result = torch.stack(result, dim=1)
            for j, r in enumerate(result):
                save_image(r,
                           os.path.join(cfg['save_dir'], str(i * cfg['batch_size'] + j) + '.png'),
                           nrow=20, normalize=True, value_range=(-1, 1))
        self.logger.info(f"Sampled images are saved to {cfg['save_dir']}")
        self.logger.info('End of sampling')
