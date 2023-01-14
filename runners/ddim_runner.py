import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tqdm
import math

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import models
import diffusions
from utils.logger import MessageLogger
from utils.dist import get_dist_info, master_only, init_dist
from utils.misc import get_device, init_seeds, get_bare_model, dict2namespace


class DDIMRunner:
    def __init__(self, args, config):
        self.config = config

        # INITIALIZE DISTRIBUTED MODE
        init_dist()
        self.dist_info = dict2namespace(get_dist_info())
        self.device = get_device(self.dist_info)

        # INITIALIZE SEEDS
        init_seeds(getattr(self.config, 'seed', 2001) + self.dist_info.global_rank)

        # INITIALIZE LOGGER
        self.logger = MessageLogger(log_root=None, print_freq=0)
        self.logger.info(f'Device: {self.device}')

        # BUILD MODELS, OPTIMIZERS AND SCHEDULERS
        self.DiffusionModel = diffusions.DDIM(
            total_steps=self.config.ddim.total_steps,
            beta_schedule=self.config.ddim.beta_schedule,
            beta_start=self.config.ddim.beta_start,
            beta_end=self.config.ddim.beta_end,
            objective=self.config.ddim.objective,
            eta=self.config.ddim.eta,
            skip_type=self.config.ddim.skip_type,
            skip_steps=self.config.ddim.skip_steps,
        )
        self.model = models.UNet(
            img_channels=self.config.data.img_channels,
            dim=self.config.model.dim,
            dim_mults=self.config.model.dim_mults,
            use_attn=self.config.model.use_attn,
            num_res_blocks=self.config.model.num_res_blocks,
            resblock_groups=self.config.model.resblock_groups,
            attn_groups=self.config.model.attn_groups,
            attn_heads=self.config.model.attn_heads,
            dropout=self.config.model.dropout,
        )
        self.model.to(device=self.device)
        self.ema = models.EMA(self.model, decay=self.config.model.ema_decay)

        # LOAD PRETRAINED WEIGHTS / RESUME
        cfg = getattr(config, args.func)
        if getattr(cfg, 'model_path', None):
            self.load_model(cfg.model_path, load_ema=cfg.load_ema)

        # DISTRIBUTED
        if self.dist_info.is_dist:
            self.model = DDP(
                module=self.model,
                device_ids=[self.dist_info.local_rank],
                output_device=self.dist_info.local_rank,
            )

        # TEST SAMPLES
        if args.func == 'train':
            if self.config.train.n_samples % self.dist_info.world_size != 0:
                raise ValueError('Number of samples should be divisible by WORLD_SIZE!')

    def load_model(self, model_path: str, load_ema: bool = False):
        ckpt = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(ckpt['ema']['shadow'] if load_ema else ckpt['model'])
        self.model.to(device=self.device)
        self.logger.info(f'Successfully load model from {model_path}')

        self.ema.load_state_dict(ckpt['ema'], device=self.device)
        self.logger.info(f'Successfully load ema from {model_path}')

    @master_only
    def save_model(self, save_path: str):
        state_dicts = dict(
            model=get_bare_model(self.model).state_dict(),
            ema=self.ema.state_dict(),
        )
        torch.save(state_dicts, save_path)

    @torch.no_grad()
    def sample(self):
        cfg = self.config.sample
        self.logger.info('Start sampling...')
        self.logger.info(f'Samples will be saved to {cfg.save_dir}')
        os.makedirs(cfg.save_dir, exist_ok=True)
        num_each_device = cfg.n_samples // self.dist_info.world_size
        model = get_bare_model(self.model).eval()

        total = math.ceil(num_each_device / cfg.batch_size)
        img_shape = (self.config.data.img_channels, self.config.data.img_size, self.config.data.img_size)
        for i in range(total):
            n = min(cfg.batch_size, num_each_device - i * cfg.batch_size)
            init_noise = torch.randn((n, *img_shape), device=self.device)
            X = self.DiffusionModel.sample(
                model=model,
                init_noise=init_noise,
                with_tqdm=self.dist_info.is_master,
                desc=f'Sampling({i+1}/{total})',
                ncols=120,
            ).clamp(-1, 1)
            for j, x in enumerate(X):
                idx = self.dist_info.global_rank * num_each_device + i * cfg.batch_size + j
                save_image(
                    tensor=x.cpu(), fp=os.path.join(cfg.save_dir, f'{idx}.png'),
                    nrow=1, normalize=True, value_range=(-1, 1),
                )
        self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
        self.logger.info('End of sampling')

    @torch.no_grad()
    def sample_interpolate(self):
        cfg = self.config.sample_interpolate
        self.logger.info('Start sampling...')
        self.logger.info(f'Samples will be saved to {cfg.save_dir}')
        os.makedirs(cfg.save_dir, exist_ok=True)
        num_each_device = cfg.n_samples // self.dist_info.world_size
        model = get_bare_model(self.model).eval()

        def slerp(t, z1, z2):  # noqa
            theta = torch.acos(torch.sum(z1 * z2) / (torch.linalg.norm(z1) * torch.linalg.norm(z2)))
            return torch.sin((1 - t) * theta) / torch.sin(theta) * z1 + torch.sin(t * theta) / torch.sin(theta) * z2

        total = math.ceil(num_each_device / cfg.batch_size)
        img_shape = (self.config.data.img_channels, self.config.data.img_size, self.config.data.img_size)
        for i in range(total):
            n = min(cfg.batch_size, num_each_device - i * cfg.batch_size)
            z1 = torch.randn((n, *img_shape), device=self.device)
            z2 = torch.randn((n, *img_shape), device=self.device)
            results = []
            for t in tqdm.tqdm(torch.linspace(0, 1, cfg.n_interpolate), desc=f'Sampling({i+1}/{total})', ncols=120):
                X = self.DiffusionModel.sample(
                    model=model,
                    init_noise=slerp(t, z1, z2),
                ).clamp(-1, 1)
                results.append(X)
            results = torch.stack(results, dim=1)
            for j, x in enumerate(results):
                idx = self.dist_info.global_rank * num_each_device + i * cfg.batch_size + j
                save_image(
                    tensor=x.cpu(), fp=os.path.join(cfg.save_dir, f'{idx}.png'),
                    nrow=len(x), normalize=True, value_range=(-1, 1),
                )
        self.logger.info(f"Sampled images are saved to {cfg.save_dir}")
        self.logger.info('End of sampling')
