import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import tqdm
import math

import torch
from torch.utils.data import Subset
from torch.nn.parallel import DistributedDataParallel as DDP
import torchvision.transforms as T
from torchvision.utils import save_image
from torchmetrics.image import FrechetInceptionDistance, InceptionScore

import models
from datasets import build_dataset, ImageDir
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

        # DATA
        self.dataset = build_dataset(
            dataset=self.config.data.dataset,
            dataroot=self.config.data.dataroot,
            img_size=self.config.data.img_size,
            split='train',
        )
        self.logger.info(f'Size of test set: {len(self.dataset)}')

        # BUILD MODELS, OPTIMIZERS AND SCHEDULERS
        self.DiffusionModel = models.DDIM(
            total_steps=self.config.model.total_steps,
            beta_schedule=self.config.model.beta_schedule,
            beta_start=self.config.model.beta_start,
            beta_end=self.config.model.beta_end,
            objective=self.config.model.objective,
            eta=self.config.model.eta,
            skip_type=self.config.model.skip_type,
            skip_steps=self.config.model.skip_steps,
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
    def evaluate(self):
        self.logger.info('Start evaluating...')
        cfg = self.config.evaluate

        # ground-truth data
        test_set = self.dataset
        if cfg.num_real < len(self.dataset):
            test_set = Subset(self.dataset, torch.randperm(len(self.dataset))[:cfg.num_real])
            self.logger.info(f"Use a subset of test set, {cfg.num_real}/{len(self.dataset)}")
        else:
            self.logger.warning(f'Size of test set <= num_real, ignore num_real')

        # generated data
        gen_set = ImageDir(
            root=cfg.generated_samples,
            split='',
            transform=T.Compose([T.Resize((self.config.data.img_size, self.config.data.img_size)), T.ToTensor()]),
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
