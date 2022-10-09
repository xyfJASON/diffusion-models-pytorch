import os
import yaml
import argparse
from tqdm import tqdm

import torch
from torch.utils.tensorboard import SummaryWriter
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torchvision.utils import save_image

import models
from models.modules import UNet
from utils.data import build_dataset, build_dataloader
from utils.optimizer import build_optimizer
from utils.train_utils import reduce_tensor, set_device, create_log_directory


class Trainer:
    def __init__(self, config_path: str):
        # ====================================================== #
        # READ CONFIGURATION FILE
        # ====================================================== #
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        # ====================================================== #
        # SET DEVICE
        # ====================================================== #
        self.device, self.world_size, self.local_rank, self.global_rank = set_device()
        self.is_master = self.world_size <= 1 or self.global_rank == 0
        self.is_ddp = self.world_size > 1
        print('using device:', self.device)

        # ====================================================== #
        # CREATE LOG DIRECTORY
        # ====================================================== #
        if self.is_master:
            self.log_root = create_log_directory(self.config, config_path)

        # ====================================================== #
        # TENSORBOARD
        # ====================================================== #
        if self.is_master:
            os.makedirs(os.path.join(self.log_root, 'tensorboard'), exist_ok=True)
            self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

        # ====================================================== #
        # DATA
        # ====================================================== #
        train_dataset, self.img_channels = build_dataset(self.config['dataset'], dataroot=self.config['dataroot'], img_size=self.config['img_size'], split='train')
        self.train_loader = build_dataloader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True, is_ddp=self.is_ddp)

        # ====================================================== #
        # BUILD MODELS AND OPTIMIZERS
        # ====================================================== #
        self.DiffusionModel = models.DDPM(self.config['total_steps'], self.config['beta_schedule_mode'])
        self.model = UNet(self.img_channels, self.config['img_size'], self.config['dim'], self.config['dim_mults'])
        self.model.to(device=self.device)
        self.optimizer = build_optimizer(self.model.parameters(), cfg=self.config['optimizer'])
        # distributed
        if self.is_ddp:
            self.model = DDP(self.model, device_ids=[self.local_rank], output_device=self.local_rank)

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.model.load_state_dict(ckpt['model'])
        self.model.to(device=self.device)

    def save_model(self, model_path):
        model = self.model.module if self.is_ddp else self.model
        torch.save({'model': model.state_dict()}, model_path)

    def train(self):
        print('==> Training...')
        for ep in range(self.config['epochs']):
            if self.is_ddp:
                dist.barrier()
                self.train_loader.sampler.set_epoch(ep)

            self.train_one_epoch(ep)

            if self.is_master:
                if self.config.get('save_freq') and (ep + 1) % self.config['save_freq'] == 0:
                    self.save_model(os.path.join(self.log_root, 'ckpt', f'epoch_{ep}.pt'))
                if self.config.get('sample_freq') and (ep + 1) % self.config['sample_freq'] == 0:
                    self.sample(os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))

        if self.is_master:
            self.save_model(os.path.join(self.log_root, 'model.pt'))
            self.writer.close()

    def train_one_epoch(self, ep):
        self.model.train()

        pbar = tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120) if self.is_master else self.train_loader
        for it, X in enumerate(pbar):
            if isinstance(X, (tuple, list)):
                X = X[0]
            X = X.to(device=self.device, dtype=torch.float32)

            t = torch.randint(self.config['total_steps'], (X.shape[0], ), device=self.device).long()
            loss = self.DiffusionModel.loss_func(self.model, X0=X, t=t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            if self.is_ddp:
                loss = reduce_tensor(loss.detach(), self.world_size)
            if self.is_master:
                self.writer.add_scalar('Train/loss', loss.item(), it + ep * len(self.train_loader))
                pbar.set_postfix({'loss': loss.item()})

        if self.is_master:
            pbar.close()

    @torch.no_grad()
    def sample(self, savepath):
        model = self.model.module if self.is_ddp else self.model
        model.eval()
        X = self.DiffusionModel.sample(model, shape=(64, self.img_channels, self.config['img_size'], self.config['img_size']))
        X = X[-1].cpu()
        save_image(X, savepath, nrow=8, normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yml', help='path to training configuration file')
    args = parser.parse_args()

    trainer = Trainer(args.config_path)
    trainer.train()


if __name__ == '__main__':
    main()
