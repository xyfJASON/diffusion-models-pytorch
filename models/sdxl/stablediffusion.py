from typing import List, Mapping, Any, Dict
from omegaconf import OmegaConf

import torch
from torch import Tensor

from ..base_latent import BaseLatent
from utils.misc import instantiate_from_config


class StableDiffusion(BaseLatent):
    def __init__(
            self,
            conditioner_config: OmegaConf,
            vae_config: OmegaConf,
            unet_config: OmegaConf,
            scale_factor: float = 0.13025,
            low_vram_shift_enabled: bool = False,
    ):
        super().__init__(scale_factor=scale_factor)

        self.conditioner = instantiate_from_config(conditioner_config)
        self.vae = instantiate_from_config(vae_config)
        self.unet = instantiate_from_config(unet_config)

        self.low_vram_shift_enabled = low_vram_shift_enabled

    def encode_latent(self, x: Tensor):
        if self.low_vram_shift_enabled:
            self.conditioner.to('cpu')
            self.unet.to('cpu')
            self.vae.to(self.device)
            torch.cuda.empty_cache()
        z = self.vae.encode(x)
        return self.scale_factor * z

    def decode_latent(self, z: Tensor):
        if self.low_vram_shift_enabled:
            self.conditioner.to('cpu')
            self.unet.to('cpu')
            self.vae.to(self.device)
            torch.cuda.empty_cache()
        z = 1. / self.scale_factor * z
        return self.vae.decode(z)

    def conditioner_forward(self, text: List[str], H: int, W: int):
        if self.low_vram_shift_enabled:
            self.vae.to('cpu')
            self.unet.to('cpu')
            self.conditioner.to(self.device)
            torch.cuda.empty_cache()
        batch = dict(
            txt=text,
            original_size_as_tuple=torch.tensor([1024, 1024], device=self.device).repeat(len(text), 1),
            crop_coords_top_left=torch.tensor([0, 0], device=self.device).repeat(len(text), 1),
            target_size_as_tuple=torch.tensor([H, W], device=self.device).repeat(len(text), 1),
        )
        return self.conditioner(batch)

    def unet_forward(self, x: Tensor, timesteps: Tensor, context: Tensor, y: Tensor):
        if self.low_vram_shift_enabled:
            self.vae.to('cpu')
            self.conditioner.to('cpu')
            self.unet.to(self.device)
            torch.cuda.empty_cache()
        return self.unet(x, timesteps=timesteps, context=context, y=y)

    def forward(
            self, x: Tensor, timesteps: Tensor, condition_dict: Dict = None,
            text: List[str] = None, H: int = None, W: int = None,
    ):
        if condition_dict is None:
            if text is None or H is None or W is None:
                raise ValueError('text, H and W must be provided when `condition_dict` is not provided.')
            condition_dict = self.conditioner_forward(text, H, W)
        context = condition_dict.get('crossattn')
        y = condition_dict.get('vector')
        x = self.unet_forward(x, timesteps=timesteps, context=context, y=y)
        return x

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        # state_dict_conditioner = {k[12:]: v for k, v in state_dict.items() if k.startswith('conditioner.')}
        state_dict_vae = {k[18:]: v for k, v in state_dict.items() if k.startswith('first_stage_model.')}
        state_dict_unet = {k[22:]: v for k, v in state_dict.items() if k.startswith('model.diffusion_model.')}
        # self.conditioner.load_state_dict(state_dict_conditioner, strict=strict, assign=assign)
        self.vae.load_state_dict(state_dict_vae, strict=strict, assign=assign)
        self.unet.load_state_dict(state_dict_unet, strict=strict, assign=assign)
        # del state_dict_conditioner
        del state_dict_vae
        del state_dict_unet
