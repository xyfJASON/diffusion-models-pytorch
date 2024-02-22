from typing import List, Mapping, Any
from omegaconf import OmegaConf

import torch
from torch import Tensor

from .distributions import DiagonalGaussianDistribution
from ..base_latent import BaseLatent
from utils.misc import instantiate_from_config


class StableDiffusion(BaseLatent):
    def __init__(
            self,
            text_encoder_config: OmegaConf,
            vae_config: OmegaConf,
            unet_config: OmegaConf,
            scale_factor: float = 0.18215,
            low_vram_shift_enabled: bool = False,
    ):
        super().__init__(scale_factor=scale_factor)

        self.text_encoder = instantiate_from_config(text_encoder_config)
        self.vae = instantiate_from_config(vae_config)
        self.unet = instantiate_from_config(unet_config)

        self.low_vram_shift_enabled = low_vram_shift_enabled

    def encode_latent(self, x: Tensor):
        if self.low_vram_shift_enabled:
            self.text_encoder.to('cpu')
            self.unet.to('cpu')
            self.vae.to(self.device)
            torch.cuda.empty_cache()
        z = self.vae.encode(x)
        if isinstance(z, DiagonalGaussianDistribution):
            z = z.sample()
        return self.scale_factor * z

    def decode_latent(self, z: Tensor):
        if self.low_vram_shift_enabled:
            self.text_encoder.to('cpu')
            self.unet.to('cpu')
            self.vae.to(self.device)
            torch.cuda.empty_cache()
        z = 1. / self.scale_factor * z
        return self.vae.decode(z)

    def text_encoder_encode(self, text: List[str]):
        if self.low_vram_shift_enabled:
            self.vae.to('cpu')
            self.unet.to('cpu')
            self.text_encoder.to(self.device)
            torch.cuda.empty_cache()
        return self.text_encoder.encode(text)

    def unet_forward(self, x: Tensor, timesteps: Tensor, context: Tensor):
        if self.low_vram_shift_enabled:
            self.vae.to('cpu')
            self.text_encoder.to('cpu')
            self.unet.to(self.device)
            torch.cuda.empty_cache()
        return self.unet(x, timesteps=timesteps, context=context)

    def forward(self, x: Tensor, timesteps: Tensor, text_embed: Tensor = None, text: List[str] = None):
        if text_embed is None and text is None:
            raise ValueError('Either `text_embed` or `text` must be provided.')
        if text_embed is None:
            text_embed = self.text_encoder_encode(text)
        x = self.unet_forward(x, timesteps=timesteps, context=text_embed)
        return x

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        state_dict_vae = {k[18:]: v for k, v in state_dict.items() if k.startswith('first_stage_model.')}
        state_dict_unet = {k[22:]: v for k, v in state_dict.items() if k.startswith('model.diffusion_model.')}
        self.vae.load_state_dict(state_dict_vae, strict=strict)
        self.unet.load_state_dict(state_dict_unet, strict=strict)
        del state_dict_vae
        del state_dict_unet
