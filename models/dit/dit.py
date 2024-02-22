from typing import Mapping, Any
from omegaconf import OmegaConf

from torch import Tensor

from ..base_latent import BaseLatent
from utils.misc import instantiate_from_config


class DiT(BaseLatent):
    def __init__(
            self,
            vae_config: OmegaConf,
            vit_config: OmegaConf,
            scale_factor: float = 0.18215,
    ):
        super().__init__(scale_factor=scale_factor)

        self.vae = instantiate_from_config(vae_config)
        self.vit = instantiate_from_config(vit_config)

    def decode_latent(self, z: Tensor):
        z = 1. / self.scale_factor * z
        return self.vae.decode(z).sample

    def vit_forward(self, x: Tensor, t: Tensor, y: Tensor):
        return self.vit(x, t, y)

    def forward(self, x: Tensor, timesteps: Tensor, y: Tensor = None):
        return self.vit_forward(x, timesteps, y)

    def load_state_dict(self, state_dict: Mapping[str, Any], strict: bool = True, assign: bool = False):
        self.vit.load_state_dict(state_dict, strict=strict, assign=assign)
