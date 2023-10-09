import torch
import torch.nn as nn
from .unet import UNetModel


class UNetCombined(nn.Module):
    """ Combines a conditional-only model and an unconditional-only model into a single nn.Module.

    The guided diffusion models proposed by OpenAI are trained to be either conditional or unconditional,
    leading to difficulties if we want to use their pretrained models in classifier-free guidance. This
    class wraps a conditional model and an unconditional model, and decides which one to use based on the
    input class label.

    """
    def __init__(self, *args, **kwargs):
        super().__init__()
        assert kwargs.get('num_classes') is not None
        self.unet_cond = UNetModel(*args, **kwargs)
        kwargs_uncond = kwargs.copy()
        kwargs_uncond.update({'num_classes': None})
        self.unet_uncond = UNetModel(*args, **kwargs_uncond)

    def forward(self, x, timesteps, y=None):
        unet = self.unet_uncond if y is None else self.unet_cond
        return unet(x, timesteps, y)

    def combine_weights(self, cond_path, uncond_path, save_path):
        ckpt_cond = torch.load(cond_path, map_location='cpu')
        ckpt_uncond = torch.load(uncond_path, map_location='cpu')
        self.unet_cond.load_state_dict(ckpt_cond)
        self.unet_uncond.load_state_dict(ckpt_uncond)
        torch.save(self.state_dict(), save_path)
