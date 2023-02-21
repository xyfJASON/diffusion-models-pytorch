from typing import List

import torch
import torch.nn as nn
from torch import Tensor

from diffusions import DDPM


class DDIM(DDPM):
    def __init__(self,
                 betas: Tensor = None,
                 objective: str = 'pred_eps',
                 var_type: str = 'fixed_large',
                 eta: float = 0.):
        super().__init__(betas, objective, var_type)
        self.eta = eta
        self.alphas_cumprod_next = torch.cat((self.alphas_cumprod[1:], torch.zeros(1, dtype=torch.float64)))

    def _pred_eps_from_X0(self, Xt: Tensor, t: Tensor, X0: Tensor):
        return ((self._extract(self.sqrt_recip_alphas_cumprod, t) * Xt - X0) /
                self._extract(self.sqrt_recipm1_alphas_cumprod, t))

    @torch.no_grad()
    def ddim_p_sample(self, model: nn.Module, Xt: Tensor, t: Tensor, clip_denoised: bool = True, eta: float = None):
        """ Sample from p_theta(X{t-1} | Xt) """
        if eta is None:
            eta = self.eta
        out = self.p_mean_variance(model, Xt, t, clip_denoised)
        pred_eps = self._pred_eps_from_X0(Xt, t, out['pred_X0'])
        alphas_cumprod_t = self._extract(self.alphas_cumprod, t)
        alphas_cumprod_prev_t = self._extract(self.alphas_cumprod_prev, t)
        var_t = ((eta ** 2) *
                 (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t) *
                 (1. - alphas_cumprod_t / alphas_cumprod_prev_t))
        mean_t = (torch.sqrt(alphas_cumprod_prev_t) * out['pred_X0'] +
                  torch.sqrt(1. - alphas_cumprod_prev_t - var_t) * pred_eps)
        nonzero_mask = torch.ne(t, 0).float().view(-1, 1, 1, 1)
        sample = mean_t + nonzero_mask * torch.sqrt(var_t) * torch.randn_like(Xt)
        return {'sample': sample, 'pred_X0': out['pred_X0']}

    @torch.no_grad()
    def ddim_sample_loop(self, model: nn.Module, init_noise: Tensor, clip_denoised: bool = True, eta: float = None):
        img = init_noise
        for t in range(self.total_steps-1, -1, -1):
            t_batch = torch.full((img.shape[0], ), t, device=img.device, dtype=torch.long)
            out = self.ddim_p_sample(model, img, t_batch, clip_denoised, eta)
            img = out['sample']
            yield out

    @torch.no_grad()
    def ddim_sample(self, model: nn.Module, init_noise: Tensor, clip_denoised: bool = True, eta: float = None):
        sample = None
        for out in self.ddim_sample_loop(model, init_noise, clip_denoised, eta):
            sample = out['sample']
        return sample

    @torch.no_grad()
    def ddim_p_sample_inversion(self, model: nn.Module, Xt: Tensor, t: Tensor, clip_denoised: bool = True,
                                eta: float = None):
        """ Sample X{t+1} from Xt, only valid for DDIM (eta=0) """
        if eta is None:
            eta = self.eta
        assert eta == 0., 'DDIM inversion is only valid when eta=0'
        out = self.p_mean_variance(model, Xt, t, clip_denoised)
        pred_eps = self._pred_eps_from_X0(Xt, t, out['pred_X0'])
        alphas_cumprod_next_t = self._extract(self.alphas_cumprod_next, t)
        sample = torch.sqrt(alphas_cumprod_next_t) * out['pred_X0'] + torch.sqrt(1. - alphas_cumprod_next_t) * pred_eps
        return {'sample': sample, 'pred_X0': out['pred_X0']}

    @torch.no_grad()
    def ddim_sample_inversion_loop(self, model: nn.Module, img: Tensor, clip_denoised: bool = True, eta: float = None):
        for t in range(self.total_steps - 1):
            t_batch = torch.full((img.shape[0], ), t, device=img.device, dtype=torch.long)
            out = self.ddim_p_sample_inversion(model, img, t_batch, clip_denoised, eta)
            img = out['sample']
            yield out

    @torch.no_grad()
    def ddim_sample_inversion(self, model: nn.Module, img: Tensor, clip_denoised: bool = True, eta: float = None):
        sample = None
        for out in self.ddim_sample_inversion_loop(model, img, clip_denoised, eta):
            sample = out['sample']
        return sample


class DDIMSkip(DDIM):
    """
    Code adapted from openai:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/respace.py#L63

    A diffusion process which can skip steps in a base diffusion process.
    """
    def __init__(self, timesteps: List[int] or Tensor, **kwargs):
        self.timesteps = timesteps

        # Initialize the original DDIM
        super().__init__(**kwargs)

        # Define new beta sequence
        betas = []
        last_alphas_cumprod = 1.
        for t in timesteps:
            betas.append(1 - self.alphas_cumprod[t] / last_alphas_cumprod)
            last_alphas_cumprod = self.alphas_cumprod[t]
        betas = torch.tensor(betas)

        # Reinitialize with new betas
        kwargs['betas'] = betas
        super().__init__(**kwargs)

    def p_mean_variance(self, model: nn.Module, *args, **kwargs):
        return super().p_mean_variance(self._wrap_model(model), *args, **kwargs)  # noqa

    def _wrap_model(self, model: nn.Module):
        if isinstance(model, _WrappedModel):
            return model
        return _WrappedModel(model, self.timesteps)


class _WrappedModel:
    def __init__(self, model: nn.Module, timesteps: List[int] or Tensor):
        self.model = model
        self.timesteps = timesteps
        if isinstance(timesteps, list):
            self.timesteps = torch.tensor(timesteps)

    def __call__(self, X: Tensor, t: Tensor):
        self.timesteps = self.timesteps.to(t.device)
        ts = self.timesteps[t]
        return self.model(X, ts)
