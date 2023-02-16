from typing import List

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusions.schedule import get_beta_schedule


class DDPM:
    def __init__(self, betas: Tensor = None, objective: str = 'pred_eps', var_type: str = 'fixed_large'):
        assert objective in ['pred_eps', 'pred_x0']
        assert var_type in ['fixed_small', 'fixed_large']
        self.objective = objective

        # Define betas, alphas and related terms
        if betas is None:
            betas = get_beta_schedule()
        alphas = 1. - betas
        self.total_steps = len(betas)
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.alphas_cumprod_prev = torch.cat((torch.ones(1, dtype=torch.float64), self.alphas_cumprod[:-1]))

        # q(Xt | X0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # q(X{t-1} | Xt, X0)
        self.posterior_variance = betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.posterior_log_variance = torch.log(torch.cat([self.posterior_variance[[1]], self.posterior_variance[1:]]))
        self.posterior_mean_coef1 = torch.sqrt(self.alphas_cumprod_prev) * betas / (1. - self.alphas_cumprod)
        self.posterior_mean_coef2 = torch.sqrt(alphas) * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

        # p(X{t-1} | Xt)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)
        if var_type == 'fixed_small':
            self.model_variance = self.posterior_variance
            self.model_log_variance = self.posterior_log_variance
        else:
            self.model_variance = betas
            self.model_log_variance = torch.log(betas)

    @staticmethod
    def _extract(a: Tensor, t: Tensor):
        """ Extract elements in `a` according to a batch of indices `t`, and reshape it to [bs, 1, 1, 1]
        Args:
            a (Tensor): [T]
            t (Tensor): [bs]
        Returns:
            Tensor of shape [bs, 1, 1, 1]
        """
        a = a.to(device=t.device, dtype=torch.float32)
        return a[t][:, None, None, None]

    def loss_func(self, model: nn.Module, X0: Tensor, t: Tensor, eps: Tensor = None):
        if eps is None:
            eps = torch.randn_like(X0)
        Xt = self.q_sample(X0, t, eps)
        if self.objective == 'pred_eps':
            pred_eps = model(Xt, t)
            return F.mse_loss(pred_eps, eps)
        else:
            pred_x0 = model(Xt, t)
            return F.mse_loss(pred_x0, X0)

    def q_sample(self, X0: Tensor, t: Tensor, eps: Tensor = None):
        """ Sample from q(Xt | X0)
        Args:
            X0 (Tensor): [bs, C, H, W]
            t (Tensor): [bs], time steps for each sample in X0, note that different sample may have different t
            eps (Tensor | None): [bs, C, H, W]
        """
        if eps is None:
            eps = torch.randn_like(X0)
        mean = self._extract(self.sqrt_alphas_cumprod, t) * X0
        std = self._extract(self.sqrt_one_minus_alphas_cumprod, t)
        return mean + std * eps

    def q_posterior_mean_variance(self, X0: Tensor, Xt: Tensor, t: Tensor):
        """ Compute mean and variance of q(X{t-1} | Xt, X0)
        Args:
            X0 (Tensor): [bs, C, H, W]
            Xt (Tensor): [bs, C, H, W]
            t (Tensor): [bs], time steps for each sample in X0 and Xt
        """
        mean_t = (self._extract(self.posterior_mean_coef1, t) * X0 +
                  self._extract(self.posterior_mean_coef2, t) * Xt)
        var_t = self._extract(self.posterior_variance, t)
        log_var_t = self._extract(self.posterior_log_variance, t)
        return mean_t, var_t, log_var_t

    def p_mean_variance(self, model: nn.Module, Xt: Tensor, t: Tensor, clip_denoised: bool = True):
        """ Compute mean and variance of p(X{t-1} | Xt)
        Args:
            model (nn.Module): UNet model
            Xt (Tensor): [bs, C, H, W]
            t (Tensor): [bs], time steps
            clip_denoised (bool): whether to clip predicted X0 to range [-1, 1]
        """
        if self.objective == 'pred_eps':
            pred_eps = model(Xt, t)
            pred_X0 = self._pred_X0_from_eps(Xt, t, pred_eps)
        else:
            pred_X0 = model(Xt, t)
        if clip_denoised:
            pred_X0.clamp_(-1., 1.)
        mean_t, var_t, log_var_t = self.q_posterior_mean_variance(pred_X0, Xt, t)
        var_t = self._extract(self.model_variance, t)
        log_var_t = self._extract(self.model_log_variance, t)
        return {'mean': mean_t, 'var': var_t, 'log_var': log_var_t, 'pred_X0': pred_X0}

    def _pred_X0_from_eps(self, Xt: Tensor, t: Tensor, eps: Tensor):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t) * Xt -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t) * eps)

    @torch.no_grad()
    def p_sample(self, model: nn.Module, Xt: Tensor, t: int, clip_denoised: bool = True):
        """ Sample from p_theta(X{t-1} | Xt)
        Args:
            model (nn.Module): UNet model
            Xt (Tensor): [bs, C, H, W]
            t (int): time step for all samples in Xt
            clip_denoised (bool): whether to clip predicted X0 to range [-1, 1]
        """
        t_batch = torch.full((Xt.shape[0], ), t, device=Xt.device, dtype=torch.long)
        out = self.p_mean_variance(model, Xt, t_batch, clip_denoised)
        sample = out['mean'] if t == 0 else out['mean'] + torch.exp(0.5 * out['log_var']) * torch.randn_like(Xt)
        return {'sample': sample, 'pred_X0': out['pred_X0']}

    @torch.no_grad()
    def sample_loop(self, model: nn.Module, init_noise: Tensor, clip_denoised: bool = True):
        img = init_noise
        for t in range(self.total_steps-1, -1, -1):
            out = self.p_sample(model, img, t, clip_denoised)
            img = out['sample']
            yield out

    @torch.no_grad()
    def sample(self, model: nn.Module, init_noise: Tensor, clip_denoised: bool = True):
        sample = None
        for out in self.sample_loop(model, init_noise, clip_denoised):
            sample = out['sample']
        return sample


class DDPMSkip(DDPM):
    """
    Code adapted from openai:
    https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/respace.py#L63

    A diffusion process which can skip steps in a base diffusion process.
    """
    def __init__(self, timesteps: List[int] or Tensor, **kwargs):
        self.timesteps = timesteps

        # Initialize the original DDPM
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

    def loss_func(self, model: nn.Module, *args, **kwargs):
        return super().loss_func(self._wrap_model(model), *args, **kwargs)  # noqa

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
