import tqdm
from typing import Tuple

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F


def get_beta_schedule(beta_schedule: str = 'linear',
                      total_steps: int = 1000,
                      beta_start: float = 0.0001,
                      beta_end: float = 0.02):
    if beta_schedule == 'linear':
        return torch.linspace(beta_start, beta_end, total_steps, dtype=torch.float64)
    elif beta_schedule == 'quad':
        return torch.linspace(beta_start ** 0.5, beta_end ** 0.5, total_steps, dtype=torch.float64) ** 2
    elif beta_schedule == 'const':
        return torch.full((total_steps, ), fill_value=beta_end, dtype=torch.float64)
    else:
        raise ValueError(f'Beta schedule {beta_schedule} is not supported.')


class DDPM:
    def __init__(self,
                 total_steps: int = 1000,
                 beta_schedule: str = 'linear',
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 objective: str = 'pred_eps'):
        self.total_steps = total_steps
        assert objective in ['pred_eps', 'pred_x0']
        self.objective = objective
        # Define betas, alphas and related terms
        betas = get_beta_schedule(beta_schedule, total_steps, beta_start, beta_end)
        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = torch.cat((torch.ones(1, dtype=torch.float64), self.alphas_cumprod[:-1]))
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_one_div_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.variance = betas * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.log_variance = torch.log(self.variance.clamp(min=1e-20))
        self.mean_coef1 = torch.sqrt(alphas) * (1. - alphas_cumprod_prev) / (1. - self.alphas_cumprod)
        self.mean_coef2 = torch.sqrt(alphas_cumprod_prev) * betas / (1. - self.alphas_cumprod)

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

    def p_mean_variance(self, model: nn.Module, Xt: Tensor, t: Tensor, clip_denoised: bool = True):
        """
        Args:
            model (nn.Module): UNet model
            Xt (Tensor): [bs, C, H, W]
            t (Tensor): [bs], time steps
            clip_denoised (bool): whether to clip predicted X0 to range [-1, 1]
        """
        if self.objective == 'pred_eps':
            pred_eps = model(Xt, t)
            pred_X0 = (self._extract(self.sqrt_one_div_alphas_cumprod, t) *
                       (Xt - self._extract(self.sqrt_one_minus_alphas_cumprod, t) * pred_eps))
        else:
            pred_X0 = model(Xt, t)
        if clip_denoised:
            pred_X0.clamp_(-1., 1.)
        pred_mu = self._extract(self.mean_coef1, t) * Xt + self._extract(self.mean_coef2, t) * pred_X0
        var_t = self._extract(self.variance, t)
        log_var_t = self._extract(self.log_variance, t)
        return pred_mu, var_t, log_var_t

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
        mu_t, _, log_var_t = self.p_mean_variance(model, Xt, t_batch, clip_denoised)
        if t == 0:
            return mu_t
        return mu_t + torch.exp(0.5 * log_var_t) * torch.randn_like(Xt)

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               shape: Tuple[int, int, int, int],
               clip_denoised: bool = True,
               same_XT: bool = False,
               return_all: bool = False,
               with_tqdm: bool = False):
        device = next(model.parameters()).device
        if not same_XT:
            img = torch.randn(shape, device=device)
        else:
            img = torch.randn((1, *shape[1:]), device=device).repeat(shape[0], 1, 1, 1)
        imgs = [img.cpu()] if return_all else []

        for t in tqdm.tqdm(range(self.total_steps-1, -1, -1), desc='Sampling', ncols=120, disable=not with_tqdm):
            img = self.p_sample(model, img, t, clip_denoised)
            if return_all:
                imgs.append(img.cpu())
        return imgs if return_all else img.cpu()
