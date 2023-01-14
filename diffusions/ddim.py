import math
import tqdm

import torch
import torch.nn as nn
from torch import Tensor


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


def get_skip_seq(skip_type: str = 'uniform', skip_steps: int = 1000, total_steps: int = 1000):
    if skip_type == 'uniform':
        skip = total_steps // skip_steps
        seq = torch.arange(0, total_steps, skip)
    elif skip_type == 'quad':
        seq = torch.linspace(0, math.sqrt(total_steps * 0.8), skip_steps) ** 2
        seq = torch.floor(seq).to(dtype=torch.int64)
    else:
        raise ValueError(f'skip_type {skip_type} is not valid')
    return seq


class DDIM:
    def __init__(self,
                 total_steps: int = 1000,
                 beta_schedule: str = 'linear',
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 objective: str = 'pred_eps',
                 eta: float = 0.,
                 skip_type: str = 'uniform',
                 skip_steps: int = 1000):
        assert objective in ['pred_eps', 'pred_x0']
        self.objective = objective
        self.eta = eta
        self.skip_seq = get_skip_seq(skip_type, skip_steps, total_steps)

        betas = get_beta_schedule(beta_schedule, total_steps, beta_start, beta_end)
        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1.)

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

    def _pred_X0_from_eps(self, Xt: Tensor, t: Tensor, eps: Tensor):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t) * Xt -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t) * eps)

    def _pred_eps_from_X0(self, Xt: Tensor, t: Tensor, X0: Tensor):
        return ((self._extract(self.sqrt_recip_alphas_cumprod, t) * Xt - X0) /
                self._extract(self.sqrt_recipm1_alphas_cumprod, t))

    @torch.no_grad()
    def p_sample(self, model: nn.Module, Xt: Tensor, t: int, t_prev: int, clip_denoised: bool = True):
        """ Sample from p_theta(X{t-1} | Xt)
        Args:
            model (nn.Module): UNet model
            Xt (Tensor): [bs, C, H, W]
            t (int): time step for all samples in Xt
            t_prev (int): previous time step
            clip_denoised (bool): whether to clip predicted X0 to range [-1, 1]
        """
        t_batch = torch.full((Xt.shape[0], ), t, device=Xt.device, dtype=torch.long)
        t_prev_batch = torch.full((Xt.shape[0], ), t_prev, device=Xt.device, dtype=torch.long)
        if self.objective == 'pred_eps':
            pred_eps = model(Xt, t_batch)
            pred_X0 = self._pred_X0_from_eps(Xt, t_batch, pred_eps)
        else:
            pred_X0 = model(Xt, t_batch)
            pred_eps = self._pred_eps_from_X0(Xt, t_batch, pred_X0)
        if clip_denoised:
            pred_X0.clamp_(-1., 1.)

        alphas_cumprod_t = self._extract(self.alphas_cumprod, t_batch)
        if t_prev != -1:
            alphas_cumprod_prev_t = self._extract(self.alphas_cumprod, t_prev_batch)
        else:
            alphas_cumprod_prev_t = torch.ones_like(alphas_cumprod_t)
        var_t = ((self.eta ** 2) * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t) *
                 (1. - alphas_cumprod_t / alphas_cumprod_prev_t))
        mean_t = (torch.sqrt(alphas_cumprod_prev_t) * pred_X0 +
                  torch.sqrt(1. - alphas_cumprod_prev_t - var_t) * pred_eps)
        if t == 0:
            return mean_t
        return mean_t + torch.sqrt(var_t) * torch.randn_like(Xt)

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               init_noise: Tensor,
               clip_denoised: bool = True,
               with_tqdm: bool = False,
               **kwargs):
        img = init_noise
        skip_seq_prev = torch.cat([torch.tensor([-1.]), self.skip_seq[:-1]])
        kwargs['disable'] = kwargs.get('disable', False) or (not with_tqdm)
        for i in tqdm.tqdm(range(len(self.skip_seq)-1, -1, -1), **kwargs):
            img = self.p_sample(model, img, self.skip_seq[i], skip_seq_prev[i], clip_denoised)
        return img.cpu()


def _test():
    seq = get_skip_seq(
        skip_type='uniform',
        skip_steps=10,
        total_steps=1000,
    )
    print(seq)


if __name__ == '__main__':
    _test()
