import math
import tqdm

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
    elif beta_schedule == "cosine":
        betas = []
        for i in range(total_steps):
            t1 = i / total_steps
            t2 = (i + 1) / total_steps

            def alpha_bar(t):
                return math.cos((t + 0.008) / 1.008 * math.pi / 2) ** 2
            betas.append(min(1 - alpha_bar(t2) / alpha_bar(t1), 0.999))
        return torch.tensor(betas)
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


class ClassifierFree:
    def __init__(self,
                 total_steps: int = 1000,
                 beta_schedule: str = 'linear',
                 beta_start: float = 0.0001,
                 beta_end: float = 0.02,
                 objective: str = 'pred_eps',
                 var_type: str = 'fixed_large'):
        self.total_steps = total_steps
        assert objective in ['pred_eps', 'pred_x0']
        assert var_type in ['fixed_small', 'fixed_large']
        self.objective = objective

        # Define betas, alphas and related terms
        betas = get_beta_schedule(beta_schedule, total_steps, beta_start, beta_end)
        alphas = 1. - betas
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

    def loss_func(self, model: nn.Module, X0: Tensor, y: Tensor, t: Tensor, eps: Tensor = None):
        if eps is None:
            eps = torch.randn_like(X0)
        Xt = self.q_sample(X0, t, eps)
        if self.objective == 'pred_eps':
            pred_eps = model(Xt, y, t)
            return F.mse_loss(pred_eps, eps)
        else:
            pred_x0 = model(Xt, y, t)
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

    def _pred_X0_from_eps(self, Xt: Tensor, t: Tensor, eps: Tensor):
        return (self._extract(self.sqrt_recip_alphas_cumprod, t) * Xt -
                self._extract(self.sqrt_recipm1_alphas_cumprod, t) * eps)

    def _pred_eps_from_X0(self, Xt: Tensor, t: Tensor, X0: Tensor):
        return ((self._extract(self.sqrt_recip_alphas_cumprod, t) * Xt - X0) /
                self._extract(self.sqrt_recipm1_alphas_cumprod, t))

    def p_mean_variance(self, model: nn.Module, Xt: Tensor, y: Tensor, t: Tensor,
                        clip_denoised: bool = True, guidance_scale: float = 1.):
        """ Compute mean and variance of p(X{t-1} | Xt)
        Args:
            model (nn.Module): UNet model
            Xt (Tensor): [bs, C, H, W]
            y (Tensor): [bs], class labels
            t (Tensor): [bs], time steps
            clip_denoised (bool): whether to clip predicted X0 to range [-1, 1]
            guidance_scale (float): guidance scale, note that it's a bit different from the original paper
        """
        if self.objective == 'pred_eps':
            pred_eps_cond = model(Xt, y, t)
            pred_eps_uncond = model(Xt, None, t)
        else:
            pred_X0_cond = model(Xt, y, t)
            pred_eps_cond = self._pred_eps_from_X0(Xt, t, pred_X0_cond)
            pred_X0_uncond = model(Xt, None, t)
            pred_eps_uncond = self._pred_eps_from_X0(Xt, t, pred_X0_uncond)
        pred_eps = (1 - guidance_scale) * pred_eps_uncond + guidance_scale * pred_eps_cond
        pred_X0 = self._pred_X0_from_eps(Xt, t, pred_eps)
        if clip_denoised:
            pred_X0.clamp_(-1., 1.)
        mean_t, var_t, log_var_t = self.q_posterior_mean_variance(pred_X0, Xt, t)
        var_t = self._extract(self.model_variance, t)
        log_var_t = self._extract(self.model_log_variance, t)
        return {'mean': mean_t, 'var': var_t, 'log_var': log_var_t, 'pred_X0': pred_X0}

    @torch.no_grad()
    def p_sample(self, model: nn.Module, Xt: Tensor, y: int or Tensor, t: int,
                 clip_denoised: bool = True, guidance_scale: float = 1.):
        """ Sample from p_theta(X{t-1} | Xt)
        Args:
            model (nn.Module): UNet model
            Xt (Tensor): [bs, C, H, W]
            y (int or Tensor): class label(s)
            t (int): time step for all samples in Xt
            clip_denoised (bool): whether to clip predicted X0 to range [-1, 1]
            guidance_scale (float): guidance scale, note that it's a bit different from the original paper
        """
        t_batch = torch.full((Xt.shape[0], ), t, device=Xt.device, dtype=torch.long)
        if isinstance(y, Tensor):
            assert y.shape == (Xt.shape[0], )
            y_batch = y
        else:
            y_batch = torch.full((Xt.shape[0], ), y, device=Xt.device, dtype=torch.long)
        out = self.p_mean_variance(model, Xt, y_batch, t_batch, clip_denoised, guidance_scale)
        sample = out['mean'] if t == 0 else out['mean'] + torch.exp(0.5 * out['log_var']) * torch.randn_like(Xt)
        return {'sample': sample, 'pred_X0': out['pred_X0']}

    @torch.no_grad()
    def sample_loop(self,
                    model: nn.Module,
                    class_label: int or Tensor,
                    init_noise: Tensor,
                    clip_denoised: bool = True,
                    guidance_scale: float = 1.,
                    with_tqdm: bool = False,
                    **kwargs):
        img = init_noise
        kwargs['disable'] = kwargs.get('disable', False) or (not with_tqdm)
        for t in tqdm.tqdm(range(self.total_steps-1, -1, -1), **kwargs):
            out = self.p_sample(model, img, class_label, t, clip_denoised, guidance_scale)
            img = out['sample']
            yield out

    @torch.no_grad()
    def sample(self,
               model: nn.Module,
               class_label: int or Tensor,
               init_noise: Tensor,
               clip_denoised: bool = True,
               guidance_scale: float = 1.,
               with_tqdm: bool = False,
               **kwargs):
        sample = None
        for out in self.sample_loop(
            model=model,
            class_label=class_label,
            init_noise=init_noise,
            clip_denoised=clip_denoised,
            guidance_scale=guidance_scale,
            with_tqdm=with_tqdm,
            **kwargs,
        ):
            sample = out['sample']
        return sample

    @torch.no_grad()
    def ddim_p_sample(self,
                      model: nn.Module,
                      Xt: Tensor,
                      y: int or Tensor,
                      t: int,
                      t_prev: int,
                      clip_denoised: bool = True,
                      guidance_scale: float = 1.,
                      eta=0.0):
        """ Sample from p_theta(X{t-1} | Xt) using DDIM
        Args:
            model (nn.Module): UNet model
            Xt (Tensor): [bs, C, H, W]
            y (int or Tensor): class label(s)
            t (int): time step for all samples in Xt
            t_prev (int): previous time step
            clip_denoised (bool): whether to clip predicted X0 to range [-1, 1]
            guidance_scale (float): guidance scale, note that it's a bit different from the original paper
            eta (float): ddim variance parameter
        """
        t_batch = torch.full((Xt.shape[0], ), t, device=Xt.device, dtype=torch.long)
        t_prev_batch = torch.full((Xt.shape[0], ), t_prev, device=Xt.device, dtype=torch.long)
        if isinstance(y, Tensor):
            assert y.shape == (Xt.shape[0], )
            y_batch = y
        else:
            y_batch = torch.full((Xt.shape[0], ), y, device=Xt.device, dtype=torch.long)
        if self.objective == 'pred_eps':
            pred_eps_cond = model(Xt, y_batch, t_batch)
            pred_eps_uncond = model(Xt, None, t_batch)
        else:
            pred_X0_cond = model(Xt, y_batch, t_batch)
            pred_eps_cond = self._pred_eps_from_X0(Xt, t_batch, pred_X0_cond)
            pred_X0_uncond = model(Xt, None, t_batch)
            pred_eps_uncond = self._pred_eps_from_X0(Xt, t_batch, pred_X0_uncond)
        pred_eps = (1 - guidance_scale) * pred_eps_uncond + guidance_scale * pred_eps_cond
        pred_X0 = self._pred_X0_from_eps(Xt, t_batch, pred_eps)
        if clip_denoised:
            pred_X0.clamp_(-1., 1.)

        alphas_cumprod_t = self._extract(self.alphas_cumprod, t_batch)
        if t_prev != -1:
            alphas_cumprod_prev_t = self._extract(self.alphas_cumprod, t_prev_batch)
        else:
            alphas_cumprod_prev_t = torch.ones_like(alphas_cumprod_t)
        var_t = ((eta ** 2) * (1. - alphas_cumprod_prev_t) / (1. - alphas_cumprod_t) *
                 (1. - alphas_cumprod_t / alphas_cumprod_prev_t))
        mean_t = (torch.sqrt(alphas_cumprod_prev_t) * pred_X0 +
                  torch.sqrt(1. - alphas_cumprod_prev_t - var_t) * pred_eps)
        sample = mean_t if t == 0 else mean_t + torch.sqrt(var_t) * torch.randn_like(Xt)
        return {'sample': sample, 'pred_X0': pred_X0}

    @torch.no_grad()
    def ddim_sample_loop(self,
                         model: nn.Module,
                         class_label: int or Tensor,
                         init_noise: Tensor,
                         clip_denoised: bool = True,
                         guidance_scale: float = 1.,
                         eta: float = 0.,
                         skip_type: str = 'uniform',
                         skip_steps: int = 1000,
                         with_tqdm: bool = False,
                         **kwargs):
        img = init_noise
        kwargs['disable'] = kwargs.get('disable', False) or (not with_tqdm)
        skip_seq = get_skip_seq(skip_type, skip_steps, self.total_steps)
        skip_seq_prev = torch.cat([torch.tensor([-1.]), skip_seq[:-1]])
        for i in tqdm.tqdm(range(len(skip_seq)-1, -1, -1), **kwargs):
            out = self.ddim_p_sample(
                model=model,
                Xt=img,
                y=class_label,
                t=skip_seq[i],
                t_prev=skip_seq_prev[i],
                clip_denoised=clip_denoised,
                guidance_scale=guidance_scale,
                eta=eta,
            )
            img = out['sample']
            yield out

    @torch.no_grad()
    def ddim_sample(self,
                    model: nn.Module,
                    class_label: int or Tensor,
                    init_noise: Tensor,
                    clip_denoised: bool = True,
                    guidance_scale: float = 1.,
                    eta: float = 0.,
                    skip_type: str = 'uniform',
                    skip_steps: int = 1000,
                    with_tqdm: bool = False,
                    **kwargs):
        sample = None
        for out in self.ddim_sample_loop(
            model=model,
            class_label=class_label,
            init_noise=init_noise,
            clip_denoised=clip_denoised,
            guidance_scale=guidance_scale,
            eta=eta,
            skip_type=skip_type,
            skip_steps=skip_steps,
            with_tqdm=with_tqdm,
            **kwargs,
        ):
            sample = out['sample']
        return sample


def _test():
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))

    betas_linear = get_beta_schedule('linear', total_steps=1000, beta_start=0.0001, beta_end=0.02)
    alphas_bar_linear = torch.cumprod(1. - betas_linear, dim=0)
    betas_quad = get_beta_schedule('quad', total_steps=1000, beta_start=0.0001, beta_end=0.02)
    alphas_bar_quad = torch.cumprod(1. - betas_quad, dim=0)
    betas_cosine = get_beta_schedule('cosine', total_steps=1000, beta_start=0.0001, beta_end=0.02)
    alphas_bar_cosine = torch.cumprod(1. - betas_cosine, dim=0)
    ax[0].plot(torch.arange(1000), betas_linear, label='linear')
    ax[0].plot(torch.arange(1000), betas_quad, label='quad')
    ax[0].plot(torch.arange(1000), betas_cosine, label='cosine')
    ax[0].set_title('betas')
    ax[0].legend()
    ax[1].plot(torch.arange(1000), alphas_bar_linear, label='linear')
    ax[1].plot(torch.arange(1000), alphas_bar_quad, label='quad')
    ax[1].plot(torch.arange(1000), alphas_bar_cosine, label='cosine')
    ax[1].set_title('alphas_bar')
    ax[1].legend()
    plt.show()


if __name__ == '__main__':
    _test()
