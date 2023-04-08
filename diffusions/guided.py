from typing import Dict

import torch
from torch import Tensor
import torch.nn as nn

from diffusions.schedule import get_beta_schedule, get_skip_seq


class Guided:
    """ Diffusion Models with Explicit Guidance.

    The idea was first proposed by classifier guidance paper, and can be used for conditions besides
    categorial labels, such as text.

    The guidance can be applied to the predicted eps, or the predicted x0, or the mean of the reverse distribution.

    This class can be viewed as an extension of DDPM class and DDIM class, with all of their functions equipped
    with guidance.

    """
    def __init__(
            self,
            total_steps: int = 1000,
            beta_schedule: str = 'linear',
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            betas: Tensor = None,
            objective: str = 'pred_eps',

            var_type: str = 'fixed_large',
            clip_denoised: bool = True,
            skip_type: str = None,
            skip_steps: int = 100,
            skip_seq: Tensor = None,

            device: torch.device = 'cpu',
    ):
        """ Diffusion Models with Explicit Guidance.

        The arguments can be divided into two kinds. The first kind of arguments are used in training and should be
        kept unchanged during inference. The second kind of arguments do not affect training (although they may be
        used for visualization during training) and can be overridden by passing arguments to sampling functions
        during inference.

        Args:
            total_steps: Total number of timesteps used to train the model.
            beta_schedule: Type of beta schedule. Options: 'linear', 'quad', 'const', 'cosine'.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            betas: A 1-D Tensor of pre-defined beta schedule. If provided, arguments `beta_*` will be ignored.
            objective: Prediction objective of the model. Options: 'pred_eps', 'pred_x0'.

            var_type: Type of variance of the reverse process.
                      Options: 'fixed_large', 'fixed_small', 'learned_range'.
                      This argument doesn't affect training and can be overridden by sampling functions.
            clip_denoised: Clip the predicted x0 in range [-1, 1].
                           This argument doesn't affect training and can be overridden by sampling functions.
            skip_type: Type of skipping timestep sequence.
                       Options: 'uniform', 'uniform2', 'quad', None.
                       This argument doesn't affect training and can be overridden by `set_skip_seq()`.
            skip_steps: Length of skipping timestep sequence, i.e., number of sampling steps during inference.
                        This argument doesn't affect training and can be overridden by `set_skip_seq()`.
            skip_seq: A 1-D Tensor of pre-defined skip sequence. If provided, arguments `skip_*` will be ignored.
                      This argument doesn't affect training and can be overridden by `set_skip_seq()`.

        """
        assert objective in ['pred_eps', 'pred_x0']
        assert var_type in ['fixed_small', 'fixed_large', 'learned_range']
        self.total_steps = total_steps
        self.objective = objective
        self.var_type = var_type
        self.clip_denoised = clip_denoised
        self.device = device

        # Define betas, alphas and related terms
        if betas is None:
            betas = get_beta_schedule(
                total_steps=total_steps,
                beta_schedule=beta_schedule,
                beta_start=beta_start,
                beta_end=beta_end,
            )
        assert isinstance(betas, Tensor)
        assert betas.shape == (total_steps, )

        alphas = 1. - betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0).to(device, torch.float)

        # Define skip sequence for sampling
        if skip_seq is None:
            skip_seq = get_skip_seq(
                total_steps=total_steps,
                skip_type=skip_type,
                skip_steps=skip_steps,
            )
        assert isinstance(skip_seq, Tensor)
        assert skip_seq.ndim == 1
        self.skip_seq = skip_seq.to(device)

    def set_skip_seq(self, skip_type: str = 'uniform', skip_steps: int = 100):
        self.skip_seq = get_skip_seq(
            total_steps=self.total_steps,
            skip_type=skip_type,
            skip_steps=skip_steps,
        ).to(self.device)

    def q_sample(self, x0: Tensor, t: Tensor, eps: Tensor = None):
        """ Sample from q(xt | x0).

        Args:
            x0: A Tensor of shape [B, D, ...], the original samples.
            t: A Tensor of shape [B], time steps for each sample in x0.
               Note that different samples may have different t.
            eps: A Tensor of shape [B, D, ...], the noise added to the original samples.
        """
        if eps is None:
            eps = torch.randn_like(x0)

        sqrt_alphas_cumprod = self.alphas_cumprod[t] ** 0.5
        while sqrt_alphas_cumprod.ndim < x0.ndim:
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)

        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]) ** 0.5
        while sqrt_one_minus_alphas_cumprod.ndim < x0.ndim:
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * eps

    def pred_x0_from_eps(self, xt: Tensor, t: int, eps: Tensor):
        sqrt_recip_alphas_cumprod_t = (1. / self.alphas_cumprod[t]) ** 0.5
        sqrt_recipm1_alphas_cumprod_t = (1. / self.alphas_cumprod[t] - 1.) ** 0.5
        return sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * eps

    def pred_eps_from_x0(self, xt: Tensor, t: int, x0: Tensor):
        sqrt_recip_alphas_cumprod_t = (1. / self.alphas_cumprod[t]) ** 0.5
        sqrt_recipm1_alphas_cumprod_t = (1. / self.alphas_cumprod[t] - 1.) ** 0.5
        return (sqrt_recip_alphas_cumprod_t * xt - x0) / sqrt_recipm1_alphas_cumprod_t

    def pred_mu_from_x0(self, xt: Tensor, t: int, t_prev: int, x0: Tensor):
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        alphas_t = alphas_cumprod_t / alphas_cumprod_t_prev
        betas_t = 1. - alphas_t
        mean_coef1 = (alphas_cumprod_t_prev ** 0.5) * betas_t / (1. - alphas_cumprod_t)
        mean_coef2 = (alphas_t ** 0.5) * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
        return mean_coef1 * x0 + mean_coef2 * xt

    def pred_x0_from_mu(self, xt: Tensor, t: int, t_prev: int, mu: Tensor):
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        alphas_t = alphas_cumprod_t / alphas_cumprod_t_prev
        betas_t = 1. - alphas_t
        mean_coef1 = (alphas_cumprod_t_prev ** 0.5) * betas_t / (1. - alphas_cumprod_t)
        mean_coef2 = (alphas_t ** 0.5) * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
        return (mu - mean_coef2 * xt) / mean_coef1

    def p_mean_variance(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int,
                        var_type: str = None, clip_denoised: bool = None):
        """ Compute mean and variance of p(x{t-1} | xt).

        Args:
            model_output: Output of the UNet model.
            xt: A Tensor of shape [B, D, ...], the noisy samples.
            t: Current timestep.
            t_prev: Previous timestep.
            var_type: Override `self.var_type` if not None.
            clip_denoised: Override `self.clip_denoised` if not None.

        """
        if var_type is None:
            var_type = self.var_type
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        alphas_t = alphas_cumprod_t / alphas_cumprod_t_prev
        betas_t = 1. - alphas_t

        # Process model's output
        learned_var = None
        if model_output.shape[1] > xt.shape[1]:
            model_output, learned_var = torch.split(model_output, xt.shape[1], dim=1)

        # Calculate the predicted x0 and predicted eps
        if self.objective == 'pred_eps':
            pred_eps = model_output
            pred_x0 = self.pred_x0_from_eps(xt, t, pred_eps)
        elif self.objective == 'pred_x0':
            pred_x0 = model_output
        else:
            raise ValueError
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)
        pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)

        # Calculate the mean of p_theta(x{t-1} | xt)
        mean = self.pred_mu_from_x0(xt, t, t_prev, pred_x0)

        # Calculate the variance of p_theta(x{t-1} | xt)
        if var_type == 'fixed_small':
            var = betas_t * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
            logvar = torch.log(var)
        elif var_type == 'fixed_large':
            var = betas_t
            logvar = torch.log(var)
        elif var_type == 'learned_range':
            min_var = betas_t * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
            min_logvar = torch.log(torch.clamp_min(min_var, 1e-20))
            max_logvar = torch.log(betas_t)
            frac = (learned_var + 1) / 2  # [-1, 1] --> [0, 1]
            logvar = frac * max_logvar + (1 - frac) * min_logvar
        else:
            raise ValueError

        return {'mean': mean, 'logvar': logvar, 'pred_x0': pred_x0, 'pred_eps': pred_eps}

    def cond_fn_eps(self, t: int, xt: Tensor, pred_x0: Tensor, pred_eps: Tensor,
                    mean: Tensor, logvar: Tensor) -> Tensor:
        """ Apply guidance on the predicted eps. """
        pass

    def cond_fn_x0(self, t: int, xt: Tensor, pred_x0: Tensor, pred_eps: Tensor,
                   mean: Tensor, logvar: Tensor) -> Tensor:
        """ Function that applies guidance on the predicted x0. """
        pass

    def cond_fn_mean(self, t: int, xt: Tensor, pred_x0: Tensor, pred_eps: Tensor,
                     mean: Tensor, logvar: Tensor) -> Tensor:
        """ Function that applies guidance on the mean of p(x{t-1} | xt). """
        pass

    def p_mean_variance_guided(self, p_mean_variance: Dict, xt: Tensor, t: int, t_prev: int):
        """ Apply guidance to p(x{t-1} | xt).

        Args:
            p_mean_variance: Original mean, variance and pred_x0 of p({t-1} | xt) without guidance.
            xt: A Tensor of shape [B, D, ...], the noisy samples.
            t: Current timestep.
            t_prev: Previous timestep.

        """
        cond_fn_kwargs = {**p_mean_variance, 'xt': xt, 't': t}

        # Apply guidance on the predicted eps
        guidance = self.cond_fn_eps(**cond_fn_kwargs)
        if guidance is not None:
            new_eps = p_mean_variance['pred_eps'] + guidance
            new_x0 = self.pred_x0_from_eps(xt, t, new_eps)
            new_mean = self.pred_mu_from_x0(xt, t, t_prev, new_x0)
            p_mean_variance.update({'mean': new_mean, 'pred_x0': new_x0, 'pred_eps': new_eps})

        # Apply guidance on the predicted x0
        guidance = self.cond_fn_x0(**cond_fn_kwargs)
        if guidance is not None:
            new_x0 = p_mean_variance['pred_x0'] + guidance
            new_eps = self.pred_eps_from_x0(xt, t, new_x0)
            new_mean = self.pred_mu_from_x0(xt, t, t_prev, new_x0)
            p_mean_variance.update({'mean': new_mean, 'pred_x0': new_x0, 'pred_eps': new_eps})

        # Apply guidance on mean of p(x{t-1} | xt)
        guidance = self.cond_fn_mean(**cond_fn_kwargs)
        if guidance is not None:
            new_mean = p_mean_variance['mean'] + guidance
            new_x0 = self.pred_x0_from_mu(xt, t, t_prev, new_mean)
            new_eps = self.pred_eps_from_x0(xt, t, new_x0)
            p_mean_variance.update({'mean': new_mean, 'pred_x0': new_x0, 'pred_eps': new_eps})

        return p_mean_variance

    def p_sample(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int,
                 var_type: str = None, clip_denoised: bool = None):
        """ Sample from p_theta(x{t-1} | xt). """
        out = self.p_mean_variance(model_output, xt, t, t_prev, var_type, clip_denoised)
        out = self.p_mean_variance_guided(out, xt, t, t_prev)
        if t == 0:
            sample = out['mean']
        else:
            sample = out['mean'] + torch.exp(0.5 * out['logvar']) * torch.randn_like(xt)
        return {'sample': sample, 'pred_x0': out['pred_x0']}

    def sample_loop(self, model: nn.Module, init_noise: Tensor,
                    var_type: str = None, clip_denoised: bool = None, **model_kwargs):
        img = init_noise
        skip_seq = self.skip_seq.tolist()
        skip_seq_prev = [-1] + self.skip_seq[:-1].tolist()
        for t, t_prev in zip(reversed(skip_seq), reversed(skip_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device, dtype=torch.long)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.p_sample(model_output, img, t, t_prev, var_type, clip_denoised)
            img = out['sample']
            yield out

    def sample(self, model: nn.Module, init_noise: Tensor,
               var_type: str = None, clip_denoised: bool = None, **model_kwargs):
        sample = None
        for out in self.sample_loop(model, init_noise, var_type, clip_denoised, **model_kwargs):
            sample = out['sample']
        return sample

    def ddim_p_sample(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int,
                      clip_denoised: bool = None, eta: float = 0.0):
        """ Sample from p(x{t-1} | xt) using DDIM, similar to p_sample. """
        # Prepare alphas, betas and other parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Calculate the predicted x0 and predicted eps
        out = self.p_mean_variance(model_output, xt, t, t_prev, None, clip_denoised)
        out = self.p_mean_variance_guided(out, xt, t, t_prev)

        # Calculate the mean and variance of p_theta(x{t-1} | xt)
        var = ((eta ** 2) *
               (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t) *
               (1. - alphas_cumprod_t / alphas_cumprod_t_prev))
        mean = (torch.sqrt(alphas_cumprod_t_prev) * out['pred_x0'] +
                torch.sqrt(1. - alphas_cumprod_t_prev - var) * out['pred_eps'])
        if t == 0:
            sample = mean
        else:
            sample = mean + torch.sqrt(var) * torch.randn_like(xt)
        return {'sample': sample, 'pred_x0': out['pred_x0']}

    def ddim_sample_loop(self, model: nn.Module, init_noise: Tensor,
                         clip_denoised: bool = None, eta: float = 0.0, **model_kwargs):
        img = init_noise
        skip_seq = self.skip_seq.tolist()
        skip_seq_prev = [-1] + self.skip_seq[:-1].tolist()
        for t, t_prev in zip(reversed(skip_seq), reversed(skip_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device, dtype=torch.long)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.ddim_p_sample(model_output, img, t, t_prev, clip_denoised, eta)
            img = out['sample']
            yield out

    def ddim_sample(self, model: nn.Module, init_noise: Tensor,
                    clip_denoised: bool = None, eta: float = 0.0, **model_kwargs):
        sample = None
        for out in self.ddim_sample_loop(model, init_noise, clip_denoised, eta, **model_kwargs):
            sample = out['sample']
        return sample
