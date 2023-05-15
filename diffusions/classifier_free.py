from typing import Any

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from diffusions.schedule import get_beta_schedule, get_skip_seq


class ClassifierFree:
    """ Diffusion Models with Classifier-Free Guidance.

    The idea was first proposed by classifier-free guidance paper, but can also be used for conditions besides
    categorial labels, such as text.

    This class can be viewed as an extension of DDPM class and DDIM class, with all of their functions equipped
    with conditions.

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
            guidance_scale: float = 1.,

            device: torch.device = 'cpu',
    ):
        """ Diffusion Models with Classifier-Free Guidance.

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
            guidance_scale: Guidance scale. Note it actually follows the definition in classifier guidance paper,
                            not classifier-free guidance paper. To be specific, let the former be `s` and latter
                            be `w`, then `s=w+1`, and:
                             - `s=0`: unconditional generation.
                             - `s=1`: non-guided conditional generation.
                             - `s>1`: guided conditional generation.
                            This argument doesn't affect training and can be overridden by sampling functions.

        """
        assert objective in ['pred_eps', 'pred_x0']
        assert var_type in ['fixed_small', 'fixed_large']
        self.total_steps = total_steps
        self.objective = objective
        self.var_type = var_type
        self.clip_denoised = clip_denoised
        self.guidance_scale = guidance_scale
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

    def loss_func(self, model: nn.Module, x0: Tensor, t: Tensor, cond: Any = None, eps: Tensor = None, **model_kwargs):
        if eps is None:
            eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        if self.objective == 'pred_eps':
            pred_eps = model(xt, t, cond, **model_kwargs)
            return F.mse_loss(pred_eps, eps)
        else:
            pred_x0 = model(xt, t, cond, **model_kwargs)
            return F.mse_loss(pred_x0, x0)

    def q_sample(self, x0: Tensor, t: Tensor, eps: Tensor = None):
        """ Sample from q(xt | x0).

        Args:
            x0: A Tensor of shape [B, D, ...], the original samples.
            t: A Tensor of shape [B], timesteps for each sample in x0.
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

    def p_sample(self, model_output_cond: Tensor, model_output_uncond: Tensor, xt: Tensor, t: int, t_prev: int,
                 var_type: str = None, clip_denoised: bool = None, guidance_scale: float = None):
        """ Sample from p_theta(x{t-1} | xt).

        Args:
            model_output_cond: Output of the UNet model.
            model_output_uncond: Output of the UNet model.
            xt: A Tensor of shape [B, D, ...], the noisy samples.
            t: Current timestep.
            t_prev: Previous timestep.
            var_type: Override `self.var_type` if not None.
            clip_denoised: Override `self.clip_denoised` if not None.
            guidance_scale: Override `self.guidance_scale` if not None.

        """
        if var_type is None:
            var_type = self.var_type
        if clip_denoised is None:
            clip_denoised = self.clip_denoised
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        alphas_t = alphas_cumprod_t / alphas_cumprod_t_prev
        betas_t = 1. - alphas_t

        # Calculate the predicted x0 and predicted eps
        if self.objective == 'pred_eps':
            pred_eps_cond = model_output_cond
            pred_eps_uncond = model_output_uncond
        elif self.objective == 'pred_x0':
            pred_x0_cond = model_output_cond
            pred_x0_uncond = model_output_uncond
            pred_eps_cond = self.pred_eps_from_x0(xt, t, pred_x0_cond)
            pred_eps_uncond = self.pred_eps_from_x0(xt, t, pred_x0_uncond)
        else:
            raise ValueError
        pred_eps = (1 - guidance_scale) * pred_eps_uncond + guidance_scale * pred_eps_cond
        pred_x0 = self.pred_x0_from_eps(xt, t, pred_eps)
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)

        # Calculate the mean of p_theta(x{t-1} | xt)
        mean_coef1 = (alphas_cumprod_t_prev ** 0.5) * betas_t / (1. - alphas_cumprod_t)
        mean_coef2 = (alphas_t ** 0.5) * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
        mean = mean_coef1 * pred_x0 + mean_coef2 * xt

        # Calculate the variance of p_theta(x{t-1} | xt)
        if t == 0:
            sample = mean
        else:
            if var_type == 'fixed_small':
                var = betas_t * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
                logvar = torch.log(var)
            elif var_type == 'fixed_large':
                var = betas_t
                logvar = torch.log(var)
            else:
                raise ValueError
            sample = mean + torch.exp(0.5 * logvar) * torch.randn_like(xt)

        return {'sample': sample, 'pred_x0': pred_x0}

    def sample_loop(self, model: nn.Module, init_noise: Tensor, cond: Any = None,
                    var_type: str = None, clip_denoised: bool = None, guidance_scale: float = None, **model_kwargs):
        img = init_noise
        skip_seq = self.skip_seq.tolist()
        skip_seq_prev = [-1] + self.skip_seq[:-1].tolist()
        for t, t_prev in zip(reversed(skip_seq), reversed(skip_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device, dtype=torch.long)
            model_output_cond = model(img, t_batch, cond, **model_kwargs)
            model_output_uncond = model(img, t_batch, None, **model_kwargs)
            out = self.p_sample(
                model_output_cond, model_output_uncond, img, t, t_prev, var_type, clip_denoised, guidance_scale,
            )
            img = out['sample']
            yield out

    def sample(self, model: nn.Module, init_noise: Tensor, cond: Any = None,
               var_type: str = None, clip_denoised: bool = None, guidance_scale: float = None, **model_kwargs):
        sample = None
        for out in self.sample_loop(model, init_noise, cond, var_type, clip_denoised, guidance_scale, **model_kwargs):
            sample = out['sample']
        return sample

    def ddim_p_sample(self, model_output_cond: Tensor, model_output_uncond: Tensor, xt: Tensor, t: int, t_prev: int,
                      clip_denoised: bool = None, guidance_scale: float = None, eta: float = 0.0):
        """ Sample from p_theta(x{t-1} | xt) using DDIM, similar to `p_sample` """
        if clip_denoised is None:
            clip_denoised = self.clip_denoised
        if guidance_scale is None:
            guidance_scale = self.guidance_scale

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Calculate the predicted x0 and predicted eps
        if self.objective == 'pred_eps':
            pred_eps_cond = model_output_cond
            pred_eps_uncond = model_output_uncond
        elif self.objective == 'pred_x0':
            pred_x0_cond = model_output_cond
            pred_x0_uncond = model_output_uncond
            pred_eps_cond = self.pred_eps_from_x0(xt, t, pred_x0_cond)
            pred_eps_uncond = self.pred_eps_from_x0(xt, t, pred_x0_uncond)
        else:
            raise ValueError
        pred_eps = (1 - guidance_scale) * pred_eps_uncond + guidance_scale * pred_eps_cond
        pred_x0 = self.pred_x0_from_eps(xt, t, pred_eps)
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)
        pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)

        # Calculate the mean and variance of p_theta(x{t-1} | xt)
        var = ((eta ** 2) *
               (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t) *
               (1. - alphas_cumprod_t / alphas_cumprod_t_prev))
        mean = (torch.sqrt(alphas_cumprod_t_prev) * pred_x0 +
                torch.sqrt(1. - alphas_cumprod_t_prev - var) * pred_eps)
        if t == 0:
            sample = mean
        else:
            sample = mean + torch.sqrt(var) * torch.randn_like(xt)
        return {'sample': sample, 'pred_x0': pred_x0}

    def ddim_sample_loop(self, model: nn.Module, init_noise: Tensor, cond: Any = None,
                         clip_denoised: bool = None, guidance_scale: float = None, eta: float = 0.0, **model_kwargs):
        img = init_noise
        skip_seq = self.skip_seq.tolist()
        skip_seq_prev = [-1] + self.skip_seq[:-1].tolist()
        for t, t_prev in zip(reversed(skip_seq), reversed(skip_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device, dtype=torch.long)
            model_output_cond = model(img, t_batch, cond, **model_kwargs)
            model_output_uncond = model(img, t_batch, None, **model_kwargs)
            out = self.ddim_p_sample(model_output_cond, model_output_uncond, img, t, t_prev,
                                     clip_denoised, guidance_scale, eta)
            img = out['sample']
            yield out

    def ddim_sample(self, model: nn.Module, init_noise: Tensor, cond: Any = None,
                    clip_denoised: bool = None, guidance_scale: float = None, eta: float = 0.0, **model_kwargs):
        sample = None
        for out in self.ddim_sample_loop(model, init_noise, cond, clip_denoised, guidance_scale, eta, **model_kwargs):
            sample = out['sample']
        return sample