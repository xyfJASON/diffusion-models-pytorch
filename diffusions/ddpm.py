import tqdm
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from diffusions.schedule import get_beta_schedule, get_respaced_seq


class DDPM:
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
            respace_type: str = None,
            respace_steps: int = 100,
            respaced_seq: Tensor = None,

            device: torch.device = 'cpu',
    ):
        """Denoising Diffusion Probabilistic Models.

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
            objective: Prediction objective of the model. Options: 'pred_eps', 'pred_x0', 'pred_v'.

            var_type: Type of variance of the reverse process. Options: 'fixed_large', 'fixed_small', 'learned_range'.
             This argument doesn't affect training and can be overridden in sampling functions.
            clip_denoised: Clip the predicted x0 in range [-1, 1]. This argument doesn't affect training and can be
             overridden in sampling functions.
            respace_type: Type of respaced timestep sequence. Options: 'uniform', 'uniform-leading', 'uniform-linspace',
             'uniform-trailing', 'quad', 'none', None. This argument doesn't affect training and can be overridden in
             `set_respaced_seq()`.
            respace_steps: Length of respaced timestep sequence, i.e., number of sampling steps during inference. This
             argument doesn't affect training and can be overridden in `set_respaced_seq()`.
            respaced_seq: A 1-D Tensor of pre-defined respaced sequence. If provided, arguments `respace_*` will be
             ignored. This argument doesn't affect training and can be overridden in `set_respaced_seq()`.

        References:
            [1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models."
            Advances in neural information processing systems 33 (2020): 6840-6851.

            [2] Nichol, Alexander Quinn, and Prafulla Dhariwal. "Improved denoising diffusion probabilistic models."
            In International Conference on Machine Learning, pp. 8162-8171. PMLR, 2021.

            [3] Salimans, Tim, and Jonathan Ho. "Progressive distillation for fast sampling of diffusion models."
            arXiv preprint arXiv:2202.00512 (2022).

        """
        if objective not in ['pred_eps', 'pred_x0', 'pred_v']:
            raise ValueError(f'Invalid objective: {objective}')
        if var_type not in ['fixed_small', 'fixed_large', 'learned_range']:
            raise ValueError(f'Invalid var_type: {var_type}')

        self.total_steps = total_steps
        self.objective = objective
        self.var_type = var_type
        self.clip_denoised = clip_denoised
        self.device = device

        # Define betas and alphas
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

        # Define respaced sequence for sampling
        if respaced_seq is None:
            respaced_seq = get_respaced_seq(
                total_steps=total_steps,
                respace_type=respace_type,
                respace_steps=respace_steps,
            )
        assert isinstance(respaced_seq, Tensor)
        assert respaced_seq.ndim == 1
        self.respaced_seq = respaced_seq.to(device)

    def set_respaced_seq(self, respace_type: str = 'uniform', respace_steps: int = 100):
        self.respaced_seq = get_respaced_seq(
            total_steps=self.total_steps,
            respace_type=respace_type,
            respace_steps=respace_steps,
        ).to(self.device)

    def pred_x0_from_eps(self, xt: Tensor, t: int, eps: Tensor):
        sqrt_recip_alphas_cumprod_t = (1. / self.alphas_cumprod[t]) ** 0.5
        sqrt_recipm1_alphas_cumprod_t = (1. / self.alphas_cumprod[t] - 1.) ** 0.5
        return sqrt_recip_alphas_cumprod_t * xt - sqrt_recipm1_alphas_cumprod_t * eps

    def pred_eps_from_x0(self, xt: Tensor, t: int, x0: Tensor):
        sqrt_recip_alphas_cumprod_t = (1. / self.alphas_cumprod[t]) ** 0.5
        sqrt_recipm1_alphas_cumprod_t = (1. / self.alphas_cumprod[t] - 1.) ** 0.5
        return (sqrt_recip_alphas_cumprod_t * xt - x0) / sqrt_recipm1_alphas_cumprod_t

    def pred_x0_from_v(self, xt: Tensor, t: int, v: Tensor):
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]) ** 0.5
        return sqrt_alphas_cumprod_t * xt - sqrt_one_minus_alphas_cumprod * v

    def pred_eps_from_v(self, xt: Tensor, t: int, v: Tensor):
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]) ** 0.5
        return sqrt_one_minus_alphas_cumprod * xt + sqrt_alphas_cumprod_t * v

    def loss_func(self, model: nn.Module, x0: Tensor, t: Tensor, eps: Tensor = None, model_kwargs: Dict = None):
        if model_kwargs is None:
            model_kwargs = dict()
        if eps is None:
            eps = torch.randn_like(x0)
        xt = self.q_sample(x0, t, eps)
        if self.objective == 'pred_eps':
            pred_eps = model(xt, t, **model_kwargs)
            return F.mse_loss(pred_eps, eps)
        elif self.objective == 'pred_x0':
            pred_x0 = model(xt, t, **model_kwargs)
            return F.mse_loss(pred_x0, x0)
        elif self.objective == 'pred_v':
            v = self.get_v(x0, eps, t)
            pred_v = model(xt, t, **model_kwargs)
            return F.mse_loss(pred_v, v)
        else:
            raise ValueError(f'Objective {self.objective} is not supported.')

    def get_v(self, x0: Tensor, eps: Tensor, t: Tensor):
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        while sqrt_alphas_cumprod_t.ndim < x0.ndim:
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)

        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]) ** 0.5
        while sqrt_one_minus_alphas_cumprod.ndim < x0.ndim:
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        v = sqrt_alphas_cumprod_t * eps - sqrt_one_minus_alphas_cumprod * x0
        return v

    def q_sample(self, x0: Tensor, t: Tensor, eps: Tensor = None):
        """Sample from q(xt | x0).

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

    def p_sample(
            self, model_output: Tensor, xt: Tensor, t: int, t_prev: int,
            var_type: str = None, clip_denoised: bool = None,
    ):
        """Sample from p_theta(x{t-1} | xt).

        Args:
            model_output: Output of the UNet model.
            xt: A Tensor of shape [B, D, ...], the noisy samples.
            t: Current timestep.
            t_prev: Previous timestep.
            var_type: Override self.var_type if not None.
            clip_denoised: Override self.clip_denoised if not None.

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

        # Calculate the predicted x0
        if self.objective == 'pred_eps':
            pred_eps = model_output
            pred_x0 = self.pred_x0_from_eps(xt, t, pred_eps)
        elif self.objective == 'pred_x0':
            pred_x0 = model_output
            pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)
        elif self.objective == 'pred_v':
            pred_v = model_output
            pred_x0 = self.pred_x0_from_v(xt, t, pred_v)
            pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)
        else:
            raise ValueError(f'Invalid objective: {self.objective}')
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)

        # Calculate the mean of p_theta(x{t-1} | xt)
        mean_coef1 = (alphas_cumprod_t_prev ** 0.5) * betas_t / (1. - alphas_cumprod_t)
        mean_coef2 = (alphas_t ** 0.5) * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
        mean = mean_coef1 * pred_x0 + mean_coef2 * xt

        # Calculate the variance of p_theta(x{t-1} | xt)
        reverse_eps = torch.randn_like(xt)
        if t == 0:
            var = torch.zeros_like(betas_t)
            sample = mean
        else:
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
                frac = (learned_var + 1) / 2  # [-1, 1] -> [0, 1]
                logvar = frac * max_logvar + (1 - frac) * min_logvar
                var = torch.exp(logvar)
            else:
                raise ValueError
            sample = mean + torch.exp(0.5 * logvar) * reverse_eps

        return {
            'sample': sample,
            'mean': mean,
            'var': var,
            'pred_x0': pred_x0,
            'pred_eps': pred_eps,
            'reverse_eps': reverse_eps,
        }

    def sample_loop(
            self, model: nn.Module, init_noise: Tensor,
            var_type: str = None, clip_denoised: bool = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        if tqdm_kwargs is None:
            tqdm_kwargs = dict()
        if model_kwargs is None:
            model_kwargs = dict()
        img = init_noise
        sample_seq = self.respaced_seq.tolist()
        sample_seq_prev = [-1] + self.respaced_seq[:-1].tolist()
        pbar = tqdm.tqdm(total=len(sample_seq), **tqdm_kwargs)
        for t, t_prev in zip(reversed(sample_seq), reversed(sample_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.p_sample(model_output, img, t, t_prev, var_type, clip_denoised)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample(
            self, model: nn.Module, init_noise: Tensor,
            var_type: str = None, clip_denoised: bool = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        sample = None
        for out in self.sample_loop(model, init_noise, var_type, clip_denoised, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample
