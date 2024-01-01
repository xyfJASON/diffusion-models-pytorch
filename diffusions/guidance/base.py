import tqdm
from typing import Dict

import torch
from torch import Tensor
import torch.nn as nn

from diffusions.ddpm import DDPM


class BaseGuidance(DDPM):
    def __init__(self, *args, **kwargs):
        """Denoising Diffusion Probabilistic Models with Explicit Guidance.

        The idea was first proposed by [2], and can be used for conditions besides categorial labels, such as text.
        The guidance can be applied to predicted eps, predicted x0, mean of the reverse distribution, or sampled x{t-1}.

        References:
            [1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models."
            Advances in neural information processing systems 33 (2020): 6840-6851.

            [2] Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances
            in neural information processing systems 34 (2021): 8780-8794.

    """
        super().__init__(*args, **kwargs)

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

    def cond_fn_eps(
            self, sample: Tensor, mean: Tensor, var: Tensor, pred_x0: Tensor,
            pred_eps: Tensor, xt: Tensor, t: int, t_prev: int,
    ) -> Tensor:
        """Apply guidance to predicted eps. """
        pass

    def cond_fn_x0(
            self, sample: Tensor, mean: Tensor, var: Tensor, pred_x0: Tensor,
            pred_eps: Tensor, xt: Tensor, t: int, t_prev: int,
    ) -> Tensor:
        """Apply guidance to predicted x0. """
        pass

    def cond_fn_mean(
            self, sample: Tensor, mean: Tensor, var: Tensor, pred_x0: Tensor,
            pred_eps: Tensor, xt: Tensor, t: int, t_prev: int,
    ) -> Tensor:
        """Apply guidance to mean of p(x{t-1} | xt). """
        pass

    def cond_fn_sample(
            self, sample: Tensor, mean: Tensor, var: Tensor, pred_x0: Tensor,
            pred_eps: Tensor, xt: Tensor, t: int, t_prev: int,
    ) -> Tensor:
        """Apply guidance to sampled x{t-1}. """
        pass

    def apply_guidance(
            self,
            sample: Tensor,
            mean: Tensor,
            var: Tensor,
            pred_x0: Tensor,
            pred_eps: Tensor,
            reverse_eps: Tensor,
            xt: Tensor,
            t: int,
            t_prev: int,
    ):
        """Apply guidance to p_theta(x{t-1} | xt).

        Args:
            sample: A Tensor of shape [B, D, ...], the sampled x{t-1} before applying guidance.
            mean: A Tensor of shape [B, D, ...], the mean of p_theta(x{t-1} | xt) before applying guidance.
            var: A one-element Tensor, the variance of p_theta(x{t-1} | xt) before applying guidance.
            pred_x0: A Tensor of shape [B, D, ...], the predicted x0 before applying guidance.
            pred_eps: A Tensor of shape [B, D, ...], the predicted eps before applying guidance.
            reverse_eps: A Tensor of shape [B, D, ...], the reverse eps before applying guidance.
            xt: A Tensor of shape [B, D, ...], the noisy samples.
            t: Current timestep.
            t_prev: Previous timestep.

        """
        new_sample, new_mean, new_x0, new_eps = sample, mean, pred_x0, pred_eps
        cond_fn_kwargs = {
            'sample': sample, 'mean': mean, 'var': var, 'pred_x0': pred_x0,
            'pred_eps': pred_eps, 'xt': xt, 't': t, 't_prev': t_prev,
        }

        # Apply guidance to predicted eps
        guidance = self.cond_fn_eps(**cond_fn_kwargs)
        if guidance is not None:
            new_eps = pred_eps + guidance
            new_x0 = self.pred_x0_from_eps(xt, t, new_eps)
            new_mean = self.pred_mu_from_x0(xt, t, t_prev, new_x0)
            new_sample = new_mean if t == 0 else new_mean + torch.sqrt(var) * reverse_eps

        # Apply guidance to predicted x0
        guidance = self.cond_fn_x0(**cond_fn_kwargs)
        if guidance is not None:
            new_x0 = pred_x0 + guidance
            new_eps = self.pred_eps_from_x0(xt, t, new_x0)
            new_mean = self.pred_mu_from_x0(xt, t, t_prev, new_x0)
            new_sample = new_mean if t == 0 else new_mean + torch.sqrt(var) * reverse_eps

        # Apply guidance to mean of p(x{t-1} | xt)
        guidance = self.cond_fn_mean(**cond_fn_kwargs)
        if guidance is not None:
            new_mean = mean + guidance
            new_x0 = self.pred_x0_from_mu(xt, t, t_prev, new_mean)
            new_eps = self.pred_eps_from_x0(xt, t, new_x0)
            new_sample = new_mean if t == 0 else new_mean + torch.sqrt(var) * reverse_eps

        # Apply guidance to sampled x{t-1}
        guidance = self.cond_fn_sample(**cond_fn_kwargs)
        if guidance is not None:
            new_sample = sample + guidance

        return {
            'sample': new_sample,
            'mean': new_mean,
            'var': var,
            'pred_x0': new_x0,
            'pred_eps': new_eps,
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
            out = self.apply_guidance(**out, xt=img, t=t, t_prev=t_prev)  # apply guidance
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()
