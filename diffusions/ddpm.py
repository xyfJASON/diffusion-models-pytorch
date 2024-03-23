import tqdm
from typing import Dict, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor

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

        Args:
            total_steps: Total number of timesteps used to train the model.
            beta_schedule: Type of beta schedule. Options: 'linear', 'quad', 'const', 'cosine'.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            betas: A 1-D Tensor of pre-defined beta schedule. If provided, arguments `beta_*` will be ignored.
            objective: Prediction objective of the model. Options: 'pred_eps', 'pred_x0', 'pred_v'.

            var_type: Type of variance of the reverse process. Options: 'fixed_large', 'fixed_small', 'learned_range'.
            clip_denoised: Clip the predicted x0 in range [-1, 1].
            respace_type: Type of respaced timestep sequence. Options: 'uniform', 'uniform-leading', 'uniform-linspace',
             'uniform-trailing', 'quad', 'none', None.
            respace_steps: Length of respaced timestep sequence, i.e., number of sampling steps during inference.
            respaced_seq: A 1-D Tensor of pre-defined respaced sequence. If provided, `respace_type` and `respace_steps`
             will be ignored.

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

    def get_v(self, x0: Tensor, eps: Tensor, t: Tensor):
        sqrt_alphas_cumprod_t = self.alphas_cumprod[t] ** 0.5
        while sqrt_alphas_cumprod_t.ndim < x0.ndim:
            sqrt_alphas_cumprod_t = sqrt_alphas_cumprod_t.unsqueeze(-1)

        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]) ** 0.5
        while sqrt_one_minus_alphas_cumprod.ndim < x0.ndim:
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        v = sqrt_alphas_cumprod_t * eps - sqrt_one_minus_alphas_cumprod * x0
        return v

    def diffuse(self, x0: Tensor, t: Tensor, eps: Tensor = None):
        """Sample from q(xt | x0).

        Args:
            x0: A Tensor of shape [B, D, ...], the original samples.
            t: A Tensor of shape [B], timesteps for each sample in x0.
               Note that different samples may have different t.
            eps: A Tensor of shape [B, D, ...], the noise added to the original samples.

        """
        eps = torch.randn_like(x0) if eps is None else eps

        sqrt_alphas_cumprod = self.alphas_cumprod[t] ** 0.5
        while sqrt_alphas_cumprod.ndim < x0.ndim:
            sqrt_alphas_cumprod = sqrt_alphas_cumprod.unsqueeze(-1)

        sqrt_one_minus_alphas_cumprod = (1. - self.alphas_cumprod[t]) ** 0.5
        while sqrt_one_minus_alphas_cumprod.ndim < x0.ndim:
            sqrt_one_minus_alphas_cumprod = sqrt_one_minus_alphas_cumprod.unsqueeze(-1)

        return sqrt_alphas_cumprod * x0 + sqrt_one_minus_alphas_cumprod * eps

    def predict(self, model_output: Tensor, xt: Tensor, t: int):
        """Predict x0 or eps from model's output, i.e., x_theta(xt, t) or eps_theta(xt, t).

        Args:
            model_output: Output of the neural network.
            xt: A Tensor of shape [B, D, ...], the noisy samples.
            t: Current timestep.

        """
        # Process model's output
        learned_var = None
        if model_output.shape[1] > xt.shape[1]:
            model_output, learned_var = torch.split(model_output, xt.shape[1], dim=1)

        # Calculate the predicted x0 or eps
        if self.objective == 'pred_eps':
            pred_eps = model_output
            pred_x0 = self.pred_x0_from_eps(xt, t, pred_eps)
        elif self.objective == 'pred_x0':
            pred_x0 = model_output
        elif self.objective == 'pred_v':
            pred_v = model_output
            pred_x0 = self.pred_x0_from_v(xt, t, pred_v)
        else:
            raise ValueError(f'Invalid objective: {self.objective}')
        if self.clip_denoised:
            pred_x0.clamp_(-1., 1.)
        pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)

        return {'pred_x0': pred_x0, 'pred_eps': pred_eps, 'learned_var': learned_var}

    def denoise(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int):
        """Sample from p_theta(x{t-1} | xt).

        Args:
            model_output: Output of the UNet model.
            xt: A Tensor of shape [B, D, ...], the noisy samples.
            t: Current timestep.
            t_prev: Previous timestep.

        """
        # Predict x0 and eps
        predict = self.predict(model_output, xt, t)
        pred_x0 = predict['pred_x0']
        pred_eps = predict['pred_eps']
        learned_var = predict['learned_var']

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)
        alphas_t = alphas_cumprod_t / alphas_cumprod_t_prev
        betas_t = 1. - alphas_t

        # Calculate the mean of p_theta(x{t-1} | xt)
        mean_coef1 = (alphas_cumprod_t_prev ** 0.5) * betas_t / (1. - alphas_cumprod_t)
        mean_coef2 = (alphas_t ** 0.5) * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
        mean = mean_coef1 * pred_x0 + mean_coef2 * xt

        # Calculate the variance of p_theta(x{t-1} | xt)
        if t == 0:
            var = torch.zeros_like(betas_t)
        else:
            if self.var_type == 'fixed_small':
                var = betas_t * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
            elif self.var_type == 'fixed_large':
                var = betas_t
            elif self.var_type == 'learned_range':
                min_var = betas_t * (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t)
                min_logvar = torch.log(torch.clamp_min(min_var, 1e-20))
                max_logvar = torch.log(betas_t)
                frac = (learned_var + 1) / 2  # [-1, 1] -> [0, 1]
                logvar = frac * max_logvar + (1 - frac) * min_logvar
                var = torch.exp(logvar)
            else:
                raise ValueError(f'Invalid var_type: {self.var_type}')

        # Sample x{t-1}
        reverse_eps = torch.randn_like(xt)
        sample = mean if t == 0 else mean + torch.sqrt(var) * reverse_eps

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
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        tqdm_kwargs = dict() if tqdm_kwargs is None else tqdm_kwargs
        model_kwargs = dict() if model_kwargs is None else model_kwargs

        img = init_noise
        sample_seq = self.respaced_seq.tolist()
        sample_seq_prev = [-1] + self.respaced_seq[:-1].tolist()
        pbar = tqdm.tqdm(total=len(sample_seq), **tqdm_kwargs)
        for t, t_prev in zip(reversed(sample_seq), reversed(sample_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device, dtype=torch.long)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.denoise(model_output, img, t, t_prev)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample(
            self, model: nn.Module, init_noise: Tensor,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        sample = None
        for out in self.sample_loop(model, init_noise, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample


class DDPMCFG(DDPM):
    def __init__(self, guidance_scale: float = 1., cond_kwarg: str = 'y', *args, **kwargs):
        """Denoising Diffusion Probabilistic Models with Classifier-Free Guidance.

        Args:
            guidance_scale: Strength of guidance. Note we actually use the definition in classifier guidance paper
             instead of classifier-free guidance paper. Specifically, let the former be `s` and latter be `w`, then we
             have `s=w+1`, where `s=0` means unconditional generation, `s=1` means non-guided conditional generation,
             and `s>1` means guided conditional generation.
            cond_kwarg: Name of the condition argument passed to model. Default to `y`.

        References:
            [1] Ho, Jonathan, Ajay Jain, and Pieter Abbeel. "Denoising diffusion probabilistic models."
            Advances in neural information processing systems 33 (2020): 6840-6851.

            [2] Ho, Jonathan, and Tim Salimans. "Classifier-free diffusion guidance." arXiv preprint
            arXiv:2207.12598 (2022).

            [3] Dhariwal, Prafulla, and Alexander Nichol. "Diffusion models beat gans on image synthesis." Advances
            in neural information processing systems 34 (2021): 8780-8794.

        """
        super().__init__(*args, **kwargs)
        self.guidance_scale = guidance_scale
        self.cond_kwarg = cond_kwarg

    def sample_loop(
            self, model: nn.Module, init_noise: Tensor, uncond_conditioning: Any = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        tqdm_kwargs = dict() if tqdm_kwargs is None else tqdm_kwargs

        if self.cond_kwarg not in model_kwargs.keys():
            raise ValueError(f'Condition argument `{self.cond_kwarg}` not found in model_kwargs.')
        uncond_model_kwargs = model_kwargs.copy()
        uncond_model_kwargs[self.cond_kwarg] = uncond_conditioning

        img = init_noise
        sample_seq = self.respaced_seq.tolist()
        sample_seq_prev = [-1] + self.respaced_seq[:-1].tolist()
        pbar = tqdm.tqdm(total=len(sample_seq), **tqdm_kwargs)
        for t, t_prev in zip(reversed(sample_seq), reversed(sample_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device)
            # conditional branch
            model_output_cond = model(img, t_batch, **model_kwargs)
            pred_eps_cond = self.predict(model_output_cond, img, t)['pred_eps']
            # unconditional branch
            model_output_uncond = model(img, t_batch, **uncond_model_kwargs)
            pred_eps_uncond = self.predict(model_output_uncond, img, t)['pred_eps']
            # combine
            pred_eps = (1 - self.guidance_scale) * pred_eps_uncond + self.guidance_scale * pred_eps_cond
            if self.var_type == 'learned_range':
                pred_eps = torch.cat([pred_eps, model_output_cond[:, pred_eps.shape[1]:]], dim=1)
            with self.hack_objective('pred_eps'):
                out = self.denoise(pred_eps, img, t, t_prev)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample(
            self, model: nn.Module, init_noise: Tensor, uncond_conditioning: Any = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        sample = None
        for out in self.sample_loop(model, init_noise, uncond_conditioning, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample

    @contextmanager
    def hack_objective(self, objective: str):
        """Hack objective temporarily."""
        tmp = self.objective
        self.objective = objective
        yield
        self.objective = tmp
