import tqdm
from typing import Dict, Any
from contextlib import contextmanager

import torch
import torch.nn as nn
from torch import Tensor

from diffusions.ddpm import DDPM


class DDIM(DDPM):
    def __init__(
            self,
            total_steps: int = 1000,
            beta_schedule: str = 'linear',
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            betas: Tensor = None,
            objective: str = 'pred_eps',

            clip_denoised: bool = True,
            respace_type: str = None,
            respace_steps: int = 100,
            respaced_seq: Tensor = None,
            eta: float = 0.,

            device: torch.device = 'cpu',
            **kwargs,
    ):
        """Denoising Diffusion Implicit Models.

        Args:
            eta: DDIM hyperparameter.

        References:
            [1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models."
            arXiv preprint arXiv:2010.02502 (2020).

        """
        super().__init__(
            total_steps=total_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            betas=betas,
            objective=objective,
            clip_denoised=clip_denoised,
            respace_type=respace_type,
            respace_steps=respace_steps,
            respaced_seq=respaced_seq,
            device=device,
            **kwargs,
        )
        self.eta = eta

    def denoise(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int):
        """Sample from p_theta(x{t-1} | xt) """
        # Predict x0 and eps
        predict = self.predict(model_output, xt, t)
        pred_x0 = predict['pred_x0']
        pred_eps = predict['pred_eps']

        # Prepare parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Calculate the mean and variance of p_theta(x{t-1} | xt)
        var = ((self.eta ** 2) *
               (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t) *
               (1. - alphas_cumprod_t / alphas_cumprod_t_prev))
        mean = (torch.sqrt(alphas_cumprod_t_prev) * pred_x0 +
                torch.sqrt(1. - alphas_cumprod_t_prev - var) * pred_eps)

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

    def denoise_inversion(self, model_output: Tensor, xt: Tensor, t: int, t_next: int):
        """Sample x{t+1} from xt, only valid for DDIM (eta=0) """
        if self.eta != 0.:
            raise ValueError(f'DDIM inversion is only valid when eta=0, get {self.eta}')

        # Predict x0 and eps
        predict = self.predict(model_output, xt, t)
        pred_x0 = predict['pred_x0']
        pred_eps = predict['pred_eps']

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t_next = self.alphas_cumprod[t_next] if t_next < self.total_steps else torch.tensor(0.0)

        # Sample x{t+1}
        sample = (torch.sqrt(alphas_cumprod_t_next) * pred_x0 +
                  torch.sqrt(1. - alphas_cumprod_t_next) * pred_eps)
        return {'sample': sample, 'pred_x0': pred_x0, 'pred_eps': pred_eps}

    def sample_inversion_loop(
            self, model: nn.Module, img: Tensor,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        tqdm_kwargs = dict() if tqdm_kwargs is None else tqdm_kwargs
        model_kwargs = dict() if model_kwargs is None else model_kwargs

        sample_seq = self.respaced_seq[:-1].tolist()
        sample_seq_next = self.respaced_seq[1:].tolist()
        pbar = tqdm.tqdm(total=len(sample_seq), **tqdm_kwargs)
        for t, t_next in zip(sample_seq, sample_seq_next):
            t_batch = torch.full((img.shape[0], ), t, device=img.device, dtype=torch.long)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.denoise_inversion(model_output, img, t, t_next)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample_inversion(
            self, model: nn.Module, img: Tensor,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        sample = None
        for out in self.sample_inversion_loop(model, img, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample


class DDIMCFG(DDIM):
    def __init__(self, guidance_scale: float = 1., cond_kwarg: str = 'y', *args, **kwargs):
        """Denoising Diffusion Implicit Models with Classifier-Free Guidance.

        Args:
            guidance_scale: Strength of guidance. Note we actually use the definition in classifier guidance paper
             instead of classifier-free guidance paper. Specifically, let the former be `s` and latter be `w`, then we
             have `s=w+1`, where `s=0` means unconditional generation, `s=1` means non-guided conditional generation,
             and `s>1` means guided conditional generation.
            cond_kwarg: Name of the condition argument passed to model. Default to `y`.

        References:
            [1] Song, Jiaming, Chenlin Meng, and Stefano Ermon. "Denoising diffusion implicit models."
            arXiv preprint arXiv:2010.02502 (2020).

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

    def sample_inversion_loop(
            self, model: nn.Module, img: Tensor, uncond_conditioning: Any = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        tqdm_kwargs = dict() if tqdm_kwargs is None else tqdm_kwargs

        if self.cond_kwarg not in model_kwargs.keys():
            raise ValueError(f'Condition argument `{self.cond_kwarg}` not found in model_kwargs.')
        uncond_model_kwargs = model_kwargs.copy()
        uncond_model_kwargs[self.cond_kwarg] = uncond_conditioning

        sample_seq = self.respaced_seq[:-1].tolist()
        sample_seq_next = self.respaced_seq[1:].tolist()
        pbar = tqdm.tqdm(total=len(sample_seq), **tqdm_kwargs)
        for t, t_next in zip(sample_seq, sample_seq_next):
            t_batch = torch.full((img.shape[0], ), t, device=img.device, dtype=torch.long)
            # conditional branch
            model_output_cond = model(img, t_batch, **model_kwargs)
            pred_eps_cond = self.predict(model_output_cond, img, t)['pred_eps']
            # unconditional branch
            model_output_uncond = model(img, t_batch, **uncond_model_kwargs)
            pred_eps_uncond = self.predict(model_output_uncond, img, t)['pred_eps']
            # combine
            pred_eps = (1 - self.guidance_scale) * pred_eps_uncond + self.guidance_scale * pred_eps_cond
            with self.hack_objective('pred_eps'):
                out = self.denoise_inversion(pred_eps, img, t, t_next)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample_inversion(
            self, model: nn.Module, img: Tensor,
            clip_denoised: bool = None, eta: float = None,
            guidance_scale: float = None, uncond_conditioning: Any = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        sample = None
        for out in self.sample_inversion_loop(model, img, uncond_conditioning, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample

    @contextmanager
    def hack_objective(self, objective: str):
        """Hack objective temporarily."""
        tmp = self.objective
        self.objective = objective
        yield
        self.objective = tmp
