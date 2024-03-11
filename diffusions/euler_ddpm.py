import tqdm
from typing import Dict

import torch
import torch.nn as nn
from torch import Tensor

from diffusions.ddpm import DDPM


class EulerDDPMSampler(DDPM):
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
        """Euler sampler for DDPM-like diffusion process.

        Args:
            total_steps: Total number of timesteps used to train the model.
            beta_schedule: Type of beta schedule. Options: 'linear', 'quad', 'const', 'cosine'.
            beta_start: Starting beta value.
            beta_end: Ending beta value.
            betas: A 1-D Tensor of pre-defined beta schedule. If provided, arguments `beta_*` will be ignored.
            objective: Prediction objective of the model. Options: 'pred_eps', 'pred_x0', 'pred_v'.

            var_type: Not used in Euler sampler.
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
            [1] Liu, Luping, Yi Ren, Zhijie Lin, and Zhou Zhao. "Pseudo numerical methods for diffusion models on
            manifolds." arXiv preprint arXiv:2202.09778 (2022).

        """
        super().__init__(
            total_steps=total_steps,
            beta_schedule=beta_schedule,
            beta_start=beta_start,
            beta_end=beta_end,
            betas=betas,
            objective=objective,
            var_type=var_type,
            clip_denoised=clip_denoised,
            respace_type=respace_type,
            respace_steps=respace_steps,
            respaced_seq=respaced_seq,
            device=device,
        )
        self.device = device

    def diffuse(self, x0: Tensor, t: Tensor, eps: Tensor = None):
        """Diffuse from x0 to xt."""
        return self.q_sample(x0=x0, t=t, eps=eps)

    def denoise(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int, clip_denoised: bool = True):
        """Denoise from xt to x{t-1}."""
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Process model's output
        if model_output.shape[1] > xt.shape[1]:
            model_output = model_output[:, :xt.shape[1]]

        # Calculate the predicted eps
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
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)
        pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)

        # Calculate the x{t-1}
        derivative = 1 / (2 * alphas_cumprod_t) * (xt - pred_eps / (1 - alphas_cumprod_t).sqrt())
        sample = xt + derivative * (alphas_cumprod_t_prev - alphas_cumprod_t)

        return {'sample': sample, 'pred_x0': pred_x0, 'pred_eps': pred_eps}

    def sample_loop(
            self, model: nn.Module, init_noise: Tensor, clip_denoised: bool = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None, *args, **kwargs,
    ):
        tqdm_kwargs = dict() if tqdm_kwargs is None else tqdm_kwargs
        model_kwargs = dict() if model_kwargs is None else model_kwargs

        img = init_noise
        sample_seq = self.respaced_seq.tolist()
        sample_seq_prev = [-1] + self.respaced_seq[:-1].tolist()
        pbar = tqdm.tqdm(total=len(sample_seq), **tqdm_kwargs)
        for t, t_prev in zip(reversed(sample_seq), reversed(sample_seq_prev)):
            t_batch = torch.full((img.shape[0], ), t, device=self.device)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.denoise(model_output, img, t, t_prev, clip_denoised)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample(
            self, model: nn.Module, init_noise: Tensor, clip_denoised: bool = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None, *args, **kwargs,
    ):
        sample = None
        for out in self.sample_loop(model, init_noise, clip_denoised, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample
