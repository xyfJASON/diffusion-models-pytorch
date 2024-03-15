import torch
from torch import Tensor

from diffusions.ddpm import DDPM


class EulerSampler(DDPM):
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

            device: torch.device = 'cpu',
            **kwargs,
    ):
        """Euler sampler for DDPM-like diffusion process.

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
            clip_denoised=clip_denoised,
            respace_type=respace_type,
            respace_steps=respace_steps,
            respaced_seq=respaced_seq,
            device=device,
            **kwargs,
        )

        self.sigmas = ((1 - self.alphas_cumprod) / self.alphas_cumprod).sqrt()

    def denoise(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int):
        """Denoise from x_t to x_{t-1}."""
        # Prepare parameters
        sigmas_t = self.sigmas[t]
        sigmas_t_prev = self.sigmas[t_prev] if t_prev >= 0 else torch.tensor(0.0)

        # Predict x0 and eps
        predict = self.predict(model_output, xt, t)
        pred_x0 = predict['pred_x0']

        # Calculate the x{t-1}
        bar_xt = (1 + sigmas_t ** 2).sqrt() * xt
        derivative = (bar_xt - pred_x0) / sigmas_t
        bar_sample = bar_xt + derivative * (sigmas_t_prev - sigmas_t)
        sample = bar_sample / (1 + sigmas_t_prev ** 2).sqrt()

        return {'sample': sample, 'pred_x0': pred_x0}
