import torch
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

    def denoise(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int):
        """Denoise from xt to x{t-1}."""
        # Predict x0 and eps
        predict = self.predict(model_output, xt, t)
        pred_x0 = predict['pred_x0']
        pred_eps = predict['pred_eps']

        # Prepare parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Calculate the x{t-1}
        derivative = 1 / (2 * alphas_cumprod_t) * (xt - pred_eps / (1 - alphas_cumprod_t).sqrt())
        sample = xt + derivative * (alphas_cumprod_t_prev - alphas_cumprod_t)

        return {'sample': sample, 'pred_x0': pred_x0, 'pred_eps': pred_eps}
