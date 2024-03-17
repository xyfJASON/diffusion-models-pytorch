import tqdm
from typing import Dict

import torch
from torch import Tensor, nn as nn

from diffusions.ddpm import DDPM


class HeunSampler(DDPM):
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
        """Heun sampler for DDPM-like diffusion process.

        References:
            [1] Karras, Tero, Miika Aittala, Timo Aila, and Samuli Laine. "Elucidating the design space of
            diffusion-based generative models." Advances in Neural Information Processing Systems 35 (2022):
            26565-26577.

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

        self._1st_order_derivative = None
        self._1st_order_xt = None

    def denoise_1st_order(self, model_output: Tensor, xt: Tensor, t: int, t_prev: int):
        """1st order step. Same as euler sampler."""
        # Prepare parameters
        sigmas_t = self.sigmas[t]
        sigmas_t_prev = self.sigmas[t_prev] if t_prev >= 0 else torch.tensor(0.0)

        # Predict x0
        predict = self.predict(model_output, xt, t)
        pred_x0 = predict['pred_x0']

        # Calculate the x{t-1}
        bar_xt = (1 + sigmas_t ** 2).sqrt() * xt
        derivative = (bar_xt - pred_x0) / sigmas_t
        bar_sample = bar_xt + derivative * (sigmas_t_prev - sigmas_t)
        sample = bar_sample / (1 + sigmas_t_prev ** 2).sqrt()

        # Store the 1st order info
        self._1st_order_derivative = derivative
        self._1st_order_xt = xt

        return {'sample': sample, 'pred_x0': pred_x0}

    def denoise_2nd_order(self, model_output: Tensor, xt_prev: Tensor, t: int, t_prev: int):
        """2nd order step."""
        # Prepare parameters
        sigmas_t = self.sigmas[t]
        sigmas_t_prev = self.sigmas[t_prev] if t_prev >= 0 else torch.tensor(0.0)

        # Predict x0
        predict = self.predict(model_output, xt_prev, t_prev)
        pred_x0 = predict['pred_x0']

        # Calculate derivative
        bar_xt_prev = (1 + sigmas_t_prev ** 2).sqrt() * xt_prev
        derivative = (bar_xt_prev - pred_x0) / sigmas_t_prev
        derivative = (derivative + self._1st_order_derivative) / 2

        # Calculate the x{t-1}
        bar_xt = (1 + sigmas_t ** 2).sqrt() * self._1st_order_xt
        bar_sample = bar_xt + derivative * (sigmas_t_prev - sigmas_t)
        sample = bar_sample / (1 + sigmas_t_prev ** 2).sqrt()

        # Clear the 1st order info
        self._1st_order_derivative = None
        self._1st_order_xt = None

        return {'sample': sample, 'pred_x0': pred_x0}

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
            # 1st order step
            t_batch = torch.full((img.shape[0], ), t, device=self.device, dtype=torch.long)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.denoise_1st_order(model_output, img, t, t_prev)
            img = out['sample']

            if t_prev >= 0:
                # 2nd order step
                t_prev_batch = torch.full((img.shape[0], ), t_prev, device=self.device, dtype=torch.long)
                model_output = model(img, t_prev_batch, **model_kwargs)
                out = self.denoise_2nd_order(model_output, img, t, t_prev)
                img = out['sample']

            pbar.update(1)
            yield out
        pbar.close()
