import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusions import DDPM


class DDPM_IP(DDPM):
    def __init__(
            self,
            total_steps: int = 1000,
            beta_schedule: str = 'linear',
            beta_start: float = 0.0001,
            beta_end: float = 0.02,
            betas: Tensor = None,
            objective: str = 'pred_eps',
            gamma: float = 0.1,

            var_type: str = 'fixed_large',
            clip_denoised: bool = True,
            skip_type: str = None,
            skip_steps: int = 100,
            skip_seq: Tensor = None,

            device: torch.device = 'cpu',
    ):
        """ Denoising Diffusion Probabilistic Models with Input Perturbation.

        Perturb the input (xt) during training to simulate the gap between training and testing.
        Surprisingly simple but effective.

        Args:
            gamma: Perturbation strength.

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
            skip_type=skip_type,
            skip_steps=skip_steps,
            skip_seq=skip_seq,
            device=device,
        )
        self.gamma = gamma

    def loss_func(self, model: nn.Module, x0: Tensor, t: Tensor, eps: Tensor = None, **model_kwargs):
        if eps is None:
            eps = torch.randn_like(x0)
        # input perturbation
        perturbed_eps = eps + self.gamma * torch.randn_like(eps)
        xt = self.q_sample(x0, t, perturbed_eps)
        if self.objective == 'pred_eps':
            pred_eps = model(xt, t, **model_kwargs)
            return F.mse_loss(pred_eps, eps)
        elif self.objective == 'pred_x0':
            pred_x0 = model(xt, t, **model_kwargs)
            return F.mse_loss(pred_x0, x0)
        else:
            raise ValueError(f'Objective {self.objective} is not supported.')
