import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from diffusions import DDPM


class DDPM_IP(DDPM):
    def __init__(self, gamma: float = 0.1, *args, **kwargs):
        """ Denoising Diffusion Probabilistic Models with Input Perturbation.

        Perturb the input (xt) during training to simulate the gap between training and testing.
        Surprisingly simple but effective.

        Args:
            gamma: Perturbation strength.

        """
        super().__init__(*args, **kwargs)
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
