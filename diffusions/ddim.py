import tqdm
from typing import Dict

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

            var_type: str = 'fixed_large',
            clip_denoised: bool = True,
            respace_type: str = None,
            respace_steps: int = 100,
            respaced_seq: Tensor = None,
            eta: float = 0.,

            device: torch.device = 'cpu',
    ):
        """Denoising Diffusion Implicit Models.

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

            var_type: Not used in DDIM.
            clip_denoised: Clip the predicted x0 in range [-1, 1]. This argument doesn't affect training and can be
             overridden in sampling functions.
            respace_type: Type of respaced timestep sequence. Options: 'uniform', 'uniform-leading', 'uniform-linspace',
             'uniform-trailing', 'quad', 'none', None. This argument doesn't affect training and can be overridden in
             `set_respaced_seq()`.
            respace_steps: Length of respaced timestep sequence, i.e., number of sampling steps during inference. This
             argument doesn't affect training and can be overridden in `set_respaced_seq()`.
            respaced_seq: A 1-D Tensor of pre-defined respaced sequence. If provided, arguments `respace_*` will be
             ignored. This argument doesn't affect training and can be overridden in `set_respaced_seq()`.
            eta: DDIM hyperparameter. This argument doesn't affect training and can be overridden in sampling functions.

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
            var_type=var_type,
            clip_denoised=clip_denoised,
            respace_type=respace_type,
            respace_steps=respace_steps,
            respaced_seq=respaced_seq,
            device=device,
        )
        self.eta = eta
        self.device = device

    def p_sample(
            self, model_output: Tensor, xt: Tensor, t: int, t_prev: int,
            clip_denoised: bool = True, eta: float = None,
    ):
        """Sample from p_theta(x{t-1} | xt) """
        if eta is None:
            eta = self.eta
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_prev = self.alphas_cumprod[t_prev] if t_prev >= 0 else torch.tensor(1.0)

        # Process model's output
        if model_output.shape[1] > xt.shape[1]:
            model_output, _ = torch.split(model_output, xt.shape[1], dim=1)

        # Calculate the predicted x0 and predicted eps
        if self.objective == 'pred_eps':
            pred_eps = model_output
            pred_x0 = self.pred_x0_from_eps(xt, t, pred_eps)
        elif self.objective == 'pred_x0':
            pred_x0 = model_output
        elif self.objective == 'pred_v':
            pred_v = model_output
            pred_x0 = self.pred_x0_from_v(xt, t, pred_v)
        else:
            raise ValueError
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)
        pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)

        # Calculate the mean and variance of p_theta(x{t-1} | xt)
        var = ((eta ** 2) *
               (1. - alphas_cumprod_t_prev) / (1. - alphas_cumprod_t) *
               (1. - alphas_cumprod_t / alphas_cumprod_t_prev))
        mean = (torch.sqrt(alphas_cumprod_t_prev) * pred_x0 +
                torch.sqrt(1. - alphas_cumprod_t_prev - var) * pred_eps)
        reverse_eps = torch.randn_like(xt)
        if t == 0:
            sample = mean
        else:
            sample = mean + torch.sqrt(var) * reverse_eps
        return {'sample': sample, 'pred_x0': pred_x0, 'pred_eps': pred_eps, 'reverse_eps': reverse_eps}

    def p_sample_inversion(
            self, model_output: Tensor, xt: Tensor, t: int, t_next: int,
            clip_denoised: bool = None, eta: float = None,
    ):
        """Sample x{t+1} from xt, only valid for DDIM (eta=0) """
        if eta is None:
            eta = self.eta
        if eta != 0.:
            raise ValueError('DDIM inversion is only valid when eta=0')
        if clip_denoised is None:
            clip_denoised = self.clip_denoised

        # Prepare alphas, betas and other parameters
        alphas_cumprod_t_next = self.alphas_cumprod[t_next] if t_next < self.total_steps else torch.tensor(0.0)

        # Process model's output
        if model_output.shape[1] > xt.shape[1]:
            model_output, _ = torch.split(model_output, xt.shape[1], dim=1)

        # Calculate the predicted x0 and predicted eps
        if self.objective == 'pred_eps':
            pred_eps = model_output
            pred_x0 = self.pred_x0_from_eps(xt, t, pred_eps)
        elif self.objective == 'pred_x0':
            pred_x0 = model_output
        elif self.objective == 'pred_v':
            pred_v = model_output
            pred_x0 = self.pred_x0_from_v(xt, t, pred_v)
        else:
            raise ValueError
        if clip_denoised:
            pred_x0.clamp_(-1., 1.)
        pred_eps = self.pred_eps_from_x0(xt, t, pred_x0)

        # Calculate x{t+1}
        sample = (torch.sqrt(alphas_cumprod_t_next) * pred_x0 +
                  torch.sqrt(1. - alphas_cumprod_t_next) * pred_eps)
        return {'sample': sample, 'pred_x0': pred_x0, 'pred_eps': pred_eps}

    def sample_loop(
            self, model: nn.Module, init_noise: Tensor,
            clip_denoised: bool = None, eta: float = None,
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
            t_batch = torch.full((img.shape[0], ), t, device=self.device, dtype=torch.long)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.p_sample(model_output, img, t, t_prev, clip_denoised, eta)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample(
            self, model: nn.Module, init_noise: Tensor,
            clip_denoised: bool = None, eta: float = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        sample = None
        for out in self.sample_loop(model, init_noise, clip_denoised, eta, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample

    def sample_inversion_loop(
            self, model: nn.Module, img: Tensor,
            clip_denoised: bool = None, eta: float = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        if tqdm_kwargs is None:
            tqdm_kwargs = dict()
        if model_kwargs is None:
            model_kwargs = dict()
        sample_seq = self.respaced_seq[:-1].tolist()
        sample_seq_next = self.respaced_seq[1:].tolist()
        pbar = tqdm.tqdm(total=len(sample_seq), **tqdm_kwargs)
        for t, t_next in zip(sample_seq, sample_seq_next):
            t_batch = torch.full((img.shape[0], ), t, device=img.device, dtype=torch.long)
            model_output = model(img, t_batch, **model_kwargs)
            out = self.p_sample_inversion(model_output, img, t, t_next, clip_denoised, eta)
            img = out['sample']
            pbar.update(1)
            yield out
        pbar.close()

    def sample_inversion(
            self, model: nn.Module, img: Tensor,
            clip_denoised: bool = None, eta: float = None,
            tqdm_kwargs: Dict = None, model_kwargs: Dict = None,
    ):
        sample = None
        for out in self.sample_inversion_loop(model, img, clip_denoised, eta, tqdm_kwargs, model_kwargs):
            sample = out['sample']
        return sample
