import torch
import torch.nn as nn
from torch import Tensor

from diffusions.guided import Guided


class MaskGuided(Guided):
    r""" Diffusion Models with Mask Guidance.

    The idea was first proposed in ref[1] and further developed in ref[2], ref[3], etc. for image inpainting.
    In each reverse step, xt is computed by composing the noisy known part and denoised unknown part of the image.

    .. math::
        x_{t−1} = m \odot x^{known}_{t−1} + (1 − m) \odot x^{unknown}_{t-1}

    References:
        [1]. Song, Yang, and Stefano Ermon. “Generative modeling by estimating gradients of the data distribution.”
        Advances in neural information processing systems 32 (2019).

        [2]. Avrahami, Omri, Dani Lischinski, and Ohad Fried. “Blended diffusion for text-driven editing of natural
        images.” In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 18208
        -18218. 2022.

        [3]. Lugmayr, Andreas, Martin Danelljan, Andres Romero, Fisher Yu, Radu Timofte, and Luc Van Gool. “Repaint:
        Inpainting using denoising diffusion probabilistic models.” In Proceedings of the IEEE/CVF Conference on
        Computer Vision and Pattern Recognition, pp. 11461-11471. 2022.

    """
    def __init__(
            self,
            masked_image: Tensor = None,
            mask: Tensor = None,
            **kwargs,
    ):
        """ Diffusion Models with Mask Guidance.

        Args:
            masked_image: The masked input images of shape [B, C, H, W].
            mask: The binary masks of shape [B, 1, H, W]. Note that 1 denotes known areas and 0 denotes unknown areas.

        """
        super().__init__(**kwargs)
        self.masked_image = masked_image
        self.mask = mask

    def set_mask_and_image(self, masked_image: Tensor, mask: Tensor):
        self.masked_image = masked_image
        self.mask = mask

    def cond_fn_sample(self, t: int, t_prev: int, sample: Tensor, **kwargs) -> Tensor:
        assert self.masked_image is not None, f'Please call `set_mask_and_image()` before sampling.'
        assert self.mask is not None, f'Please call `set_mask_and_image()` before sampling.'
        if t == 0:
            noisy_known = self.masked_image
        else:
            noisy_known = self.q_sample(
                x0=self.masked_image,
                t=torch.full((sample.shape[0], ), t_prev, device=self.device),
            )
        return (noisy_known - sample) * self.mask

    def q_sample_one_step(self, xt: Tensor, t: int, t_next: int):
        """ Sample from q(x{t+1} | xt). """
        alphas_cumprod_t = self.alphas_cumprod[t]
        alphas_cumprod_t_next = self.alphas_cumprod[t_next] if t_next < self.total_steps else torch.tensor(0.0)
        alphas_t_next = alphas_cumprod_t_next / alphas_cumprod_t
        return torch.sqrt(alphas_t_next) * xt + torch.sqrt(1. - alphas_t_next) * torch.randn_like(xt)

    def resample_loop(self, model: nn.Module, init_noise: Tensor,
                      var_type: str = None, clip_denoised: bool = None,
                      resample_r: int = 10, resample_j: int = 10, **model_kwargs):
        """ Sample following RePaint paper. """
        img = init_noise
        resample_seq1 = self.get_resample_seq(resample_r, resample_j)
        resample_seq2 = resample_seq1[1:] + [-1]
        for t1, t2 in zip(resample_seq1, resample_seq2):
            if t1 > t2:
                t_batch = torch.full((img.shape[0], ), t1, device=self.device, dtype=torch.long)
                model_output = model(img, t_batch, **model_kwargs)
                out = self.p_sample(model_output, img, t1, t2, var_type, clip_denoised)
                img = out['sample']
                yield out
            else:
                img = self.q_sample_one_step(img, t1, t2)
                yield {'sample': img}

    def resample(self, model: nn.Module, init_noise: Tensor,
                 var_type: str = None, clip_denoised: bool = None,
                 resample_r: int = 10, resample_j: int = 10, **model_kwargs):
        sample = None
        for out in self.resample_loop(
                model, init_noise, var_type, clip_denoised, resample_r, resample_j, **model_kwargs,
        ):
            sample = out['sample']
        return sample

    def get_resample_seq(self, resample_r: int = 10, resample_j: int = 10):
        """ Figure 9 in RePaint paper.

        Args:
            resample_r: Number of resampling, as proposed in RePaint paper.
            resample_j: Jump lengths of resampling, as proposed in RePaint paper.

        """
        t_T = len(self.skip_seq)

        jumps = {}
        for j in range(0, t_T - resample_j, resample_j):
            jumps[j] = resample_r - 1

        t = t_T
        ts = []
        while t >= 1:
            t = t - 1
            ts.append(self.skip_seq[t].item())
            if jumps.get(t, 0) > 0:
                jumps[t] = jumps[t] - 1
                for _ in range(resample_j):
                    t = t + 1
                    ts.append(self.skip_seq[t].item())
        return ts


def _test(r, j):
    import matplotlib.pyplot as plt

    dummy_image = torch.rand((10, 3, 256, 256))
    dummy_mask = torch.randint(0, 2, (10, 1, 256, 256))
    mask_guided = MaskGuided(dummy_image, dummy_mask, skip_type='uniform', skip_steps=250)

    ts = mask_guided.get_resample_seq(resample_r=r, resample_j=j)
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.plot(range(len(ts)), ts)
    plt.title(f'r={r}, j={j}')
    plt.show()


if __name__ == '__main__':
    _test(1, 10)
    _test(5, 10)
    _test(10, 10)
