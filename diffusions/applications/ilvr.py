import torch
from torch import Tensor

from diffusions.guided import Guided
from utils.resize_right import resize_right, interp_methods


class ILVR(Guided):
    r""" Iterative Latent Variable Refinement (ILVR).

    References:
        Choi, Jooyoung, Sungwon Kim, Yonghyun Jeong, Youngjune Gwon, and Sungroh Yoon. “ILVR: Conditioning Method
        for Denoising Diffusion Probabilistic Models.” In 2021 IEEE/CVF International Conference on Computer Vision
        (ICCV), pp. 14347-14356. IEEE, 2021.

    """
    def __init__(
            self,
            ref_images: Tensor = None,
            downsample_factor: int = 8,
            interp_method: str = 'cubic',
            **kwargs,
    ):
        """ Iterative Latent Variable Refinement (ILVR).

        Args:
            ref_images: The reference images of shape [B, C, H, W].

        """
        super().__init__(**kwargs)
        self.ref_images = ref_images
        self.downsample_factor = downsample_factor
        self.interp_method = getattr(interp_methods, interp_method)

    def set_ref_images(self, ref_images: Tensor):
        self.ref_images = ref_images

    def cond_fn_sample(self, t: int, t_prev: int, sample: Tensor, **kwargs) -> Tensor:
        assert self.ref_images is not None, f'Please call `set_ref_images()` before sampling.'
        if t == 0:
            noisy_ref_images = self.ref_images
        else:
            noisy_ref_images = self.q_sample(
                x0=self.ref_images,
                t=torch.full((sample.shape[0], ), t_prev, device=self.device),
            )
        return self.low_pass_filter(noisy_ref_images) - self.low_pass_filter(sample)

    def low_pass_filter(self, x: Tensor):
        x = resize_right.resize(x, scale_factors=1./self.downsample_factor, interp_method=self.interp_method)
        x = resize_right.resize(x, scale_factors=self.downsample_factor, interp_method=self.interp_method)
        return x
