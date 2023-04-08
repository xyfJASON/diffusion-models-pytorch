import torch
from torch import Tensor
import torchvision.transforms as T

from transformers import CLIPProcessor, CLIPModel

from diffusions.guided import Guided
from utils.misc import image_norm_to_float


class CLIPGuided(Guided):
    """ Diffusion Models with CLIP Guidance.

    Guide the diffusion process with similarity between CLIP image feature and text feature,
    so that the generated image matches the description of the input text.

    In each step, the guidance is applied on the predicted x0 to avoid training CLIP on noisy images.

    """
    def __init__(
            self,
            guidance_weight: float = 1.0,
            clip_pretrained: str = 'openai/clip-vit-base-patch32',
            **kwargs,
    ):
        super().__init__(**kwargs)
        self.guidance_weight = guidance_weight

        self.clip_processor = CLIPProcessor.from_pretrained(clip_pretrained)
        self.clip_model = CLIPModel.from_pretrained(clip_pretrained).to(self.device)

        self.text = None

    def set_text(self, text: str):
        assert isinstance(text, str)
        self.text = text

    @torch.enable_grad()
    def cond_fn_mean(self, t: int, xt: Tensor, pred_x0: Tensor, pred_eps: Tensor,
                     mean: Tensor, logvar: Tensor) -> Tensor:
        assert self.text is not None, f'Please call `set_text()` before sampling.'
        images = image_norm_to_float(pred_x0)
        processed = self.clip_processor(text=self.text, images=images, return_tensors="pt", padding=True)
        processed = {k: v.to(self.device) for k, v in processed.items()}
        processed['pixel_values'].requires_grad_(True)
        out = self.clip_model(**processed)
        similarities = torch.matmul(out['image_embeds'], out['text_embeds'].t()).squeeze(dim=1)
        grad = torch.autograd.grad(outputs=similarities.sum(), inputs=processed['pixel_values'])[0]
        grad = T.Resize(xt.shape[-2:], antialias=True)(grad)
        return self.guidance_weight * ((1. / self.alphas_cumprod[t]) ** 0.5) * torch.exp(logvar) * grad
