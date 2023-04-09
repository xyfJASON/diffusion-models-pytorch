import os
import math
from PIL import Image, ImageDraw
from typing import Tuple, List, Union

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class DatasetWithMask(Dataset):
    """ Wraps a dataset to return images with masks. """
    def __init__(
            self,
            dataset: Dataset,
            mask_type: Union[str, List[str]] = 'center',
            dir_path: str = None,
            dir_invert_color: bool = False,
            center_length_ratio: Tuple[float, float] = (0.25, 0.25),
            rect_num: Tuple[int, int] = (1, 4),
            rect_length_ratio: Tuple[float, float] = (0.2, 0.8),
            brush_num: Tuple[int, int] = (1, 9),
            brush_n_vertex: Tuple[int, int] = (4, 18),
            brush_mean_angle: float = 2 * math.pi / 5,
            brush_angle_range: float = 2 * math.pi / 15,
            brush_width_ratio: Tuple[float, float] = (0.02, 0.1),
            is_train: bool = False,
    ):
        self.dataset = dataset
        self.mask_generator = MaskGenerator(
            mask_type=mask_type,
            dir_path=dir_path,
            dir_invert_color=dir_invert_color,
            center_length_ratio=center_length_ratio,
            rect_num=rect_num,
            rect_length_ratio=rect_length_ratio,
            brush_num=brush_num,
            brush_n_vertex=brush_n_vertex,
            brush_mean_angle=brush_mean_angle,
            brush_angle_range=brush_angle_range,
            brush_width_ratio=brush_width_ratio,
            is_train=is_train,
        )

    def __len__(self):
        return len(self.dataset)  # type: ignore

    def __getitem__(self, item):
        image = self.dataset[item]
        image = image[0] if isinstance(image, (tuple, list)) else image
        C, H, W = image.shape
        mask = self.mask_generator.sample(int(H), int(W), item)
        return image, mask


class MaskGenerator:
    def __init__(
            self,
            mask_type: Union[str, List[str]] = 'center',
            dir_path: str = None,                                       # dir
            dir_invert_color: bool = False,                             # dir
            center_length_ratio: Tuple[float, float] = (0.25, 0.25),    # center
            rect_num: Tuple[int, int] = (1, 4),                         # rect
            rect_length_ratio: Tuple[float, float] = (0.2, 0.8),        # rect
            brush_num: Tuple[int, int] = (1, 9),                        # brush
            brush_n_vertex: Tuple[int, int] = (4, 18),                  # brush
            brush_mean_angle: float = 2 * math.pi / 5,                  # brush
            brush_angle_range: float = 2 * math.pi / 15,                # brush
            brush_width_ratio: Tuple[float, float] = (0.02, 0.1),       # brush
            is_train: bool = False,
    ):
        """ Generates random masks of various types.

        Args:
            mask_type: Type of the masks.
                Options:
                 - 'dir': masks are stored in a directory as binary images.
                 - 'center': a rectangular mask at the center of the image.
                 - 'rect': rectangular masks of random sizes at random positions.
                 - 'brush': random brushstroke-like masks at random positions.
                 - 'half': mask half of the image.
                 - 'every-second-line': mask every second line of the image.
                 - 'sr2x': mask every second row and every second column of the image.
                Multiple types of masks can overlap with each other by giving them in a list.

            dir_path: Valid only when `mask_type` is or contains 'dir'.
                Path to the directory storing mask images.

            dir_invert_color: Valid only when `mask_type` is or contains 'dir'.
                Whether to invert the color (black to white and vice versa) or not.
                Note that white pixels denote unmasked region and black pixels denote holes.

            center_length_ratio: Valid only when `mask_type` is or contains 'center'.
                The ratio of the edge length of the central rectangle to the edge length of the entire image.

            rect_num: Valid only when `mask_type` is or contains 'rect'.
                The number of rectangular masks.

            rect_length_ratio: Valid only when `mask_type` is or contains 'rect'.
                The ratio of the edge length of the rectangles to the edge length of the entire image.

            brush_num: Valid only when `mask_type` is or contains 'brush'.
                The number of brushstroke-like masks.

            brush_n_vertex: Valid only when `mask_type` is or contains 'brush'.
                The number of vertices in a brushstroke.

            brush_mean_angle: Valid only when `mask_type` is or contains 'brush'.
                The mean of angle of each turn in a brushstroke.

            brush_angle_range: Valid only when `mask_type` is or contains 'brush'.
                The range of angle of each turn in a brushstroke.

            brush_width_ratio: Valid only when `mask_type` is or contains 'brush'.
                The ratio of the width of the brushstrokes to the edge length of the entire image.

            is_train: Whether the masks are generated for training set or not.
                If False, the generation process will be seeded on the index `item` in `sample()`.

        """
        self.mask_type = mask_type
        self.dir_invert_color = dir_invert_color
        self.center_length_ratio = center_length_ratio
        self.rect_num = rect_num
        self.rect_length_ratio = rect_length_ratio
        self.brush_num = brush_num
        self.brush_n_vertex = brush_n_vertex
        self.brush_mean_angle = brush_mean_angle
        self.brush_angle_range = brush_angle_range
        self.brush_width_ratio = brush_width_ratio
        self.is_train = is_train

        if isinstance(mask_type, str):
            self.mask_type = [mask_type]
        self.mask_type = list(set(self.mask_type))
        if 'dir' in self.mask_type:
            dir_path = os.path.expanduser(dir_path)
            assert os.path.isdir(dir_path), f'{dir_path} is not a valid directory'
            img_ext = ['.png', '.jpg', '.jpeg']
            self.mask_paths = []
            for curdir, subdir, files in os.walk(dir_path):
                for file in files:
                    if os.path.splitext(file)[1].lower() in img_ext:
                        self.mask_paths.append(os.path.join(curdir, file))
            self.mask_paths = sorted(self.mask_paths)

    def sample(self, H: int, W: int, item: int = None):
        if isinstance(item, torch.Tensor):
            item = item.item()
        if self.is_train is False and item is not None:
            rndgn = torch.Generator()
            rndgn.manual_seed(item + 3407)
        else:
            rndgn = torch.default_generator

        mask = torch.ones((1, H, W), dtype=torch.bool)
        for t in self.mask_type:
            if t == 'dir':
                m = self._sample_dir(H, W, rndgn)
            elif t == 'center':
                m = self._sample_center(H, W, rndgn)
            elif t == 'rect':
                m = self._sample_rectangles(H, W, rndgn)
            elif t == 'brush':
                m = self._sample_brushes(H, W, rndgn)
            elif t == 'half':
                m = self._sample_half(H, W, rndgn)
            elif t == 'every-second-line':
                m = self._sample_every_second_line(H, W)
            elif t == 'sr2x':
                m = self._sample_sr2x(H, W)
            else:
                raise ValueError(f'mask type {t} is not supported')
            mask = torch.logical_and(mask, m)
        return mask

    def _sample_dir(self, H: int, W: int, rndgn: torch.Generator):
        path = self.mask_paths[torch.randint(0, len(self.mask_paths), (1, ), generator=rndgn).item()]
        mask = Image.open(path)
        mask = T.Resize((H, W))(mask)
        mask = T.ToTensor()(mask)
        if self.dir_invert_color:
            mask = torch.where(mask < 0.5, 1., 0.).bool()
        else:
            mask = torch.where(mask < 0.5, 0., 1.).bool()
        return mask

    def _sample_center(self, H: int, W: int, rndgn: torch.Generator):
        mask = torch.ones((1, H, W)).float()
        min_ratio, max_ratio = self.center_length_ratio
        ratio = torch.rand((1, ), generator=rndgn).item() * (max_ratio - min_ratio) + min_ratio
        h, w = int(ratio * H), int(ratio * W)
        mask[:, H//2-h//2:H//2+h//2, W//2-w//2:W//2+w//2] = 0.
        return mask.bool()

    def _sample_rectangles(self, H: int, W: int, rndgn: torch.Generator):
        min_num, max_num = self.rect_num
        min_ratio, max_ratio = self.rect_length_ratio
        n_rect = torch.randint(min_num, max_num + 1, (1, ), generator=rndgn).item()
        min_h, max_h = int(min_ratio * H), int(max_ratio * H)
        min_w, max_w = int(min_ratio * W), int(max_ratio * W)
        mask = torch.ones((1, H, W)).float()
        for i in range(n_rect):
            h = torch.randint(min_h, max_h + 1, (1, ), generator=rndgn).item()
            w = torch.randint(min_w, max_w + 1, (1, ), generator=rndgn).item()
            y = torch.randint(0, H - h + 1, (1, ), generator=rndgn).item()
            x = torch.randint(0, W - w + 1, (1, ), generator=rndgn).item()
            mask[:, y:y+h, x:x+w] = 0.
        return mask.bool()

    def _sample_brushes(self, H: int, W: int, rndgn: torch.Generator):
        min_num, max_num = self.brush_num
        min_n_vertex, max_n_vertex = self.brush_n_vertex
        min_width = int(self.brush_width_ratio[0] * min(H, W))
        max_width = int(self.brush_width_ratio[1] * min(H, W))
        n_brush = torch.randint(min_num, max_num + 1, (1, ), generator=rndgn).item()
        average_radius = math.sqrt(H * H + W * W) / 8
        mask = Image.new('L', (W, H), 255)
        for i in range(n_brush):
            n_vertex = torch.randint(min_n_vertex, max_n_vertex + 1, (1, ), generator=rndgn).item()
            width = torch.randint(min_width, max_width + 1, (1, ), generator=rndgn).item()
            min_angle = self.brush_mean_angle - torch.rand((1, ), generator=rndgn).item() * self.brush_angle_range
            max_angle = self.brush_mean_angle + torch.rand((1, ), generator=rndgn).item() * self.brush_angle_range
            vertex = [(
                torch.randint(0, W, (1, ), generator=rndgn).item(),
                torch.randint(0, H, (1, ), generator=rndgn).item(),
            )]
            for j in range(n_vertex):
                angle = torch.rand(1, generator=rndgn).item() * (max_angle - min_angle) + min_angle
                if j % 2 == 0:
                    angle = 2 * math.pi - angle
                r = torch.clip(
                    torch.normal(mean=average_radius, std=average_radius // 2, size=(1, ), generator=rndgn),
                    0, 2 * average_radius,
                ).item()
                new_x = min(max(vertex[-1][0] + r * math.cos(angle), 0), W)
                new_y = min(max(vertex[-1][1] + r * math.sin(angle), 0), H)
                vertex.append((new_x, new_y))
            draw = ImageDraw.Draw(mask)
            draw.line(vertex, fill=0, width=width)
            for v in vertex:
                draw.ellipse((v[0] - width // 2,
                              v[1] - width // 2,
                              v[0] + width // 2,
                              v[1] + width // 2), fill=0)
            if torch.rand(1, generator=rndgn) > 0.5:
                mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)  # noqa
            if torch.rand(1, generator=rndgn) > 0.5:
                mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)  # noqa
        if torch.rand(1, generator=rndgn) > 0.5:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)      # noqa
        if torch.rand(1, generator=rndgn) > 0.5:
            mask = mask.transpose(Image.Transpose.FLIP_TOP_BOTTOM)      # noqa
        mask = T.ToTensor()(mask)
        mask = torch.where(mask < 0.5, 0., 1.).bool()
        return mask

    @staticmethod
    def _sample_half(H: int, W: int, rndgn: torch.Generator):
        mask = torch.ones((1, H, W)).float()
        direction = torch.randint(0, 4, (1, ), generator=rndgn).item()
        if direction == 0:
            mask[:, :H//2, :] = 0.
        elif direction == 1:
            mask[:, H//2:, :] = 0.
        elif direction == 2:
            mask[:, :, :W//2] = 0.
        else:
            mask[:, :, W//2:] = 0.
        return mask.bool()

    @staticmethod
    def _sample_every_second_line(H: int, W: int):
        mask = torch.ones((1, H, W)).float()
        mask[:, ::2, :] = 0.
        return mask.bool()

    @staticmethod
    def _sample_sr2x(H: int, W: int):
        mask = torch.ones((1, H, W)).float()
        mask[:, ::2, :] = 0.
        mask[:, :, ::2] = 0.
        return mask.bool()


def _test(**kwargs):
    from torchvision.utils import make_grid
    show = []
    mask_gen = MaskGenerator(**kwargs, is_train=True)
    for i in range(5):
        mask = mask_gen.sample(512, 512, item=0).float()
        show.append(mask)
    mask_gen = MaskGenerator(**kwargs, is_train=False)
    for i in range(5):
        mask = mask_gen.sample(512, 512, item=0).float()
        show.append(mask)
    show = make_grid(show, nrow=5)
    T.ToPILImage()(show).show()


def _test_statistics(num: int = 36500, **kwargs):
    import tqdm
    import numpy as np
    import matplotlib.pyplot as plt
    mask_gen = MaskGenerator(**kwargs, is_train=True)
    ratios = []
    for _ in tqdm.tqdm(range(num)):
        mask = mask_gen.sample(512, 512)
        ratios.append(torch.sum(~mask).float().item() / (512 * 512))
    plt.hist(ratios, bins=np.linspace(0, 1, 20), density=False, rwidth=0.7)
    plt.show()


if __name__ == '__main__':
    # _test(mask_type='center', center_length_ratio=(0.25, 0.5))
    _test(mask_type='brush')
    # _test(mask_type='rect')
    # _test(mask_type=['brush', 'rect'])
    # _test(mask_type=['half', 'sr2x'])
    # _test(mask_type=['every-second-line'])
    # _test(
    #     mask_type='dir',
    #     dir_path='/Users/jason/data/NVIDIAIrregularMaskDataset/train/',
    #     dir_invert_color=True,
    # )
    # _test_statistics(mask_type='brush')
    # _test_statistics(mask_type='rect')
