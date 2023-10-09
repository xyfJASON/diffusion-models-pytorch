import os
from PIL import Image
from typing import Optional, Callable

import torchvision.transforms as T
from torch.utils.data import Dataset


def extract_images(root):
    """ Extract all images under root """
    img_ext = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    root = os.path.expanduser(root)
    img_paths = []
    for curdir, subdirs, files in os.walk(root):
        for file in files:
            if os.path.splitext(file)[1].lower() in img_ext:
                img_paths.append(os.path.join(curdir, file))
    img_paths = sorted(img_paths)
    return img_paths


class CelebAHQ(Dataset):
    """The CelebA-HQ Dataset.

    The CelebA-HQ dataset is a high-quality version of CelebA that consists of 30,000 images at 1024×1024 resolution.
    (Copied from PaperWithCode)

    The official way to prepare the dataset is to download img_celeba.7z from the original CelebA dataset and the delta
    files from the official GitHub repository. Then use dataset_tool.py to generate the high-quality images.

    However, I personally recommend downloading the CelebAMask-HQ dataset, which contains processed CelebA-HQ images.
    Nevertheless, the filenames in CelebAMask-HQ are sorted from 0 to 29999, which is inconsistent with the original
    CelebA filenames. I provide a python script (scripts/celebahq_map_filenames.py) to help convert the filenames.

    To load data with this class, the dataset should be organized in the following structure:

    root
    ├── CelebA-HQ-img
    │   ├── 000004.jpg
    │   ├── ...
    │   └── 202591.jpg
    └── CelebA-HQ-to-CelebA-mapping.txt

    The train/valid/test sets are split according to the original CelebA dataset,
    resulting in 24,183 training images, 2,993 validation images, and 2,824 test images.

    This class has one pre-defined transform:
      - 'resize' (default): Resize the image directly to the target size

    References:
      - https://github.com/tkarras/progressive_growing_of_gans
      - https://paperswithcode.com/dataset/celeba-hq
      - https://github.com/switchablenorms/CelebAMask-HQ

    """
    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
            transform_type: str = 'default',
            transform: Optional[Callable] = None,
    ):
        if split not in ['train', 'valid', 'test', 'all']:
            raise ValueError(f'Invalid split: {split}')
        root = os.path.expanduser(root)
        image_root = os.path.join(root, 'CelebA-HQ-img')
        if not os.path.isdir(image_root):
            raise ValueError(f'{image_root} is not an existing directory')

        self.root = root
        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        self.transform = transform
        if transform is None:
            self.transform = self.get_transform()

        def filter_func(p):
            if split == 'all':
                return True
            celeba_splits = [1, 162771, 182638, 202600]
            k = 0 if split == 'train' else (1 if split == 'valid' else 2)
            return celeba_splits[k] <= int(os.path.splitext(os.path.basename(p))[0]) < celeba_splits[k+1]

        self.img_paths = extract_images(image_root)
        self.img_paths = list(filter(filter_func, self.img_paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, item):
        X = Image.open(self.img_paths[item])
        if self.transform is not None:
            X = self.transform(X)
        return X

    def get_transform(self):
        flip_p = 0.5 if self.split in ['train', 'all'] else 0.0
        if self.transform_type in ['default', 'resize']:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size)),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none':
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform
