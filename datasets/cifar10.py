from typing import Optional, Callable

import torchvision.datasets
import torchvision.transforms as T
from torch.utils.data import Dataset


class CIFAR10(Dataset):
    """Extend torchvision.datasets.CIFAR10 with one pre-defined transform.

    The pre-defined transform is:
      - 'resize' (default): Resize the image directly to the target size, followed by random horizontal flipping.

    """

    def __init__(
            self,
            root: str,
            img_size: int,
            split: str = 'train',
            transform_type: str = 'default',
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            download: bool = False,
    ):
        if split not in ['train', 'test']:
            raise ValueError(f'Invalid split: {split}')

        self.img_size = img_size
        self.split = split
        self.transform_type = transform_type
        if transform is None:
            transform = self.get_transform()

        self.cifar10 = torchvision.datasets.CIFAR10(
            root=root,
            train=(split == 'train'),
            transform=transform,
            target_transform=target_transform,
            download=download,
        )

    def __len__(self):
        return len(self.cifar10)

    def __getitem__(self, item):
        X, y = self.cifar10[item]
        return X, y

    def get_transform(self):
        flip_p = 0.5 if self.split == 'train' else 0.0
        if self.transform_type in ['default', 'resize']:
            transform = T.Compose([
                T.Resize((self.img_size, self.img_size), antialias=True),
                T.RandomHorizontalFlip(flip_p),
                T.ToTensor(),
                T.Normalize([0.5] * 3, [0.5] * 3),
            ])
        elif self.transform_type == 'none':
            transform = None
        else:
            raise ValueError(f'Invalid transform_type: {self.transform_type}')
        return transform
