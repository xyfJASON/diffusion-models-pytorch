import torchvision.transforms as T
import torchvision.datasets as dset


CIFAR10 = dset.CIFAR10


def get_default_transforms(img_size: int, split: str):
    flip_p = 0.5 if split == 'train' else 0.0
    transforms = T.Compose([
        T.Resize((img_size, img_size)),
        T.RandomHorizontalFlip(flip_p),
        T.ToTensor(),
        T.Normalize([0.5] * 3, [0.5] * 3),
    ])
    return transforms
