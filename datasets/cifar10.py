import torchvision.transforms as T
import torchvision.datasets as dset


CIFAR10 = dset.CIFAR10


def get_default_transforms(img_size: int, split: str):
    if split == 'train':
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
    else:
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize([0.5] * 3, [0.5] * 3),
        ])
    return transforms
