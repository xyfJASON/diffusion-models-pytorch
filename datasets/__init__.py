from .CelebA_HQ import CelebA_HQ
from .ImageDir import ImageDir

import torchvision.transforms as T
import torchvision.datasets as dset


def build_dataset(dataset: str, dataroot: str, img_size: int, split: str):
    assert split in ['train', 'valid', 'test']

    if dataset.lower() == 'mnist':
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        if split == 'train':
            dataset = dset.MNIST(root=dataroot, train=True, transform=transforms)
        elif split == 'valid':
            dataset = None
        else:
            dataset = dset.MNIST(root=dataroot, train=False, transform=transforms)

    elif dataset.lower() in ['fashion_mnist', 'fashion-mnist']:
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5], std=[0.5])
        ])
        if split == 'train':
            dataset = dset.FashionMNIST(root=dataroot, train=True, transform=transforms)
        elif split == 'valid':
            dataset = None
        else:
            dataset = dset.FashionMNIST(root=dataroot, train=False, transform=transforms)

    elif dataset.lower() in ['cifar10', 'cifar-10']:
        train_transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        test_transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        if split == 'train':
            dataset = dset.CIFAR10(root=dataroot, train=True, transform=train_transforms)
        elif split == 'valid':
            dataset = None
        else:
            dataset = dset.CIFAR10(root=dataroot, train=False, transform=test_transforms)

    elif dataset.lower() == 'celeba':
        transforms = T.Compose([
            T.CenterCrop((140, 140)),
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        if split == 'train':
            dataset = dset.CelebA(root=dataroot, split='train', transform=transforms)
        elif split == 'valid':
            dataset = dset.CelebA(root=dataroot, split='valid', transform=transforms)
        else:
            dataset = dset.CelebA(root=dataroot, split='test', transform=transforms)

    elif dataset.lower() == 'celeba-hq':
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        if split == 'train':
            dataset = CelebA_HQ(root=dataroot, split='train', transform=transforms)
        elif split == 'valid':
            dataset = CelebA_HQ(root=dataroot, split='valid', transform=transforms)
        else:
            dataset = CelebA_HQ(root=dataroot, split='test', transform=transforms)

    elif dataset.lower() == 'image-dir':
        transforms = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.5]*3, std=[0.5]*3)
        ])
        if split == 'train':
            dataset = ImageDir(root=dataroot, split='train', transform=transforms)
        elif split == 'valid':
            dataset = ImageDir(root=dataroot, split='valid', transform=transforms)
        else:
            dataset = ImageDir(root=dataroot, split='test', transform=transforms)

    else:
        raise ValueError(f"Dataset {dataset} is not supported.")

    return dataset
