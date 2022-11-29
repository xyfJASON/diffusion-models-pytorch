from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as T
import torchvision.datasets as dset

import datasets
from utils.dist import get_dist_info


def build_dataset(name, dataroot, img_size, split, transforms=None, subset_ids=None):
    if name.lower() == 'mnist':
        assert split in ['train', 'test'], f'{name} only has train/test split, get {split}.'
        if transforms is None:
            transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        dataset = dset.MNIST(root=dataroot, train=(split == 'train'), transform=transforms)
        img_channels = 1

    elif name.lower() in ['fashion_mnist', 'fashion-mnist']:
        assert split in ['train', 'test'], f'{name} only has train/test split, get {split}.'
        if transforms is None:
            transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5], std=[0.5])
            ])
        dataset = dset.FashionMNIST(root=dataroot, train=(split == 'train'), transform=transforms)
        img_channels = 1

    elif name.lower() in ['cifar10', 'cifar-10']:
        assert split in ['train', 'test'], f'{name} only has train/test split, get {split}.'
        if transforms is None:
            transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.RandomHorizontalFlip(),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        dataset = dset.CIFAR10(root=dataroot, train=(split == 'train'), transform=transforms)
        img_channels = 3

    elif name.lower() == 'celeba':
        if transforms is None:
            transforms = T.Compose([
                T.CenterCrop((140, 140)),
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        dataset = dset.CelebA(root=dataroot, split=split, transform=transforms)
        img_channels = 3

    elif name.lower() == 'celeba-hq':
        if transforms is None:
            transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        dataset = datasets.CelebA_HQ(root=dataroot, split=split, transform=transforms)
        img_channels = 3

    elif name.lower() == 'image-dir':
        if transforms is None:
            transforms = T.Compose([
                T.Resize((img_size, img_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.5]*3, std=[0.5]*3)
            ])
        dataset = datasets.ImageDir(root=dataroot, split=split, transform=transforms)
        img_channels = 3

    else:
        raise ValueError(f"Dataset {name} is not supported.")

    if subset_ids is not None and len(subset_ids) < len(dataset):
        dataset = Subset(dataset, subset_ids)

    return dataset, img_channels


def build_dataloader(dataset, batch_size, shuffle=False, num_workers=0, pin_memory=False, prefetch_factor=2):
    if get_dist_info()['is_dist']:
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        dataloader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers,
                                pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    else:
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                                pin_memory=pin_memory, prefetch_factor=prefetch_factor)
    return dataloader
