from torch.utils.data import DataLoader, Subset
from torch.utils.data.distributed import DistributedSampler
from utils.logger import get_logger
from utils.dist import is_dist_avail_and_initialized


def check_split(name, split, strict_valid_test):
    available_split = {
        'mnist': ['train', 'test'],
        'cifar10': ['train', 'test'],
        'cifar-10': ['train', 'test'],
        'celebahq': ['train', 'valid', 'test', 'all'],
        'celeba-hq': ['train', 'valid', 'test', 'all'],
        'imagenet': ['train', 'valid', 'test'],
    }
    assert split in ['train', 'valid', 'test', 'all']
    if split in ['train', 'all'] or strict_valid_test:
        assert split in available_split[name.lower()], f"Dataset {name} doesn't have split: {split}"
    elif split not in available_split[name.lower()]:
        replace_split = 'test' if split == 'valid' else 'valid'
        assert replace_split in available_split[name.lower()], f"Dataset {name} doesn't have split: {split}"
        logger = get_logger()
        logger.warning(f'Replace split `{split}` with split `{replace_split}`')
        split = replace_split
    return split


def get_dataset(name, dataroot, img_size, split, transforms=None, subset_ids=None, strict_valid_test=False):
    """
    Args:
        name: name of dataset
        dataroot: path to dataset
        img_size: size of images
        split: 'train', 'valid', 'test', 'all'
        transforms: if None, will use default transforms
        subset_ids: select a subset of the full dataset
        strict_valid_test: replace validation split with test split (or vice versa)
                           if the dataset doesn't have a validation / test split
    """
    split = check_split(name, split, strict_valid_test)

    if name.lower() == 'mnist':
        from datasets.mnist import MNIST, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size)
        dataset = MNIST(root=dataroot, train=(split == 'train'), transform=transforms)

    elif name.lower() in ['cifar10', 'cifar-10']:
        from datasets.cifar10 import CIFAR10, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size, split)
        dataset = CIFAR10(root=dataroot, train=(split == 'train'), transform=transforms)

    elif name.lower() in ['celeba-hq', 'celebahq']:
        from datasets.celeba_hq import CelebA_HQ, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size)
        dataset = CelebA_HQ(root=dataroot, split=split, transform=transforms)

    elif name.lower() == 'imagenet':
        from datasets.imagenet import ImageNet, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size, split)
        dataset = ImageNet(root=dataroot, split=split, transform=transforms)

    else:
        raise ValueError(f"Dataset {name} is not supported.")

    if subset_ids is not None and len(subset_ids) < len(dataset):
        dataset = Subset(dataset, subset_ids)
    return dataset


def get_dataloader(dataset,
                   batch_size,
                   shuffle=False,
                   num_workers=0,
                   collate_fn=None,
                   pin_memory=False,
                   drop_last=False,
                   prefetch_factor=2):
    if is_dist_avail_and_initialized():
        sampler = DistributedSampler(dataset, shuffle=shuffle)
        shuffle = None
    else:
        sampler = None
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
        drop_last=drop_last,
        prefetch_factor=prefetch_factor,
    )
    return dataloader


def get_data_generator(dataloader, start_epoch=0):
    ep = start_epoch
    while True:
        if hasattr(dataloader.sampler, 'set_epoch'):
            dataloader.sampler.set_epoch(ep)
        for batch in dataloader:
            yield batch
        ep += 1
