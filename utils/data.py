import tqdm
from torch.utils.data import Subset
from utils.logger import get_logger


def check_split(name, split, strict_valid_test):
    available_split = {
        'mnist': ['train', 'test'],
        'cifar10': ['train', 'test'],
        'cifar-10': ['train', 'test'],
        'celebahq': ['train', 'valid', 'test', 'all'],
        'celeba-hq': ['train', 'valid', 'test', 'all'],
        'ffhq': ['train', 'test'],
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
            transforms = get_default_transforms(img_size, split)
        dataset = CelebA_HQ(root=dataroot, split=split, transform=transforms)

    elif name.lower() == 'ffhq':
        from datasets.ffhq import FFHQ, get_default_transforms
        if transforms is None:
            transforms = get_default_transforms(img_size, split)
        dataset = FFHQ(root=dataroot, split=split, transform=transforms)

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


def get_data_generator(dataloader, is_main_process=True, with_tqdm=True):
    disable = not (with_tqdm and is_main_process)
    while True:
        for batch in tqdm.tqdm(dataloader, disable=disable, desc='Epoch', leave=False):
            yield batch
