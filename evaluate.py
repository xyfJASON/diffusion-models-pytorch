"""
python evaluate.py --dataset cifar10 \
                   --dataroot /data/CIFAR-10/ \
                   --img_size 32 \
                   --n_eval 50000 \
                   --fake_dir ./samples/ddpm-cifar10/random-fixedlarge-10steps
"""

import tqdm
import argparse

import torch
from torch.utils.data import Subset
import torchvision.transforms as T
from torchmetrics.image import FrechetInceptionDistance, InceptionScore

from datasets import ImageDir, build_dataset
from utils.logger import get_logger
from utils.dist import init_dist, get_dist_info
from utils.misc import dict2namespace, get_device


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='name of ground-truth dataset')
    parser.add_argument('--dataroot', type=str, help='path to dataroot')
    parser.add_argument('--img_size', type=int, default=32, help='size of images')
    parser.add_argument('--n_eval', type=int, default=50000, help='number of images to evaluate')
    parser.add_argument('--fake_dir', type=str, help='path to the directory containing fake images')
    args = parser.parse_args()
    return args


def evaluate(args):
    # initialize distributed mode, set device
    init_dist()
    dist_info = dict2namespace(get_dist_info())
    device = get_device(dist_info)
    print(f'Device: {device}')

    # define logger
    logger = get_logger()

    # ground-truth data
    test_set = build_dataset(
        dataset=args.dataset,
        dataroot=args.dataroot,
        img_size=args.img_size,
        split='train',
    )
    if args.n_eval < len(test_set):
        test_set = Subset(test_set, torch.randperm(len(test_set))[:args.n_eval])
        logger.info(f'Use a subset of test set, {args.n_eval}/{len(test_set)}')
    elif args.n_eval > len(test_set):
        logger.warning(f'Size of test set ({len(test_set)}) < n_eval ({args.n_eval}), ignore n_eval')

    # generated data
    fake_set = ImageDir(
        root=args.fake_dir,
        split='',
        transform=T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()]),
    )
    if args.n_eval != len(fake_set):
        logger.warning(f'Number of fake images ({len(fake_set)}) is not equal to n_eval ({args.n_eval})')

    # define metrics
    metric_fid = FrechetInceptionDistance().to(device)
    metric_iscore = InceptionScore().to(device)

    # start evaluating
    logger.info('Sample real images for FID')
    for real_img in tqdm.tqdm(test_set, desc='Sampling', ncols=120):
        real_img = real_img[0] if isinstance(real_img, (tuple, list)) else real_img
        real_img = real_img.unsqueeze(0).to(device)
        real_img = ((real_img + 1) / 2 * 255).to(dtype=torch.uint8)
        metric_fid.update(real_img, real=True)

    logger.info('Sample fake images for FID and IS')
    for fake_img in tqdm.tqdm(fake_set, desc='Sampling', ncols=120):
        fake_img = fake_img.unsqueeze(0).to(device)
        fake_img = (fake_img * 255).to(dtype=torch.uint8)
        metric_fid.update(fake_img, real=False)
        metric_iscore.update(fake_img)

    fid = metric_fid.compute().item()
    iscore = metric_iscore.compute()
    iscore = (iscore[0].item(), iscore[1].item())
    logger.info(f'fid: {fid}')
    logger.info(f'iscore: {iscore[0]} ({iscore[1]})')
    logger.info('End of evaluation')


def main():
    args = parse_args()
    evaluate(args)


if __name__ == '__main__':
    main()
