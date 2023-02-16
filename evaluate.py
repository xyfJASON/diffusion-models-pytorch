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

from datasets import ImageDir
from utils.logger import get_logger
from utils.data import get_dataloader
from utils.misc import image_float_to_uint8
from utils.dist import init_distributed_mode, get_world_size


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_size', type=int, default=32, help='size of images')
    parser.add_argument('--n_eval', type=int, default=50000, help='number of images to evaluate')
    parser.add_argument('--batch_size', type=int, default=1, help='evaluate by batch size')
    parser.add_argument('--real_dir', type=str, help='path to the directory containing ground-truth images')
    parser.add_argument('--fake_dir', type=str, help='path to the directory containing fake images')
    return parser


def evaluate():
    metric_fid = FrechetInceptionDistance().to(device)
    metric_iscore = InceptionScore().to(device)

    logger.info('Sample real images for FID')
    for real_img in tqdm.tqdm(real_loader, desc='Sampling', ncols=120):
        real_img = real_img.to(device)
        real_img = image_float_to_uint8(real_img)
        metric_fid.update(real_img, real=True)

    logger.info('Sample fake images for FID and IS')
    for fake_img in tqdm.tqdm(fake_loader, desc='Sampling', ncols=120):
        fake_img = fake_img.to(device)
        fake_img = image_float_to_uint8(fake_img)
        metric_fid.update(fake_img, real=False)
        metric_iscore.update(fake_img)

    fid = metric_fid.compute().item()
    iscore = metric_iscore.compute()
    iscore = (iscore[0].item(), iscore[1].item())
    logger.info(f'fid: {fid}')
    logger.info(f'iscore: {iscore[0]} ({iscore[1]})')
    logger.info('End of evaluation')


if __name__ == '__main__':
    # PARSE ARGS
    args = get_parser().parse_args()

    # INITIALIZE DISTRIBUTED MODE
    init_distributed_mode()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # INITIALIZE LOGGER
    logger = get_logger()
    logger.info(f'Device: {device}')
    logger.info(f"Number of devices: {get_world_size()}")

    # BUILD DATASETS
    transform = T.Compose([
        T.Resize((args.img_size, args.img_size)),
        T.ToTensor(),
    ])
    real_set = ImageDir(root=args.real_dir, split='', transform=transform)
    fake_set = ImageDir(root=args.fake_dir, split='', transform=transform)

    if args.n_eval > len(real_set) or args.n_eval > len(fake_set):
        args.n_eval = min(len(real_set), len(fake_set))
        logger.warning(f'Change n_eval to {args.n_eval}')
    if args.n_eval < len(real_set):
        logger.info(f'Use a subset of ground-truth images, {args.n_eval}/{len(real_set)}')
        real_set = Subset(real_set, torch.randperm(len(real_set))[:args.n_eval])
    if args.n_eval < len(fake_set):
        logger.info(f'Use a subset of fake images, {args.n_eval}/{len(fake_set)}')
        fake_set = Subset(fake_set, torch.randperm(len(fake_set))[:args.n_eval])

    # BUILD DATALOADER
    real_loader = get_dataloader(real_set, batch_size=args.batch_size)
    fake_loader = get_dataloader(fake_set, batch_size=args.batch_size)

    # EVALUATE
    evaluate()
