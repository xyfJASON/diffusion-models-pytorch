import argparse

import torch
from torchvision.utils import save_image

import models
from models.modules import UNet


@torch.no_grad()
def generate(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.model_path, map_location='cpu')
    model = UNet(args.img_channels, args.img_size, args.dim, args.dim_mults)
    model.load_state_dict(ckpt['model'])
    model.to(device=device)
    model.eval()
    DiffusionModel = models.DDPM(args.total_steps, args.beta_schedule_mode)

    if args.mode == 'random':
        X = DiffusionModel.sample(model, (64, args.img_channels, args.img_size, args.img_size))
        save_image(X[-1].cpu(), args.save_path, nrow=8, normalize=True, value_range=(-1, 1))
    elif args.mode == 'denoise':
        X = DiffusionModel.sample(model, (8, args.img_channels, args.img_size, args.img_size))
        result = [X[i].cpu() for i in range(0, len(X) - 1, (len(X) - 1 + 18) // 19)] + [X[-1].cpu()]
        result = torch.stack(result, dim=1).reshape(-1, args.img_channels, args.img_size, args.img_size)
        save_image(result, args.save_path, nrow=20, normalize=True, value_range=(-1, 1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to the saved model')
    parser.add_argument('--mode', choices=['random', 'denoise'], required=True, help='generation mode. Options: random, denoise')
    parser.add_argument('--save_path', type=str, required=True, help='path to save the generated result')
    parser.add_argument('--cpu', action='store_true', help='use cpu instead of cuda')
    # Model settings
    parser.add_argument('--img_channels', type=int, default=3, help='number of channels of output images')
    parser.add_argument('--img_size', type=int, default=32, help='size of output images')
    parser.add_argument('--dim', type=int, default=64, help='dim of the first stage in unet')
    parser.add_argument('--dim_mults', nargs='+', type=int, default=[1, 2, 4, 8], help='multiplier of dim in each stage')
    parser.add_argument('--total_steps', type=int, default=1000, help='total steps of diffusion model')
    parser.add_argument('--beta_schedule_mode', type=str, default='linear', help='beta schedule')
    args = parser.parse_args()

    generate(args)


if __name__ == '__main__':
    main()
