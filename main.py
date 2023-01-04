import yaml
import argparse

from utils.misc import dict2namespace
from runners import DDPMRunner, DDIMRunner


if __name__ == '__main__':
    # ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('model', choices=['ddpm', 'ddim'], help='choose a model')
    parser.add_argument('func', choices=['train',               # train
                                         'evaluate',            # evaluate
                                         'sample',              # sample images
                                         'sample_denoise',      # sample images with denoising process
                                         'sample_progressive',  # sample images with predicted X0 over time
                                         'sample_skip',         # sample images with fewer timesteps
                                         'sample_interpolate',  # interpolate between two images
                                         ], help='choose a function')
    parser.add_argument('-c', '--config', type=str, help='path to configuration file')
    args = parser.parse_args()

    # READ CONFIGURATION FILE
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    if args.model == 'ddpm':
        runner = DDPMRunner(args, config)
    elif args.model == 'ddim':
        runner = DDIMRunner(args, config)
    else:
        raise ValueError

    if not hasattr(runner, args.func):
        raise AttributeError(f"Runner of {args.model} doesn't have {args.func} method")

    if args.func == 'train':
        runner.train()
    elif args.func == 'evaluate':
        runner.evaluate()
    elif args.func == 'sample':
        runner.sample()
    elif args.func == 'sample_denoise':
        runner.sample_denoise()
    elif args.func == 'sample_progressive':
        runner.sample_progressive()
    elif args.func == 'sample_skip':
        runner.sample_skip()
    elif args.func == 'sample_interpolate':
        runner.sample_interpolate()
    else:
        raise ValueError
