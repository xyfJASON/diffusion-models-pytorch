import yaml
import argparse

from utils.misc import dict2namespace
from runners import Runner


if __name__ == '__main__':
    # ARGPARSE
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['train',               # train
                                         'evaluate',            # evaluate
                                         'sample',              # sample images
                                         'sample_denoise',      # sample images with denoising process
                                         'sample_skip',         # sample images with fewer timesteps
                                         ], help='choose a function')
    parser.add_argument('-c', '--config', type=str, help='path to configuration file')
    args = parser.parse_args()

    # READ CONFIGURATION FILE
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    config = dict2namespace(config)

    runner = Runner(args, config)
    if args.func == 'train':
        runner.train()
    elif args.func == 'evaluate':
        runner.evaluate()
    elif args.func == 'sample':
        runner.sample()
    elif args.func == 'sample_denoise':
        runner.sample_denoise()
    elif args.func == 'sample_skip':
        runner.sample_skip()
    else:
        raise ValueError
