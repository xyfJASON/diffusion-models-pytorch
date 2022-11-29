import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse

from engine import Tester


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('func', choices=['evaluate', 'sample', 'sample_denoise'], help='choose a test function')
    parser.add_argument('--cfg', default='./config_test.yml', help='path to test configuration file')
    args = parser.parse_args()

    tester = Tester(args.cfg)

    if args.func == 'evaluate':
        tester.evaluate()
    elif args.func == 'sample':
        tester.sample()
    elif args.func == 'sample_denoise':
        tester.sample_denoise()
    else:
        raise ValueError
