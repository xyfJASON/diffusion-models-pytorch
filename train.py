import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import argparse

from engine import Trainer


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='./config.yml', help='path to training configuration file')
    args = parser.parse_args()

    trainer = Trainer(args.cfg)
    trainer.run()
