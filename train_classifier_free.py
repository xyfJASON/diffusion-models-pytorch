import argparse
from yacs.config import CfgNode as CN

from engine.classifier_free_trainer import ClassifierFreeTrainer


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-c', '--config',
        metavar='FILE',
        required=True,
        help='path to config file',
    )
    parser.add_argument(
        '-n', '--name',
        help='name of experiment directory, if None, use current time instead',
    )
    parser.add_argument(
        '-ni', '--no-interaction',
        action='store_true',
        help='do not interacting with the user',
    )
    parser.add_argument(
        '--opts',
        default=[],
        nargs=argparse.REMAINDER,
        help="modify config options using the command-line 'KEY VALUE' pairs",
    )
    return parser


if __name__ == '__main__':
    args = get_parser().parse_args()
    cfg = CN(new_allowed=True)
    cfg.merge_from_file(args.config)
    cfg.set_new_allowed(False)
    if args.opts is not None:
        cfg.merge_from_list(args.opts)
    cfg.freeze()

    trainer = ClassifierFreeTrainer(cfg, args)
    trainer.run_loop()
