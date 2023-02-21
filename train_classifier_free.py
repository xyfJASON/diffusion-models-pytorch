import yaml
import argparse

from utils.misc import add_args
from engine.classifier_free_trainer import ClassifierFreeTrainer


def parse_args():
    parser = argparse.ArgumentParser()
    # data configuration file
    parser.add_argument(
        '--config_data', type=str, required=True,
        help='Path to data configuration file',
    )
    # model configuration file
    parser.add_argument(
        '--config_model', type=str, required=True,
        help='Path to model configuration file',
    )
    # diffusion configuration file
    parser.add_argument(
        '--config_diffusion', type=str, required=True,
        help='Path to diffusion configuration file',
    )
    # arguments related to training
    parser.add_argument(
        '--name', type=str, default=None,
        help='Name of experiment directory. If None, use current time instead',
    )
    parser.add_argument(
        '--no_interaction', action='store_true', default=False,
        help='Do not interact with the user (always choose yes when interacting)',
    )
    parser.add_argument(
        '--seed', type=int, default=2022,
        help='Set random seed',
    )
    parser.add_argument(
        '--num_workers', type=int, default=4,
        help='Argument num_workers in torch.utils.data.DataLoader',
    )
    parser.add_argument(
        '--pin_memory', type=bool, default=True,
        help='Argument pin_memory in torch.utils.data.DataLoader',
    )
    parser.add_argument(
        '--prefetch_factor', type=int, default=2,
        help='Argument prefetch_factor in torch.utils.data.DataLoader',
    )
    parser.add_argument(
        '--batch_size', type=int, default=128,
        help='Batch size',
    )
    parser.add_argument(
        '--micro_batch', type=int, default=0,
        help='In case the GPU memory is too small, split a batch into micro batches. '
             'The gradients of micro batches will be aggregated for an update step',
    )
    parser.add_argument(
        '--weights', type=str, default=None,
        help='Path to pretrained weights',
    )
    parser.add_argument(
        '--resume', type=str, default=None,
        help='Path to the checkpoint to be resumed from',
    )
    parser.add_argument(
        '--train_steps', type=int, default=800000,
        help='Number of steps to train',
    )
    parser.add_argument(
        '--print_freq', type=int, default=200,
        help='Frequency of printing status, in steps',
    )
    parser.add_argument(
        '--sample_freq', type=int, default=1000,
        help='Frequency of sampling images, in steps',
    )
    parser.add_argument(
        '--save_freq', type=int, default=10000,
        help='Frequency of saving checkpoints, in steps',
    )
    parser.add_argument(
        '--ema_decay', type=float, default=0.9999,
        help='EMA decay rate',
    )
    parser.add_argument(
        '--ema_gradual', type=bool, default=True,
        help='Whether to gradually increase EMA decay rate',
    )
    parser.add_argument(
        '--p_uncond', type=float, default=0.2,
        help='Probability of setting condition to None during training',
    )
    parser.add_argument(
        '--optim_type', type=str, default='adamw',
        help='Choose an optimizer',
    )
    parser.add_argument(
        '--lr', type=float, default=0.0002,
        help='Learning rate',
    )

    tmp_args = parser.parse_known_args()[0]
    # merge data configurations
    with open(tmp_args.config_data, 'r') as f:
        config_data = yaml.safe_load(f)
    add_args(parser, config_data, prefix='data_')
    # merge model configurations
    with open(tmp_args.config_model, 'r') as f:
        config_model = yaml.safe_load(f)
    add_args(parser, config_model, prefix='model_')
    # merge diffusion configurations
    with open(tmp_args.config_diffusion, 'r') as f:
        config_diffusion = yaml.safe_load(f)
    add_args(parser, config_diffusion, prefix='diffusion_')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    trainer = ClassifierFreeTrainer(args)
    trainer.run_loop()
