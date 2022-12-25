import inspect

import torch
import torch.optim as optim

from utils.logger import get_logger


def optimizer_to_device(optimizer: optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def build_optimizer(params, config) -> optim.Optimizer:
    """
    Args:
        params: parameter groups of model
        config: argparse.Namespace with the following attributes:
            - type: 'sgd', 'rmsprop', 'adam', etc.
            - lr: learning rate
            - weight_decay (optional)
            - ... (other parameters)
    """
    available_optimizers = [module_name for module_name in dir(optim) if not module_name.startswith('__')]
    match_optimizer = list(filter(lambda o: config.type.lower() == o.lower(), available_optimizers))
    if len(match_optimizer) == 0:
        raise ValueError(f"{config.type} is not an available optimizer")
    elif len(match_optimizer) > 1:
        raise RuntimeError(f"{config.type} matches multiple optimizers in torch.optim")

    _optim = getattr(optim, match_optimizer[0])
    assert inspect.isclass(_optim) and issubclass(_optim, optim.Optimizer)
    config.__dict__.pop('type')

    valid_parameters = list(inspect.signature(_optim).parameters)
    valid_cfg = {k: v for k, v in config.__dict__.items() if k in valid_parameters}
    invalid_keys = set(config.__dict__.keys()) - set(valid_cfg.keys())
    if len(invalid_keys) > 0:
        logger = get_logger()
        logger.warning(f"config keys {invalid_keys} are ignored for optimizer {_optim.__name__}")
    optimizer = _optim(params, **valid_cfg)

    return optimizer


def _test():
    from utils.misc import dict2namespace

    model = torch.nn.Linear(20, 10)

    config = dict(type='sgd',
                  lr=0.001,
                  momentum=0.9,
                  weight_decay=0.05,
                  betas=(0.9, 0.999),
                  nesterov=True)
    config = dict2namespace(config)
    optimizer = build_optimizer(model.parameters(), config)
    print(optimizer)

    config.type = 'adam'
    optimizer = build_optimizer(model.parameters(), config)
    print(optimizer)


if __name__ == '__main__':
    _test()
