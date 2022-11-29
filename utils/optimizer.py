import inspect

import torch
import torch.optim as optim

from utils.logger import get_logger


def optimizer_to_device(optimizer: optim.Optimizer, device: torch.device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def build_optimizer(params, cfg: dict) -> optim.Optimizer:
    """
    Args:
        params: parameter groups of model
        cfg: configuration dictionary, whose keys include:
            - type: 'sgd', 'rmsprop', 'adam', etc.
            - lr: learning rate
            - weight_decay
            - ... (other parameters)
    """
    available_optimizers = [module_name for module_name in dir(optim) if not module_name.startswith('__')]
    match_optimizer = list(filter(lambda o: cfg['type'].lower() == o.lower(), available_optimizers))
    if len(match_optimizer) == 0:
        raise ValueError(f"{cfg['type']} is not an available optimizer")
    elif len(match_optimizer) > 1:
        raise RuntimeError(f"{cfg['type']} matches multiple optimizers in torch.optim")

    _optim = getattr(optim, match_optimizer[0])
    assert inspect.isclass(_optim) and issubclass(_optim, optim.Optimizer)
    cfg.pop('type')

    valid_parameters = list(inspect.signature(_optim).parameters)
    valid_cfg = {k: v for k, v in cfg.items() if k in valid_parameters}
    invalid_keys = set(cfg.keys()) - set(valid_cfg.keys())
    if len(invalid_keys) > 0:
        logger = get_logger()
        logger.warning(f"config keys {invalid_keys} are ignored for optimizer {_optim.__name__}")
    optimizer = _optim(params, **valid_cfg)

    return optimizer


def _test():
    model = torch.nn.Linear(20, 10)

    cfg = dict(type='sgd',
               lr=0.001,
               momentum=0.9,
               weight_decay=0.05,
               betas=(0.9, 0.999),
               nesterov=True)
    optimizer = build_optimizer(model.parameters(), cfg)
    print(optimizer)

    cfg['type'] = 'adam'
    optimizer = build_optimizer(model.parameters(), cfg)
    print(optimizer)


if __name__ == '__main__':
    _test()
