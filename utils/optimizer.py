import torch
import torch.optim as optim


def optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.to(device=device)


def build_optimizer(params, cfg):
    """
    Args:
        params: parameter groups of model
        cfg (dict): configuration dictionary, whose keys include:
            - choice: 'sgd', 'adam' or 'rmsprop'
            - sgd:
                - lr
                - weight_decay
                - momentum
                - nesterov
            - adam:
                - lr
                - betas
                - weight_decay
            - rmsprop:
                - lr
                - alpha
                - momentum
                - weight_decay
    """
    if cfg['choice'] == 'sgd':
        optimizer = optim.SGD(params,
                              lr=cfg['sgd']['lr'],
                              weight_decay=cfg['sgd'].get('weight_decay', 0),
                              momentum=cfg['sgd'].get('momentum', 0),
                              nesterov=cfg['sgd'].get('nesterov', False))

    elif cfg['choice'] == 'adam':
        optimizer = optim.Adam(params,
                               lr=cfg['adam']['lr'],
                               betas=cfg['adam'].get('betas', (0.9, 0.999)),
                               weight_decay=cfg['adam'].get('weight_decay', 0))

    elif cfg['choice'] == 'rmsprop':
        optimizer = optim.RMSprop(params,
                                  lr=cfg['rmsprop']['lr'],
                                  alpha=cfg['rmsprop'].get('alpha', 0.99),
                                  momentum=cfg['rmsprop'].get('momentum', 0),
                                  weight_decay=cfg['rmsprop'].get('weight_decay', 0))

    else:
        raise ValueError(f"Optimizer {cfg['choice']} is not supported.")

    return optimizer
