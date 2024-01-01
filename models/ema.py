from typing import Iterable

import torch
import torch.nn as nn


class EMA:
    """Exponential moving average of model parameters.

    References:
        - https://github.com/huggingface/diffusers/blob/main/src/diffusers/training_utils.py#L76
        - https://github.com/fadel/pytorch_ema/blob/master/torch_ema/ema.py#L14
        - https://github.com/CompVis/stable-diffusion/blob/main/ldm/modules/ema.py#L5
        - https://github.com/lucidrains/ema-pytorch

    """
    def __init__(
            self,
            parameters: Iterable[nn.Parameter],
            decay: float = 0.9999,
            gradual: bool = True,
    ):
        """
        Args:
            parameters: Iterable of parameters, typically from `model.parameters()`.
            decay: The decay factor for exponential moving average.
            gradual: Whether to a gradually increasing decay factor.

        """
        super().__init__()
        self.decay = decay
        self.gradual = gradual

        self.num_updates = 0
        self.shadow = [param.detach().clone() for param in parameters]
        self.backup = []

    def get_decay(self):
        if self.gradual:
            return min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        else:
            return self.decay

    @torch.no_grad()
    def update(self, parameters: Iterable[nn.Parameter]):
        self.num_updates += 1
        decay = self.get_decay()
        for s_param, param in zip(self.shadow, parameters):
            if param.requires_grad:
                s_param.sub_((1. - decay) * (s_param - param))
            else:
                s_param.copy_(param)

    def apply_shadow(self, parameters: Iterable[nn.Parameter]):
        assert len(self.backup) == 0, 'backup is not empty'
        for s_param, param in zip(self.shadow, parameters):
            self.backup.append(param.detach().cpu().clone())
            param.data.copy_(s_param.data)

    def restore(self, parameters: Iterable[nn.Parameter]):
        assert len(self.backup) > 0, 'backup is empty'
        for b_param, param in zip(self.backup, parameters):
            param.data.copy_(b_param.to(param.device).data)
        self.backup = []

    def state_dict(self):
        return dict(
            decay=self.decay,
            shadow=self.shadow,
            num_updates=self.num_updates,
        )

    def load_state_dict(self, state_dict):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.num_updates = state_dict['num_updates']

    def to(self, device):
        self.shadow = [s_param.to(device) for s_param in self.shadow]


def _test():
    # initialize to 0
    model = nn.Sequential(nn.Linear(5, 1))
    for p in model[0].parameters():
        p.data.fill_(0)
    ema = EMA(model.parameters(), decay=0.9, gradual=False)
    print(model.state_dict())           # 0
    print(ema.state_dict()['shadow'])   # 0
    print()

    # update the model to 1
    for p in model[0].parameters():
        p.data.fill_(1)
    ema.update(model.parameters())
    print(model.state_dict())           # 1
    print(ema.state_dict()['shadow'])   # 0.9 * 0 + 0.1 * 1 = 0.1
    print()

    # update the model to 2
    for p in model[0].parameters():
        p.data.fill_(2)
    ema.update(model.parameters())
    print(model.state_dict())           # 2
    print(ema.state_dict()['shadow'])   # 0.9 * 0.1 + 0.1 * 2 = 0.29
    print()

    # apply shadow
    ema.apply_shadow(model.parameters())
    print(model.state_dict())           # 0.29
    print(ema.state_dict()['shadow'])   # 0.29
    print()

    # restore
    ema.restore(model.parameters())
    print(model.state_dict())           # 2
    print(ema.state_dict()['shadow'])   # 0.29


if __name__ == '__main__':
    _test()
