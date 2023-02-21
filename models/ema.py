import torch.nn as nn


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999, gradual: bool = True):
        super().__init__()
        self.model = model
        self.decay = decay
        self.gradual = gradual
        self.num_updates = 0

        self.shadow, self.backup = {}, {}
        self.register()

    def register(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    def update(self):
        self.num_updates += 1
        decay = self.decay
        if self.gradual:
            decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates))
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_param = (1. - decay) * param.data + decay * self.shadow[name]
                self.shadow[name] = new_param.clone()

    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data
                param.data = self.shadow[name]

    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

    def state_dict(self):
        return dict(
            decay=self.decay,
            shadow=self.shadow,
            num_updates=self.num_updates
        )

    def load_state_dict(self, state_dict, device):
        self.decay = state_dict['decay']
        self.shadow = state_dict['shadow']
        self.num_updates = state_dict['num_updates']
        for k in self.shadow.keys():
            self.shadow[k] = self.shadow[k].to(device)


def _test():
    # initialize to 0
    model = nn.Sequential(nn.Linear(5, 1))
    for p in model[0].parameters():
        p.data.fill_(0)
    ema = EMA(model)
    print(model.state_dict())
    print(ema.state_dict()['shadow'])
    print(model.state_dict())
    print()

    # update the model to 1
    for p in model[0].parameters():
        p.data.fill_(1)
    ema.update()
    print(model.state_dict())
    print(ema.state_dict()['shadow'])
    print(model.state_dict())
    print()

    # update the model to 2
    for p in model[0].parameters():
        p.data.fill_(2)
    ema.update()
    print(model.state_dict())
    print(ema.state_dict()['shadow'])
    print(model.state_dict())
    print()

    # test saving and loading state_dict
    model.load_state_dict(ema.state_dict()['shadow'])
    print()
    print(model.state_dict())


if __name__ == '__main__':
    _test()
