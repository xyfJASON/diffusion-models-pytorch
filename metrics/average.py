import torch
from torchmetrics import Metric


class AverageMeter(Metric):

    full_state_update = False

    def __init__(self):
        super().__init__()
        self.add_state('sum', default=torch.tensor(0.0), dist_reduce_fx='sum')
        self.add_state('total', default=torch.tensor(0), dist_reduce_fx='sum')

    def update(self, val: torch.Tensor, n: int):
        self.sum += val * n
        self.total += n

    def compute(self):
        return self.sum / self.total
