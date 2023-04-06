from typing import List

import torch
import torch.nn as nn
from torch import Tensor


class TimestepMapping(nn.Module):
    def __init__(
            self,
            model: nn.Module,
            timesteps: List[int] or Tensor,
            timestep_arg_id: int = 1,
    ):
        super().__init__()
        self.model = model
        if isinstance(timesteps, Tensor):
            assert timesteps.ndim == 1
            timesteps = timesteps.tolist()
        self.timesteps = torch.tensor(list(sorted(set(timesteps))), dtype=torch.long)
        self.timestep_arg_id = timestep_arg_id

        self.register_buffer('timesteps', self.timesteps)

    def forward(self, *args, **kwargs):
        args = list(args)
        t = args[self.timestep_arg_id]
        args[self.timestep_arg_id] = self.timesteps[t]
        return self.model(*args, **kwargs)

    def state_dict(self, *args, **kwargs):
        return self.model.state_dict(*args, **kwargs)

    def load_state_dict(self, *args, **kwargs):
        self.model.load_state_dict(*args, **kwargs)
