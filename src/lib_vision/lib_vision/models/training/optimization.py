from typing import Callable, Optional
from dataclasses import dataclass, field

import torch


def get_default_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(model.parameters(), lr=0.001)


@dataclass
class OptimizerConfig:
    optimizer: Callable[[torch.nn.Module], torch.optim.Optimizer] = field(
        default_factory=lambda: get_default_optimizer
    )
    lr_scheduler: Optional[
        Callable[[torch.optim.Optimizer], torch.optim.lr_scheduler._LRScheduler]
    ] = None
