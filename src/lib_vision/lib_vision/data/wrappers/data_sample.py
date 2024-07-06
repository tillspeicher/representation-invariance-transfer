from dataclasses import dataclass
from typing import Iterable, Optional, TypeVar

import torch


IntermediateRepresentations = dict[str, torch.Tensor]

T = TypeVar("T")  # torch.Tensor, Image


@dataclass
class DataSample(Iterable[T]):  # torch.Tensor]):
    input: T
    target: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        if isinstance(self.input, torch.Tensor):
            return len(self.input)
        else:
            raise ValueError("Cannot get the length of a single-item sample")

    # Primarily here to make pytorch-lightning's batch size inference happy
    def __iter__(self) -> Iterable[torch.Tensor]:
        if not isinstance(self.input, torch.Tensor):
            raise ValueError("Cannot iterate over a single-item sample")
        if self.target is None:
            return iter([self.input])
        else:
            return iter([self.input, self.target])
