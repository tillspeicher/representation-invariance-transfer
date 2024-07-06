from typing import Callable
import torch
from torchmetrics import Metric


def dot_product_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.sum(X * Y, -1).mean()


def cosine_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.cosine_similarity(X, Y, dim=-1).mean()


def l2_distance(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return torch.norm(X - Y, dim=1).mean()


SimilarityFunction = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]


class SimilarityMetric(Metric):
    def __init__(self, sim_func: SimilarityFunction) -> None:
        super().__init__()
        self.sim_func = sim_func
        self.similarity: torch.Tensor
        self.add_state(
            "similarity", default=torch.tensor(0.0), dist_reduce_fx="mean"
        )
        self.num_examples: int
        self.add_state(
            "num_examples", default=torch.tensor(0), dist_reduce_fx="sum"
        )

    def update(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        assert X.shape[0] == Y.shape[0]
        self.similarity += self.sim_func(X, Y) * X.shape[0]
        self.num_examples += X.shape[0]

    def compute(self) -> torch.Tensor:
        return self.similarity / self.num_examples
