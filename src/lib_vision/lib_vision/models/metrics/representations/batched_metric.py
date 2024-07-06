from typing import List

import torch

from . import cka_torch as cka


class BatchedMetric:
    """
    Computes minibatch-CKA aggregated over multiple batches of data.

    For each minibatch and its corresponding set of representations
    X_i, Y_i, call .add_minibatch(X_i, Y_i).
    To obtain the CKA-value aggregated over all .add_minibatch() calls
    since the last time .reset() was called, use .value().
    The aggregator state is resetted when calling .reset()
    or .value(reset=True),

    See tests/test_CKA_minibatch.py for usage examples.
    """

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.hsic_xy: List[torch.Tensor] = []
        self.hsic_xx: List[torch.Tensor] = []
        self.hsic_yy: List[torch.Tensor] = []

    def add_minibatch(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.hsic_xy.append(unbiased_linear_HSIC(X, Y).float())
        self.hsic_xx.append(unbiased_linear_HSIC(X, X).float())
        self.hsic_yy.append(unbiased_linear_HSIC(Y, Y).float())

    def value(self, reset: bool = False) -> torch.Tensor:
        """
        See https://arxiv.org/pdf/2010.15327.pdf equation 2
        """
        n_batches = len(self.hsic_xy)
        agg_hsic = sum(self.hsic_xy) / n_batches
        agg_var1 = torch.sqrt(torch.div(sum(self.hsic_xx), n_batches))
        agg_var2 = torch.sqrt(torch.div(sum(self.hsic_yy), n_batches))

        if reset:
            self.reset()
        return agg_hsic / (agg_var1 * agg_var2)
