import torch

from lib_vision.models.metrics.representations.similarity.cka.cka_minibatch import (
    MinibatchCKA,
)


torch.manual_seed(58201)


def approx_equal(x: float, y: float) -> bool:
    epsilon = 0.001
    return x - epsilon <= y <= x + epsilon


def test_same_activations():
    cka_aggregator = MinibatchCKA()
    for _ in range(5):
        X = torch.rand((10, 5))
        cka_aggregator.update(X, X)

    # assert approx_equal(float(cka_aggregator.compute().item()), 1.0)
    assert approx_equal(float(cka_aggregator.compute()), 1.0)


def test_different_activations():
    cka_aggregator = MinibatchCKA()
    for _ in range(5):
        X = torch.rand((5, 8))
        Y = torch.rand((5, 10))
        cka_aggregator.update(X, Y)

    assert cka_aggregator.compute() < 1.0


def test_reset_different():
    torch.manual_seed(2849)
    cka_aggregator = MinibatchCKA()

    for _ in range(4):
        X = torch.rand((5, 8))
        Y = torch.rand((5, 10))
        cka_aggregator.update(X, Y)
    sim1 = cka_aggregator.compute()
    cka_aggregator.reset()

    for _ in range(8):
        X = torch.rand((12, 8))
        Y = torch.rand((12, 8))
        cka_aggregator.update(X, Y)
    sim2 = cka_aggregator.compute()
    cka_aggregator.reset()

    assert abs(sim1 - sim2) > 0.1
