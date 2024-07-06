"""
Compute CKA in pyorch.

Based on https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment/blob/master/CKA.py
but with added * / (n - 1)**2 normalization in the HSIC computation.
That was missing compared to the paper
"""


from typing import Optional
import torch


def centering(K: torch.Tensor) -> torch.Tensor:
    n = K.shape[0]
    unit = torch.ones([n, n], device=K.device).float()
    I = torch.eye(n, device=K.device).float()
    H = I - unit / n

    unit, I = None, None
    torch.cuda.empty_cache()

    return torch.matmul(
        torch.matmul(H, K), H
    )  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
    # return np.dot(H, K)  # KH


def rbf(X: torch.Tensor, sigma: Optional[torch.Tensor] = None) -> torch.Tensor:
    GX = torch.matmul(X, X.T)
    KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
    if sigma is None:
        mdist = torch.median(KX[KX != 0])
        sigma = torch.sqrt(mdist)
    KX *= -0.5 / (sigma * sigma)
    KX = torch.exp(KX)
    return KX


def kernel_HSIC(
    X: torch.Tensor, Y: torch.Tensor, sigma: Optional[torch.Tensor]
) -> torch.Tensor:
    n = len(X)
    return (
        torch.sum(centering(rbf(X, sigma)) * centering(rbf(Y, sigma)))
        / (n - 1) ** 2
    )


def linear_HSIC(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    n = len(X)
    return (
        torch.sum(
            centering(torch.matmul(X, X.T)) * centering(torch.matmul(Y, Y.T))
        )
        / (n - 1) ** 2
    )


def linear_CKA(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return linear_HSIC(X.float(), Y.float()) / (
        torch.sqrt(linear_HSIC(X.float(), X.float()))
        * torch.sqrt(linear_HSIC(Y.float(), Y.float()))
    )


def kernel_CKA(
    X: torch.Tensor, Y: torch.Tensor, sigma: Optional[torch.Tensor] = None
) -> torch.Tensor:
    return kernel_HSIC(X.float(), Y.float(), sigma) / (
        torch.sqrt(kernel_HSIC(X.float(), X.float(), sigma))
        * torch.sqrt(kernel_HSIC(Y.float(), Y.float(), sigma))
    )


if __name__ == "__main__":
    X = torch.random.randn(100, 64)
    Y = torch.random.randn(100, 64)

    print("Linear CKA, between X and Y: {}".format(linear_CKA(X, Y)))
    print("Linear CKA, between X and X: {}".format(linear_CKA(X, X)))

    print("RBF Kernel CKA, between X and Y: {}".format(kernel_CKA(X, Y)))
    print("RBF Kernel CKA, between X and X: {}".format(kernel_CKA(X, X)))
