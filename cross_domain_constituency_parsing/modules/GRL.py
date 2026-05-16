from typing import Any
import torch


class Grl_func(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x: torch.Tensor, lambda_: float) -> torch.Tensor:
        ctx._lambda = lambda_
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Any:
        grad_input = -ctx._lambda * grad_output
        return grad_input, None


class GRL(torch.nn.Module):

    def __init__(self) -> None:
        super(GRL, self).__init__()
        self._lambda = 1.

    def set_lambda(self, lambda_: float) -> None:
        self._lambda = lambda_

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return Grl_func.apply(x, self._lambda)
