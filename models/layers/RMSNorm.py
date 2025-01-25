import torch.nn as nn
import torch


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def _norm(self, x: torch.Tensor):
        # (B, T, d_model) * (B, T, 1) = (B, T, d_model)
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x: torch.Tensor):
        # (d_model) * (B, T, d_model) = (B, T, d_model)
        return self.weight * self._norm(x.float()).type_as(x)
