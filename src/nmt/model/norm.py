import torch
import torch.nn as nn

from ..config import ModelConfig


class LayerNorm(nn.Module):
    """LayerNorm over the last dim: centers (subtract mean) and scales.
        y = gamma * (x - mean) / sqrt(var + eps) + beta     (mean/var over the last dim)
    """

    def __init__(self, dim: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes over the last dim only."""
        mean = x.mean(dim=-1, keepdim=True)
        var = x.var(dim=-1, keepdim=True, unbiased=False)
        normalize = (x - mean)/torch.sqrt(var + self.eps)
        affine = self.gamma * normalize + self.beta
        return affine


class RMSNorm(nn.Module):
    """RMSNorm: scale only. No centering or bias.
        y = gamma * x / sqrt(mean(x^2) + eps)               (mean over the last dim)
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        # Only a gain here (no beta, no mean subtraction):
        self.gamma = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (..., dim) -> same shape."""
        mean = (x**2).mean(dim=-1, keepdim=True)
        root = x/torch.sqrt(mean + self.eps)
        affine = self.gamma * root
        return affine


def make_norm(config: ModelConfig, dim: int) -> nn.Module:
    # Return a LayerNorm or RMSNorm sized to `dim`, chosen by config.norm_type.
    return {"layernorm": LayerNorm,
    "rmsnorm": RMSNorm}[config.norm_type](dim)


class ResidualNorm(nn.Module):
    def __init__(self, config: ModelConfig, dim: int) -> None:
        super().__init__()
        self.norm = make_norm(config, dim)
        self.dropout = nn.Dropout(config.dropout)
        self.pre = (config.norm_position == "pre")

    def forward(self, x: torch.Tensor, sublayer) -> torch.Tensor:
        # x: (B, T, dim).returns (B,T,dim).
        if self.pre:
            y = self.norm(x)
            y = sublayer(y)
            y = self.dropout(y)
            return x + y
        else:
            y = sublayer(x)
            y = self.dropout(y)
            return self.norm(x + y)
