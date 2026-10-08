"""MLP encoder for tabular Deep SVDD anomaly detection.

Architecture: Linear → BatchNorm1d → LeakyReLU → Linear
All Linear layers use ``bias=False``.
BatchNorm1d uses ``affine=False`` (no learnable scale/shift).
"""

import torch
import torch.nn as nn


class MLP(nn.Module):
    """Tabular MLP encoder for Deep SVDD."""

    def __init__(
        self,
        n_features: int = 20,
        hidden_dim: int = 32,
        repdim: int = 8,
        negative_slope: float = 0.01,
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim, bias=False),
            nn.BatchNorm1d(hidden_dim, affine=False),
            nn.LeakyReLU(negative_slope),
            nn.Linear(hidden_dim, repdim, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
