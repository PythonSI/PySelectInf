"""CNN encoder for Deep SVDD anomaly detection.

Ported from padi/image/network.py — the PatchNetwork encoder used for
patch-based Deep SVDD.
"""

import torch
import torch.nn as nn


class PatchNetwork(nn.Module):
    """Patch encoder: Conv2d → BatchNorm2d → LeakyReLU → MaxPool2d … → Linear.

    All Conv2d and Linear layers use ``bias=False``.
    BatchNorm2d uses ``affine=False`` (no learnable scale/shift).
    """

    def __init__(
        self,
        in_channels: int = 1,
        repdim: int = 32,
        img_size: tuple = (30, 30),
        channels: tuple = (16, 32, 64),
        n_pool_blocks: int = 2,
    ):
        super().__init__()
        self.repdim = repdim
        self.img_size = img_size
        self.channels = channels
        self.n_pool_blocks = int(n_pool_blocks)

        layers = []
        c_in = in_channels
        for i, c_out in enumerate(channels):
            layers.append(nn.Conv2d(c_in, c_out, kernel_size=3, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(c_out, eps=0.0001, affine=False))
            layers.append(nn.LeakyReLU(0.01, inplace=True))
            if i < self.n_pool_blocks:
                layers.append(nn.MaxPool2d(2, 2))
            c_in = c_out
        self.features = nn.Sequential(*layers)

        num_pools = min(self.n_pool_blocks, len(channels))
        final_h = img_size[0] // 2**num_pools
        final_w = img_size[1] // 2**num_pools
        if final_h < 1 or final_w < 1:
            raise ValueError(f"Patch too small. Input: {img_size}, Pools: {num_pools}.")
        self.fc = nn.Linear(c_in * final_h * final_w, repdim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
