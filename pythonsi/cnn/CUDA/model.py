import numpy as np
import torch

from . import util
from .operations import BatchNorm2d, Conv2d, LeakyReLU, Linear, MaxPool2d, ReLU


class CUDAModel:
    """PyTorch-CUDA affine propagation backend for CNNs."""

    def __init__(self, model, img_shape=None):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA CNN inference requested but CUDA is unavailable")
        self.layers = util.parse_model(model)
        self.img_shape = img_shape

    def forward(self, a, b, z):
        a_np, b_np = np.asarray(a), np.asarray(b)
        if a_np.shape != b_np.shape:
            raise ValueError("a and b must have the same shape")
        if a_np.ndim == 2:
            if self.img_shape is None:
                raise ValueError("img_shape is required for flat CNN input")
            a_np = a_np.reshape(a_np.shape[0], *self.img_shape)
            b_np = b_np.reshape(b_np.shape[0], *self.img_shape)
        if a_np.ndim != 4:
            raise ValueError(
                "CNN input must be shaped (batch, channels, height, width)"
            )

        device = torch.device("cuda")
        a = torch.as_tensor(a_np, dtype=torch.float32, device=device)
        b = torch.as_tensor(b_np, dtype=torch.float32, device=device)
        z = torch.as_tensor(z, dtype=torch.float32, device=device)
        itv = torch.tensor([-float("inf"), float("inf")], device=device)

        for name, params in self.layers:
            if name == "Conv2d":
                a, b = Conv2d(a, b, params)
            elif name == "BatchNorm2d":
                a, b = BatchNorm2d(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z, itv)
            elif name == "LeakyReLU":
                a, b, itv = LeakyReLU(a, b, z, itv, params)
            elif name == "MaxPool2d":
                a, b, itv = MaxPool2d(a, b, z, itv, params)
            elif name == "Linear":
                if a.ndim > 2:
                    a = a.reshape(a.shape[0], -1)
                    b = b.reshape(b.shape[0], -1)
                a, b = Linear(a, b, params)
            else:
                raise TypeError(f"Unsupported parsed CNN layer: {name}")
            if itv is None:
                return (
                    a.detach().cpu().numpy(),
                    b.detach().cpu().numpy(),
                    [np.nan, np.nan],
                )

        return (
            a.detach().cpu().numpy(),
            b.detach().cpu().numpy(),
            [itv[0].item(), itv[1].item()],
        )
