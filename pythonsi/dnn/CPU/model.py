import numpy as np

from .operations import BatchNorm1d, LeakyReLU, Linear, ReLU
from . import util


class CPUModel:
    """CPU affine-propagation backend for feed-forward DNNs."""

    def __init__(self, model):
        self.layers = util.parse_model(model)

    def forward(self, a, b, z):
        a = np.asarray(a)
        b = np.asarray(b)
        if a.shape != b.shape:
            raise ValueError("a and b must have the same shape")
        if a.ndim != 2:
            raise ValueError(
                "DNN CPU inference expects arrays shaped (batch, features)"
            )

        itv = np.array([-np.inf, np.inf], dtype=float)
        for name, params in self.layers:
            if name == "Linear":
                a, b = Linear(a, b, params)
            elif name == "BatchNorm1d":
                a, b = BatchNorm1d(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z, itv)
            elif name == "LeakyReLU":
                a, b, itv = LeakyReLU(
                    a,
                    b,
                    z,
                    itv,
                    negative_slope=params["negative_slope"],
                )
            else:
                raise TypeError(f"Unsupported parsed DNN layer: {name}")
        return a, b, [itv[0], itv[1]]
