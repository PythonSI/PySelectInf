from .operations import (
    Linear,
    ReLU,
    Conv2d,
    BatchNorm2d,
    LeakyReLU,
    MaxPool2d,
)
from . import util
import numpy as np


class CPUModel:
    """CPU-based inference model for CNN architectures.

    Propagates parametric data (a, b) through a CNN, computing
    ``data(z) = a + b * z`` at each layer while collecting
    feasibility constraints on z.
    """

    def __init__(self, model, img_shape=None):
        self.layers = util.parse_model(model)
        self.img_shape = img_shape

    def forward(self, a, b, z):
        """Propagate (a, b) through all layers.

        Args:
            a: intercept array, shape (n, d) — flat input from x_node.inference()
            b: coefficient array, same shape as a
            z: current parameter value (float)

        Returns:
            a: output intercept after all layers
            b: output coefficient after all layers
            itv: feasibility interval [lower, upper]
        """
        a = np.asarray(a)
        b = np.asarray(b)
        itv = np.array([-np.inf, np.inf])

        # Reshape flat (n, d) → (n, C, H, W) if img_shape is known
        if self.img_shape is not None and a.ndim == 2:
            n = a.shape[0]
            a = a.reshape(n, *self.img_shape)
            b = b.reshape(n, *self.img_shape)

        # Convert NCHW → NHWC for spatial layers processing
        if a.ndim == 4:
            a = np.transpose(a, (0, 2, 3, 1))
            b = np.transpose(b, (0, 2, 3, 1))

        for name, params in self.layers:
            if name == "Linear":
                # Flatten if coming from conv layers
                if a.ndim > 2:
                    a = a.transpose(0, 3, 1, 2).reshape(a.shape[0], -1)
                    b = b.transpose(0, 3, 1, 2).reshape(b.shape[0], -1)
                a, b = Linear(a, b, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z, itv)
            elif name == "Conv2d":
                a, b = Conv2d(a, b, params)
            elif name == "BatchNorm2d":
                a, b = BatchNorm2d(a, b, params)
            elif name == "LeakyReLU":
                slope = float(params)
                a, b, itv = LeakyReLU(a, b, z, itv, negative_slope=slope)
            elif name == "MaxPool2d":
                ks = params["kernel_size"]
                st = params.get("stride", ks)
                a, b, itv = MaxPool2d(a, b, z, itv, kernel_size=ks, stride=st)

        return a, b, [itv[0], itv[1]]
