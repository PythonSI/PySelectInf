from .operations import BatchNorm1d, LeakyReLU, Linear, ReLU
from . import util
import torch
import numpy as np
import time


class CUDAModel:
    def __init__(self, model):
        self.layers = util.parse_model(model)

    def forward(self, a, b, z):
        a = torch.tensor(a, dtype=torch.float32, device="cuda")  # GPU tensor
        b = torch.tensor(b, dtype=torch.float32, device="cuda")  # GPU tensor
        itv = torch.tensor(
            [-float("inf"), float("inf")], dtype=torch.float32, device="cuda"
        )  # GPU interval
        z_gpu = torch.tensor(z, dtype=torch.float32, device="cuda")  # GPU scalar
        for name, params in self.layers:
            if name == "Linear":
                a, b = Linear(a, b, params)
            elif name == "BatchNorm1d":
                a, b = BatchNorm1d(a, b, params)
            elif name == "LeakyReLU":
                a, b, itv = LeakyReLU(a, b, z_gpu, itv, params)
            elif name == "ReLU":
                a, b, itv = ReLU(a, b, z_gpu, itv)
            else:
                raise TypeError(f"Unsupported parsed DNN layer: {name}")
        a = a.cpu().numpy()
        b = b.cpu().numpy()
        itv = [itv[0].cpu().item(), itv[1].cpu().item()]
        return a, b, itv
