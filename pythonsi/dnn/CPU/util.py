import numpy as np
import torch
from torch import nn


def parse_torch_model(model: nn.Module):
    """Parse a feed-forward PyTorch model for affine CPU inference.

    Supported layers are Linear, BatchNorm1d, ReLU, and LeakyReLU.  Containers
    are traversed by ``modules()`` and are not emitted as layers.
    """
    if not isinstance(model, nn.Module):
        raise TypeError(f"Unsupported model type: {type(model)}")

    results = []
    for module in model.modules():
        if module is model or isinstance(module, nn.Sequential):
            continue

        if isinstance(module, nn.Linear):
            weight = module.weight.detach().cpu().numpy().T
            bias = (
                module.bias.detach().cpu().numpy()
                if module.bias is not None
                else np.zeros(module.out_features, dtype=weight.dtype)
            )
            results.append(("Linear", (weight, bias)))
        elif isinstance(module, nn.BatchNorm1d):
            running_mean = module.running_mean.detach().cpu().numpy()
            running_var = module.running_var.detach().cpu().numpy()
            if module.affine:
                gamma = module.weight.detach().cpu().numpy()
                beta = module.bias.detach().cpu().numpy()
            else:
                gamma = np.ones(module.num_features, dtype=running_var.dtype)
                beta = np.zeros(module.num_features, dtype=running_var.dtype)
            scale = gamma / np.sqrt(running_var + module.eps)
            shift = beta - running_mean * scale
            results.append(("BatchNorm1d", {"scale": scale, "shift": shift}))
        elif isinstance(module, nn.ReLU):
            results.append(("ReLU", None))
        elif isinstance(module, nn.LeakyReLU):
            results.append(
                ("LeakyReLU", {"negative_slope": float(module.negative_slope)})
            )
        elif isinstance(module, (nn.Identity, nn.Flatten)):
            # The tabular target model does not use these, but they are safe
            # no-ops for a 2-D feed-forward input.
            continue
        else:
            raise TypeError(f"Unsupported DNN layer: {type(module).__name__}")

    return results


def is_torch_model(model):
    return isinstance(model, nn.Module)


def parse_model(model):
    if not is_torch_model(model):
        raise TypeError(f"Unsupported model type: {type(model)}")
    return parse_torch_model(model)
