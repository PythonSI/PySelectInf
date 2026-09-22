import numpy as np
import torch
from torch import nn


def parse_torch_model(model: nn.Module):
    """
    Flatten a CNN-based model into a list of (layer_type, params).

    Supported layers:
        - Conv2d: params = dict(weight, bias, padding, stride, dilation)
        - BatchNorm2d: params = dict(scale, shift) (pre-fused)
        - LeakyReLU: params = negative_slope (float)
        - MaxPool2d: params = dict(kernel_size, stride)
        - Linear: params = (weight.T, bias)
        - ReLU: params = None

    Args:
        model: PyTorch nn.Module
    """
    results = []

    for module in model.modules():
        if isinstance(module, nn.Sequential) or module is model:
            continue  # skip containers

        if isinstance(module, nn.Conv2d):
            w = module.weight.detach().cpu().numpy()
            b = module.bias.detach().cpu().numpy() if module.bias is not None else None
            results.append(
                (
                    "Conv2d",
                    {
                        "weight": w,
                        "bias": b,
                        "padding": module.padding[0]
                        if hasattr(module.padding, "__len__")
                        else module.padding,
                        "stride": module.stride[0]
                        if hasattr(module.stride, "__len__")
                        else module.stride,
                        "dilation": module.dilation[0]
                        if hasattr(module.dilation, "__len__")
                        else module.dilation,
                    },
                )
            )
        elif isinstance(module, nn.BatchNorm2d):
            # Pre-fuse BN into scale + shift for efficient inference
            rm = module.running_mean.detach().cpu().numpy()
            rv = module.running_var.detach().cpu().numpy()
            eps = module.eps
            gamma = (
                module.weight.detach().cpu().numpy()
                if module.affine
                else np.ones(module.num_features)
            )
            beta = (
                module.bias.detach().cpu().numpy()
                if module.affine
                else np.zeros(module.num_features)
            )
            scale = gamma / np.sqrt(rv + eps)
            shift = beta - rm * scale
            results.append(("BatchNorm2d", {"scale": scale, "shift": shift}))
        elif isinstance(module, nn.LeakyReLU):
            results.append(("LeakyReLU", module.negative_slope))
        elif isinstance(module, nn.MaxPool2d):
            ks = (
                module.kernel_size
                if isinstance(module.kernel_size, int)
                else module.kernel_size[0]
            )
            st = module.stride if module.stride is not None else ks
            if isinstance(st, (list, tuple)):
                st = st[0]
            results.append(("MaxPool2d", {"kernel_size": ks, "stride": st}))
        elif isinstance(module, nn.Linear):
            w = module.weight.detach().cpu().numpy().T
            b = module.bias.detach().cpu().numpy() if module.bias is not None else None
            results.append(("Linear", (w, b)))
        elif isinstance(module, nn.ReLU):
            results.append(("ReLU", None))
        else:
            # Skip unknown layers (Identity, Dropout, etc.)
            pass

    return results


def is_torch_model(model):
    return isinstance(model, nn.Module)


def parse_model(model):
    if is_torch_model(model):
        return parse_torch_model(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")
