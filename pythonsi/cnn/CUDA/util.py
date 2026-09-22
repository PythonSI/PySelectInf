import torch
from torch import nn


def _pair(value):
    if isinstance(value, int):
        return (value, value)
    return tuple(value)


def _to_cuda(tensor):
    return tensor.detach().to(device="cuda", dtype=torch.float32)


def parse_torch_model(model: nn.Module):
    """Parse supported CNN layers and move their parameters to CUDA."""
    if not isinstance(model, nn.Module):
        raise TypeError(f"Unsupported model type: {type(model)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA CNN inference requested but CUDA is unavailable")

    model.eval()
    layers = []
    for module in model.modules():
        if module is model or isinstance(module, nn.Sequential):
            continue
        if isinstance(module, nn.Conv2d):
            layers.append(
                (
                    "Conv2d",
                    {
                        "weight": _to_cuda(module.weight),
                        "bias": _to_cuda(module.bias)
                        if module.bias is not None
                        else None,
                        "stride": _pair(module.stride),
                        "padding": _pair(module.padding),
                        "dilation": _pair(module.dilation),
                    },
                )
            )
        elif isinstance(module, nn.BatchNorm2d):
            running_mean = _to_cuda(module.running_mean)
            running_var = _to_cuda(module.running_var)
            if module.affine:
                gamma = _to_cuda(module.weight)
                beta = _to_cuda(module.bias)
            else:
                gamma = torch.ones(module.num_features, device="cuda")
                beta = torch.zeros(module.num_features, device="cuda")
            scale = gamma / torch.sqrt(running_var + module.eps)
            shift = beta - running_mean * scale
            layers.append(("BatchNorm2d", {"scale": scale, "shift": shift}))
        elif isinstance(module, nn.ReLU):
            layers.append(("ReLU", None))
        elif isinstance(module, nn.LeakyReLU):
            layers.append(("LeakyReLU", float(module.negative_slope)))
        elif isinstance(module, nn.MaxPool2d):
            if module.ceil_mode or module.dilation != 1 and module.dilation != (1, 1):
                raise ValueError("ceil_mode and dilated MaxPool2d are unsupported")
            layers.append(
                (
                    "MaxPool2d",
                    {
                        "kernel_size": _pair(module.kernel_size),
                        "stride": _pair(
                            module.stride
                            if module.stride is not None
                            else module.kernel_size
                        ),
                        "padding": _pair(module.padding),
                    },
                )
            )
        elif isinstance(module, nn.Linear):
            weight = _to_cuda(module.weight).T.contiguous()
            bias = (
                _to_cuda(module.bias)
                if module.bias is not None
                else torch.zeros(module.out_features, device="cuda")
            )
            layers.append(("Linear", (weight, bias)))
        elif isinstance(module, (nn.Identity, nn.Flatten)):
            continue
        else:
            raise TypeError(f"Unsupported CNN layer: {type(module).__name__}")
    return layers


def parse_model(model):
    return parse_torch_model(model)
