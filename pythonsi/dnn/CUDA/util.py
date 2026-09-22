import torch
from torch import nn


def parse_torch_model(model: nn.Module):
    """
    Flatten model into a list of (layer_type, tensor_or_None)
    e.g. [("Weight", tensor), ("Bias", tensor), ("ReLU", None), ...]

    Args:
        model: PyTorch nn.Module
        to_numpy: If True, convert tensors to numpy arrays
    """
    results = []

    for module in model.modules():
        if isinstance(module, nn.Sequential) or module is model:
            continue  # skip containers

        if isinstance(module, nn.BatchNorm1d):
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA was requested but is not available")
            running_mean = module.running_mean.detach().to("cuda", dtype=torch.float32)
            running_var = module.running_var.detach().to("cuda", dtype=torch.float32)
            if module.affine:
                gamma = module.weight.detach().to("cuda", dtype=torch.float32)
                beta = module.bias.detach().to("cuda", dtype=torch.float32)
            else:
                gamma = torch.ones(module.num_features, device="cuda")
                beta = torch.zeros(module.num_features, device="cuda")
            scale = gamma / torch.sqrt(running_var + module.eps)
            shift = beta - running_mean * scale
            results.append(("BatchNorm1d", {"scale": scale, "shift": shift}))
            continue

        if isinstance(module, nn.Linear):
            w = module.weight.detach().T.to("cuda", dtype=torch.float32)
            # Preserve the legacy Linear operation: represent bias=False as a
            # zero bias tensor rather than changing that operation's contract.
            b = (
                module.bias.detach().to("cuda", dtype=torch.float32)
                if module.bias is not None
                else torch.zeros(module.out_features, device="cuda")
            )
            results.append(("Linear", (w, b)))
            continue

        if isinstance(module, nn.LeakyReLU):
            results.append(("LeakyReLU", float(module.negative_slope)))
            continue

        if isinstance(module, nn.ReLU):
            results.append(("ReLU", None))
            continue

        # Unsupported parameterized modules must not be silently ignored.
        if hasattr(module, "weight") or hasattr(module, "bias"):
            raise TypeError(f"Unsupported CUDA DNN layer: {type(module).__name__}")
        if isinstance(module, (nn.Identity, nn.Flatten)):
            continue
        if not any(p.requires_grad for p in module.parameters()):
            raise TypeError(f"Unsupported CUDA DNN layer: {type(module).__name__}")
    return results


def is_torch_model(model):
    return isinstance(model, nn.Module)


def parse_model(model):
    if is_torch_model(model):
        return parse_torch_model(model)
    else:
        raise TypeError(f"Unsupported model type: {type(model)}")


# if __name__ == "__main__":
#     model = torch.nn.Sequential(
#         torch.nn.Linear(10, 5),
#         torch.nn.ReLU(),
#         torch.nn.Linear(5, 2)
#     )
#     model.eval()
#     layers = parse_network(model)
#     for name, params in layers:
#         if name == "Linear":
#             w, b = params
#             print(f"{name}: weight shape {w.shape}, bias shape {b.shape}")
