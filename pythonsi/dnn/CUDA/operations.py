import torch


def Linear(a, b, params):
    stacked = torch.stack([a, b], dim=0)  # shape: (2, ...)
    result = torch.matmul(stacked, params[0])  # shape: (2, ...)
    result[0] += params[1]
    return result[0], result[1]


def relu_elementwise(a, b, z):
    X = a + b * z
    neg_mask = X <= 0
    b_nz = torch.abs(b) > 1e-12
    a_out = torch.where(neg_mask, torch.tensor(0.0, device=a.device), a)
    b_out = torch.where(neg_mask, torch.tensor(0.0, device=b.device), b)
    threshold = torch.where(b_nz, -a / b, torch.tensor(float("inf"), device=a.device))
    where_min = (neg_mask & (b > 0)) | (~neg_mask & (b < 0))
    where_max = (neg_mask & (b < 0)) | (~neg_mask & (b > 0))
    return a_out, b_out, threshold, where_min, where_max


def BatchNorm1d(a, b, params):
    scale = params["scale"]
    shift = params["shift"]
    return a * scale + shift, b * scale


def LeakyReLU(a, b, z, itv, negative_slope=0.01):
    x = a + b * z
    active = x >= 0
    a_out = torch.where(active, a, negative_slope * a)
    b_out = torch.where(active, b, negative_slope * b)
    # Keep the legacy CUDA interval convention: an infeasible region is NaN.
    threshold = torch.where(
        torch.abs(b) > 1e-12,
        -a / b,
        torch.full_like(a, float("inf")),
    )
    lower_mask = (active & (b > 0)) | ((~active) & (b < 0))
    upper_mask = (active & (b < 0)) | ((~active) & (b > 0))
    lower_values = torch.where(
        lower_mask, threshold, torch.full_like(threshold, float("-inf"))
    )
    upper_values = torch.where(
        upper_mask, threshold, torch.full_like(threshold, float("inf"))
    )
    new_lower = torch.maximum(itv[0], torch.max(lower_values))
    new_upper = torch.minimum(itv[1], torch.min(upper_values))
    out = torch.stack([new_lower, new_upper])
    out = torch.where(new_lower <= new_upper, out, torch.full_like(out, float("nan")))
    return a_out, b_out, out


def ReLU(a, b, z, itv):
    a_out, b_out, threshold, where_min, where_max = relu_elementwise(a, b, z)
    min_val_array = torch.where(
        where_min, threshold, torch.tensor(float("inf"), device=threshold.device)
    )
    min_val = torch.min(min_val_array)
    max_val_array = torch.where(
        where_max, threshold, torch.tensor(float("-inf"), device=threshold.device)
    )
    max_val = torch.max(max_val_array)
    new_upper = torch.minimum(itv[1], min_val)
    new_lower = torch.maximum(itv[0], max_val)
    itv_out = torch.stack([new_lower, new_upper])
    itv_out = torch.where(
        new_lower <= new_upper, itv_out, torch.full_like(itv_out, float("nan"))
    )
    return a_out, b_out, itv_out
