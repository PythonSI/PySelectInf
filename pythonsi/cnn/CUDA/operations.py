import torch
import torch.nn.functional as F

_TOLERANCE = 1e-12


def Linear(a, b, params):
    weight, bias = params
    a_out = a @ weight
    b_out = b @ weight
    return a_out + bias, b_out


def Conv2d(a, b, params):
    weight = params["weight"]
    bias = params["bias"]
    stride = params["stride"]
    padding = params["padding"]
    dilation = params["dilation"]
    a_out = F.conv2d(a, weight, bias, stride, padding, dilation)
    b_out = F.conv2d(b, weight, None, stride, padding, dilation)
    return a_out, b_out


def BatchNorm2d(a, b, params):
    scale = params["scale"].reshape(1, -1, 1, 1)
    shift = params["shift"].reshape(1, -1, 1, 1)
    return a * scale + shift, b * scale


def _interval_geq(itv, a, b):
    zero = torch.abs(b) <= _TOLERANCE
    if bool(torch.any(zero & (a < -_TOLERANCE))):
        return None
    out = itv.clone()
    positive = b > _TOLERANCE
    negative = b < -_TOLERANCE
    if bool(torch.any(positive)):
        out[0] = torch.maximum(out[0], torch.max(-a[positive] / b[positive]))
    if bool(torch.any(negative)):
        out[1] = torch.minimum(out[1], torch.min(-a[negative] / b[negative]))
    return out if bool(out[0] <= out[1]) else None


def _interval_leq(itv, a, b):
    zero = torch.abs(b) <= _TOLERANCE
    if bool(torch.any(zero & (a > _TOLERANCE))):
        return None
    out = itv.clone()
    positive = b > _TOLERANCE
    negative = b < -_TOLERANCE
    if bool(torch.any(positive)):
        out[1] = torch.minimum(out[1], torch.min(-a[positive] / b[positive]))
    if bool(torch.any(negative)):
        out[0] = torch.maximum(out[0], torch.max(-a[negative] / b[negative]))
    return out if bool(out[0] <= out[1]) else None


def ReLU(a, b, z, itv):
    x = a + b * z
    active = x >= 0
    a_out = torch.where(active, a, torch.zeros_like(a))
    b_out = torch.where(active, b, torch.zeros_like(b))
    flat_a, flat_b, flat_active = a.reshape(-1), b.reshape(-1), active.reshape(-1)
    out = _interval_geq(itv, flat_a[flat_active], flat_b[flat_active])
    if out is None and bool(torch.any(flat_active)):
        return a_out, b_out, None
    if out is None:
        out = itv
    out = _interval_leq(out, flat_a[~flat_active], flat_b[~flat_active])
    return a_out, b_out, out


def LeakyReLU(a, b, z, itv, negative_slope=0.01):
    x = a + b * z
    active = x >= 0
    a_out = torch.where(active, a, negative_slope * a)
    b_out = torch.where(active, b, negative_slope * b)
    flat_a, flat_b, flat_active = a.reshape(-1), b.reshape(-1), active.reshape(-1)
    out = _interval_geq(itv, flat_a[flat_active], flat_b[flat_active])
    if out is None and bool(torch.any(flat_active)):
        return a_out, b_out, None
    if out is None:
        out = itv
    out = _interval_leq(out, flat_a[~flat_active], flat_b[~flat_active])
    return a_out, b_out, out


def MaxPool2d(a, b, z, itv, params):
    kh, kw = params["kernel_size"]
    sh, sw = params["stride"]
    ph, pw = params["padding"]
    if kh != kw or sh != sw or ph != pw:
        raise ValueError("Only symmetric MaxPool2d parameters are supported")
    x = a + b * z
    x_windows = F.unfold(x, (kh, kw), padding=ph, stride=sh)
    a_windows = F.unfold(a, (kh, kw), padding=ph, stride=sh)
    b_windows = F.unfold(b, (kh, kw), padding=ph, stride=sh)
    batch, channels, locations = x.shape[0], x.shape[1], x_windows.shape[-1]
    area = kh * kw
    x_windows = x_windows.reshape(batch, channels, area, locations)
    a_windows = a_windows.reshape(batch, channels, area, locations)
    b_windows = b_windows.reshape(batch, channels, area, locations)
    max_values, max_indices = torch.max(x_windows, dim=2)
    gather_index = max_indices.unsqueeze(2)
    a_out = torch.gather(a_windows, 2, gather_index).squeeze(2)
    b_out = torch.gather(b_windows, 2, gather_index).squeeze(2)
    max_a = a_out.unsqueeze(2)
    max_b = b_out.unsqueeze(2)
    diff_a = max_a - a_windows
    diff_b = max_b - b_windows
    not_max = torch.ones_like(x_windows, dtype=torch.bool)
    not_max.scatter_(2, gather_index, False)
    diff_a = diff_a[not_max]
    diff_b = diff_b[not_max]
    out = _interval_geq(itv, diff_a, diff_b)
    if out is None:
        return a_out, b_out, None
    return a_out, b_out, out
