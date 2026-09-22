import numpy as np
from pythonsi.dnn.CPU.operations import ReLU  # Reuse existing implementation
from pythonsi.dnn.CPU.operations import Linear as _Linear


def Linear(a, b, params):
    """
    Wrapper around dnn.Linear that safely handles bias=None.

    The original dnn.Linear always adds bias, which crashes when
    bias is None (e.g., nn.Linear(bias=False) in Deep SVDD).
    """
    w, bias = params
    if bias is None:
        bias = np.zeros(w.shape[1], dtype=a.dtype)
    return _Linear(a, b, (w, bias))


def Conv2d(a, b, params):
    """
    Propagate parametric (a, b) through a Conv2d layer.

    Since Conv2d is a linear operation, bias only affects `a`:
        out(z) = conv(a, W) + bias + conv(b, W) * z

    Assumes dilation=1 (standard for Deep SVDD).

    Args:
        a: intercept array, shape (batch, H, W, C_in) in NHWC
        b: coefficient array, same shape as a
        params: dict with keys 'weight', 'bias', 'padding', 'stride'
    """
    weight = params["weight"]
    bias = params.get("bias", None)
    padding = params.get("padding", 0)
    stride = params.get("stride", 1)

    if isinstance(padding, (list, tuple)):
        padding = padding[0]
    if isinstance(stride, (list, tuple)):
        stride = stride[0]

    def _conv2d_numpy(x, w, pad, s):
        """NumPy Conv2d forward in NHWC format (dilation=1)."""
        batch, h_in, w_in, c_in = x.shape
        c_out, _, kh, kw = w.shape
        if pad > 0:
            x = np.pad(
                x,
                ((0, 0), (pad, pad), (pad, pad), (0, 0)),
                mode="constant",
            )
        h_out = (x.shape[1] - kh) // s + 1
        w_out = (x.shape[2] - kw) // s + 1
        patches = np.lib.stride_tricks.sliding_window_view(x, (kh, kw), axis=(1, 2))
        if s > 1:
            patches = patches[:, ::s, ::s, :, :, :]
        patches = np.moveaxis(patches, 3, -1)
        patches_flat = patches.reshape(batch, h_out, w_out, -1)
        wf = w.transpose(0, 2, 3, 1).reshape(c_out, -1)
        return np.tensordot(patches_flat, wf, axes=([-1], [-1]))

    a_out = _conv2d_numpy(a, weight, padding, stride)
    b_out = _conv2d_numpy(b, weight, padding, stride)

    if bias is not None:
        a_out = a_out + bias.reshape(1, 1, 1, -1)

    return a_out, b_out


def BatchNorm2d(a, b, params):
    """
    Propagate parametric (a, b) through a fused BatchNorm2d layer.

    BN is pre-fused into scale and shift during parsing:
        scale = gamma / sqrt(running_var + eps)
        shift = beta - running_mean * scale

    So: out(z) = (a * scale + shift) + (b * scale) * z

    Args:
        a: intercept array, shape (batch, H, W, C) in NHWC
        b: coefficient array, same shape as a
        params: dict with keys 'scale', 'shift'
    """
    scale = params["scale"].reshape(1, 1, 1, -1)
    shift = params["shift"].reshape(1, 1, 1, -1)
    a_out = a * scale + shift
    b_out = b * scale
    return a_out, b_out


def LeakyReLU(a, b, z, itv, negative_slope=0.01):
    """
    Propagate parametric (a, b) through LeakyReLU with constraints.

    Where X >= 0: keep (a, b)
    Where X < 0:  multiply by negative_slope

    Constraints ensure the activation pattern does not change:
        X >= 0 => a + b*z >= 0
        X < 0  => a + b*z < 0

    Args:
        a: intercept array
        b: coefficient array
        z: current parameter value
        itv: current feasible interval [lower, upper]
        negative_slope: slope for negative region (default 0.01)
    """
    X = a + b * z

    activated = X >= 0
    not_activated = ~activated

    # Transform (a, b)
    a_out = np.where(activated, a, negative_slope * a)
    b_out = np.where(activated, b, negative_slope * b)

    # Compute constraints
    b_nonzero = np.abs(b) > 1e-12
    threshold = np.where(b_nonzero, -a / b, np.inf)

    b_positive = b > 0
    b_negative = b < 0

    itv_out = itv.copy()

    # Activated neurons (X >= 0): need a + b*z >= 0
    act_b_pos = activated & b_positive & b_nonzero
    act_b_neg = activated & b_negative & b_nonzero

    if np.any(act_b_pos):
        itv_out[0] = max(itv_out[0], np.max(threshold[act_b_pos]))
    if np.any(act_b_neg):
        itv_out[1] = min(itv_out[1], np.min(threshold[act_b_neg]))

    # Not-activated neurons (X < 0): need a + b*z < 0
    nact_b_pos = not_activated & b_positive & b_nonzero
    nact_b_neg = not_activated & b_negative & b_nonzero

    if np.any(nact_b_pos):
        itv_out[1] = min(itv_out[1], np.min(threshold[nact_b_pos]))
    if np.any(nact_b_neg):
        itv_out[0] = max(itv_out[0], np.max(threshold[nact_b_neg]))

    if itv_out[0] > itv_out[1]:
        return a_out, b_out, np.asarray([np.nan, np.nan])

    return a_out, b_out, itv_out


def MaxPool2d(a, b, z, itv, kernel_size=2, stride=2):
    """
    Propagate parametric (a, b) through MaxPool2d with constraints.

    Selects the maximum element in each pooling window.
    Constraints ensure the argmax does not change:
        a_max + b_max*z >= a_k + b_k*z  for all k in window

    Assumes padding=0 (standard for Deep SVDD).

    Args:
        a: intercept array, shape (batch, H, W, C) in NHWC
        b: coefficient array, same shape as a
        z: current parameter value
        itv: current feasible interval [lower, upper]
        kernel_size: pooling window size
        stride: stride (default = kernel_size)
    """
    X = a + b * z

    bs, h_in, w_in, ch = X.shape

    # MaxPool2d output dimensions depend on stride.  The previous reshape
    # implementation was valid only for non-overlapping pooling
    # (stride == kernel_size); sliding windows are required for overlap.
    if isinstance(kernel_size, (tuple, list)):
        kh, kw = kernel_size
    else:
        kh = kw = kernel_size
    if isinstance(stride, (tuple, list)):
        sh, sw = stride
    else:
        sh = sw = stride

    if kh <= 0 or kw <= 0 or sh <= 0 or sw <= 0:
        raise ValueError("kernel_size and stride must contain positive values")
    if kh > h_in or kw > w_in:
        raise ValueError("pooling kernel cannot exceed input dimensions")

    oH = (h_in - kh) // sh + 1
    oW = (w_in - kw) // sw + 1

    def _pool_windows(arr):
        # sliding_window_view gives (batch, H-kh+1, W-kw+1, C, kh, kw).
        windows = np.lib.stride_tricks.sliding_window_view(arr, (kh, kw), axis=(1, 2))
        return windows[:, ::sh, ::sw, :, :, :].transpose(0, 1, 2, 4, 5, 3)

    Xf = _pool_windows(X).reshape(bs, oH, oW, kh * kw, ch)
    af = _pool_windows(a).reshape(bs, oH, oW, kh * kw, ch)
    bf = _pool_windows(b).reshape(bs, oH, oW, kh * kw, ch)

    pw = kh * kw

    # Find argmax
    mi = np.argmax(Xf, axis=3)

    # Advanced indexing to extract max values
    bi_ = np.arange(bs)[:, None, None, None]
    hi_ = np.arange(oH)[None, :, None, None]
    wi_ = np.arange(oW)[None, None, :, None]
    ci_ = np.arange(ch)[None, None, None, :]

    a_out = af[bi_, hi_, wi_, mi, ci_]
    b_out = bf[bi_, hi_, wi_, mi, ci_]

    # Constraints: max element >= all other elements in window
    itv_out = itv.copy()
    for b_i in range(bs):
        for oi in range(oH):
            for oj in range(oW):
                for c in range(ch):
                    mx = mi[b_i, oi, oj, c]
                    am = af[b_i, oi, oj, mx, c]
                    bm = bf[b_i, oi, oj, mx, c]
                    for k in range(pw):
                        if k == mx:
                            continue
                        ad = am - af[b_i, oi, oj, k, c]
                        bd = bm - bf[b_i, oi, oj, k, c]
                        # Need ad + bd*z >= 0
                        if abs(bd) < 1e-12:
                            if ad < -1e-12:
                                return (
                                    a_out,
                                    b_out,
                                    np.asarray([np.nan, np.nan]),
                                )
                        elif bd > 0:
                            itv_out[0] = max(itv_out[0], -ad / bd)
                        else:
                            itv_out[1] = min(itv_out[1], -ad / bd)

    if itv_out[0] > itv_out[1]:
        return a_out, b_out, np.asarray([np.nan, np.nan])

    return a_out, b_out, itv_out
