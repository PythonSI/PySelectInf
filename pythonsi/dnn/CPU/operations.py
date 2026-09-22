import numpy as np


def Linear(a, b, params):
    """
    (a + b * z) * W + B
    = (a * W + B) + (b * W * z)
    """
    X = np.stack([a, b], axis=0)  # (2, n, d)
    Y = np.matmul(X, params[0])  # (2, n, h)
    # Add bias with broadcasting
    a = Y[0] + params[1]  # shape (n, h) + (1, h) → (n, h)
    b = Y[1]  # shape (n, h)
    return a, b


def ReLU(a, b, z, itv):
    """
    X = a + b * z
    itv = [-inf, inf]
    where X <= 0:
        itv = intersect with solve for z: a + bz <= 0
        a = 0
        b = 0
    where X > 0:
        itv = intersect with solve for z: a + bz > 0
    """
    # Compute X = a + b * z
    X = a + b * z

    # Create masks for different regions
    negative_mask = X <= 0
    positive_mask = X > 0

    # Handle intervals - compute bounds where a + b*z <= 0 and a + b*z > 0
    # For a + b*z <= 0: z <= -a/b (if b > 0), z >= -a/b (if b < 0)
    # For a + b*z > 0: z > -a/b (if b > 0), z < -a/b (if b < 0)

    # Avoid division by zero
    b_nonzero = np.abs(b) > 1e-12
    threshold = np.where(b_nonzero, -a / b, np.inf)

    # Update intervals based on the sign of b
    b_positive = b > 0
    b_negative = b < 0

    # For negative region (X <= 0): set a = 0, b = 0
    a_out = np.where(negative_mask, 0.0, a)
    b_out = np.where(negative_mask, 0.0, b)

    # Update intervals
    itv_out = itv.copy()

    # Where X <= 0, intersect with z <= -a/b (if b > 0) or z >= -a/b (if b < 0)
    negative_and_b_pos = negative_mask & b_positive & b_nonzero
    negative_and_b_neg = negative_mask & b_negative & b_nonzero

    if np.any(negative_and_b_pos):
        valid_thresholds = threshold[negative_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    if np.any(negative_and_b_neg):
        valid_thresholds = threshold[negative_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    # Where X > 0, intersect with z > -a/b (if b > 0) or z < -a/b (if b < 0)
    positive_and_b_pos = positive_mask & b_positive & b_nonzero
    positive_and_b_neg = positive_mask & b_negative & b_nonzero

    if np.any(positive_and_b_pos):
        valid_thresholds = threshold[positive_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    if np.any(positive_and_b_neg):
        valid_thresholds = threshold[positive_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    if itv_out[0] > itv_out[1]:
        return a_out, b_out, np.asarray([np.nan, np.nan])

    return a_out, b_out, itv_out


def BatchNorm1d(a, b, params):
    """Propagate through eval-mode BatchNorm1d fused as scale and shift."""
    scale = np.asarray(params["scale"])
    shift = np.asarray(params["shift"])
    return a * scale + shift, b * scale


def LeakyReLU(a, b, z, itv, negative_slope=0.01):
    """Propagate tabular affine data through LeakyReLU."""
    X = a + b * z

    negative_mask = X < 0
    positive_mask = X >= 0

    b_nonzero = np.abs(b) > 1e-12
    threshold = np.where(b_nonzero, -a / b, np.inf)

    b_positive = b > 0
    b_negative = b < 0

    a_out = np.where(negative_mask, negative_slope * a, a)
    b_out = np.where(negative_mask, negative_slope * b, b)

    itv_out = itv.copy()

    # Negative region: a + b*z < 0
    negative_and_b_pos = negative_mask & b_positive & b_nonzero
    negative_and_b_neg = negative_mask & b_negative & b_nonzero

    if np.any(negative_and_b_pos):
        valid_thresholds = threshold[negative_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    if np.any(negative_and_b_neg):
        valid_thresholds = threshold[negative_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    # Positive region: a + b*z >= 0
    positive_and_b_pos = positive_mask & b_positive & b_nonzero
    positive_and_b_neg = positive_mask & b_negative & b_nonzero

    if np.any(positive_and_b_pos):
        valid_thresholds = threshold[positive_and_b_pos]
        if len(valid_thresholds) > 0:
            itv_out[0] = max(itv_out[0], np.max(valid_thresholds))

    if np.any(positive_and_b_neg):
        valid_thresholds = threshold[positive_and_b_neg]
        if len(valid_thresholds) > 0:
            itv_out[1] = min(itv_out[1], np.min(valid_thresholds))

    if itv_out[0] > itv_out[1]:
        return a_out, b_out, np.asarray([np.nan, np.nan])

    return a_out, b_out, itv_out
