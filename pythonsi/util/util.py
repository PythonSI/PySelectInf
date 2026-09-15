import numpy as np
import numpy.typing as npt
from mpmath import mp

mp.dps = 500


def FormatArray(arr: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    r"""Format array to ensure proper dimensionality for computations.

    Reshapes 1D arrays to column vectors for consistent matrix operations.

    Parameters
    ----------
    arr : array-like, shape (n,) or (n, d)
        Input array to format

    Returns
    -------
    formatted_arr : array-like, shape (n, 1) or (n, d)
        Formatted array with proper dimensions

    Examples
    --------
    >>> arr = np.array([1, 2, 3])
    >>> formatted = FormatArray(arr)
    >>> print(formatted.shape)  # (3, 1)
    """
    if arr.ndim == 1:
        return arr.reshape(-1, 1)
    return arr


def intersect(interval1, interval2):
    r"""Intersect two intervals with support for unbounded intervals.

    Computes the intersection of two intervals :math:`[l_1, u_1]` and
    :math:`[l_2, u_2]`, where None represents unbounded values.

    Parameters
    ----------
    interval1 : list or tuple
        First interval [lower, upper] where None means unbounded
    interval2 : list or tuple
        Second interval [lower, upper] where None means unbounded

    Returns
    -------
    intersection : list
        Intersected interval [lower, upper] or [] if no overlap

    Raises
    ------
    AssertionError
        If logic error occurs in interval computation

    Examples
    --------
    >>> intersect([1, 5], [3, 7])
    [3, 5]
    >>> intersect([None, 5], [3, None])
    [3, 5]
    >>> intersect([1, 3], [5, 7])
    []
    """
    l1, u1 = interval1
    l2, u2 = interval2

    # Handle lower bounds
    if l1 is None:
        lower = l2
    elif l2 is None:
        lower = l1
    else:
        lower = max(l1, l2)

    # Handle upper bounds
    if u1 is None:
        upper = u2
    elif u2 is None:
        upper = u1
    else:
        upper = min(u1, u2)

    # Check feasibility
    if (lower is not None) and (upper is not None) and (lower > upper):
        assert False, f"Logic error in intersect_intervals: {lower} {upper}"

    return [lower, upper]


def solve_linear_inequalities(A: np.ndarray, B: np.ndarray):
    r"""Solve system of linear inequalities vectorized.

    Solves the system of inequalities:

    .. math::
        A_i + B_i z \leq 0 \quad \forall i

    which is equivalent to finding the feasible region for :math:`z`.

    Parameters
    ----------
    A : array-like, shape (n,)
        Constant terms in inequalities
    B : array-like, shape (n,)
        Coefficients of z in inequalities

    Returns
    -------
    interval : list
        Feasible interval [lower, upper] where None means unbounded

    Raises
    ------
    ValueError
        If A and B have different shapes
    AssertionError
        If no feasible solution exists or logic error occurs

    Notes
    -----
    The algorithm handles three cases:

    - :math:`B_i > 0`: gives upper bound :math:`z \leq -A_i/B_i`
    - :math:`B_i < 0`: gives lower bound :math:`z \geq -A_i/B_i`
    - :math:`B_i = 0`: constraint is :math:`A_i \leq 0` (must be satisfied)

    Examples
    --------
    >>> A = np.array([1, -2, 3])
    >>> B = np.array([2, -1, 0])
    >>> interval = solve_linear_inequalities(A, B)
    """
    if A.shape != B.shape:
        raise ValueError("A and B must have the same shape")

    if A.size == 0:
        return [-np.inf, np.inf]

    # Case 1: B > 0  →  z <= -A/B
    mask_pos = B > 1e-10
    upper_bounds = np.full(A.shape, np.inf, dtype=float)
    upper_bounds[mask_pos] = -A[mask_pos] / B[mask_pos]

    # Case 2: B < 0  →  z >= -A/B
    mask_neg = B < -1e-10
    lower_bounds = np.full(A.shape, -np.inf, dtype=float)
    lower_bounds[mask_neg] = -A[mask_neg] / B[mask_neg]

    # Case 3: B == 0
    mask_zero = (B >= -1e-10) & (B <= 1e-10)
    if np.any(mask_zero & (A > 0)):
        assert False, "No satisfying solution"

    # Global bounds
    lower = np.max(lower_bounds)
    upper = np.min(upper_bounds)

    # Convert infinities to None for readability
    low = None if np.isneginf(lower) else lower
    up = None if np.isposinf(upper) else upper

    # Check feasibility
    if (low is not None) and (up is not None) and (low > up):
        assert False, "Logic error in solve_linear_inequalities"

    return [low, up]


def solve_quadratic_inequality(
    a: float, b: float, c: float, z: float, tol: float = 1e-12
):
    r"""Solve quadratic inequality and return interval containing z.

    Solves the quadratic inequality:

    .. math::
        a z^2 + b z + c \leq 0

    and returns the interval that contains the given point :math:`z`.

    Parameters
    ----------
    a : float
        Quadratic coefficient
    b : float
        Linear coefficient
    c : float
        Constant term
    z : float
        Point that must be contained in solution interval
    tol : float, optional
        Numerical tolerance for comparisons, default 1e-12

    Returns
    -------
    interval : list
        Solution interval [low, high] containing z

    Raises
    ------
    AssertionError
        If no solution interval contains z

    Notes
    -----
    The function handles three cases:

    - Linear case (:math:`a \approx 0`): reduces to linear inequality
    - Quadratic with no real roots (:math:`\Delta < 0`): solution depends on sign of :math:`a`
    - Quadratic with real roots (:math:`\Delta \geq 0`): solution is between/outside roots

    Examples
    --------
    >>> # Solve x^2 - 4 <= 0, check interval containing z=1
    >>> interval = solve_quadratic_inequality(1, 0, -4, 1)
    >>> print(interval)  # [-2, 2]
    """
    # Linear case
    if abs(a) < tol:
        if abs(b) < tol:
            interval = [] if c > tol else [-np.inf, np.inf]
        elif b > 0:
            interval = [-np.inf, -c / b]
        else:
            interval = [-c / b, np.inf]
    else:
        # Quadratic case
        D = b**2 - 4 * a * c
        if D < -tol:
            interval = [] if a > 0 else [-np.inf, np.inf]
        elif abs(D) <= tol:
            r = -b / (2 * a)
            if a > 0:
                interval = [r, r] if abs(z - r) <= tol else []
            else:
                interval = [-np.inf, np.inf]
        else:  # D > 0
            sqrtD = np.sqrt(D)
            r1, r2 = sorted([(-b - sqrtD) / (2 * a), (-b + sqrtD) / (2 * a)])
            if a > 0:
                interval = [r1, r2] if r1 <= z <= r2 else []
            else:
                left = [-np.inf, r1]
                right = [r2, np.inf]
                if left[0] <= z <= left[1]:
                    interval = left
                elif right[0] <= z <= right[1]:
                    interval = right
                else:
                    interval = []

    # Assert if no solution
    assert interval != [], f"No solution interval contains z={z}"
    return interval


def compute_p_value(
    test_statistic, variance, list_intervals, list_outputs, observed_output
):
    r"""Compute selective inference p-value using interval conditioning.

    Computes the p-value for selective inference by conditioning on the
    selection event. The p-value is calculated as:

    .. math::
        p = 2 \min\left(\frac{\sum_{i: S_i = S} \Phi\left(\frac{T}{\sigma}\right) - \Phi\left(\frac{l_i}{\sigma}\right)}{\sum_{i: S_i = S} \Phi\left(\frac{r_i}{\sigma}\right) - \Phi\left(\frac{l_i}{\sigma}\right)}, 1 - \text{this ratio}\right)

    where :math:`\Phi` is the standard normal CDF, :math:`S_i` are selection outputs,
    :math:`[l_i, r_i]` are intervals, and :math:`T` is the test statistic.

    Parameters
    ----------
    test_statistic : float
        Observed test statistic value
    variance : float
        Variance parameter for the test statistic
    list_intervals : list of lists
        List of intervals [left, right] for each selection region
    list_outputs : list
        List of selection outputs corresponding to each interval
    observed_output : array-like
        The observed selection output to condition on

    Returns
    -------
    p_value : float
        Two-sided selective inference p-value

    Notes
    -----
    This function uses high-precision arithmetic (500 decimal places) via
    mpmath to ensure numerical stability in tail probability computations.

    Examples
    --------
    >>> test_stat = 2.5
    >>> var = 1.0
    >>> intervals = [[-1, 3], [0, 4]]
    >>> outputs = [np.array([1, 2]), np.array([1, 2])]
    >>> observed = np.array([1, 2])
    >>> p_val = compute_p_value(test_stat, var, intervals, outputs, observed)
    """
    mp.dps = 500
    numerator = 0
    denominator = 0

    standard_deviation = np.sqrt(variance)
    for i in range(len(list_intervals)):
        left, right = list_intervals[i]
        output = list_outputs[i]

        if not np.array_equal(output, observed_output):
            continue

        denominator = (
            denominator
            + mp.ncdf(right / standard_deviation)
            - mp.ncdf(left / standard_deviation)
        )
        if test_statistic >= right:
            numerator = (
                numerator
                + mp.ncdf(right / standard_deviation)
                - mp.ncdf(left / standard_deviation)
            )
        elif (test_statistic >= left) and (test_statistic < right):
            numerator = (
                numerator
                + mp.ncdf(test_statistic / standard_deviation)
                - mp.ncdf(left / standard_deviation)
            )


    cdf = float(numerator / denominator)
    return 2 * min(cdf, 1 - cdf)
