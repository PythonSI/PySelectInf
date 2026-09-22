import numpy as np
import numpy.typing as npt
from typing import Tuple, List
from pythonsi.node.data import Data
from pythonsi.util import intersect, solve_linear_inequalities
from scipy.linalg import block_diag


class DeepSVDDTestStatistic:
    def __init__(self, x_test: Data, x_refs: Data):
        self.x_test_node = x_test
        self.x_refs_node = x_refs

    def __call__(
        self,
        _anomalies: npt.NDArray[np.floating],
        _anomaly_idx: int,
        Sigmas: List[npt.NDArray[np.floating]],
    ) -> Tuple[npt.NDArray, npt.NDArray, npt.NDArray, float, float, float, list]:
        x_test = self.x_test_node().flatten().astype(np.float64)
        x_refs = self.x_refs_node().astype(np.float64)
        if x_refs.ndim > 2:
            x_refs = x_refs.reshape(x_refs.shape[0], -1)
        elif x_refs.ndim == 1:
            x_refs = x_refs.reshape(1, -1)

        d = len(x_test)
        m = len(x_refs)
        sigma = Sigmas[0]

        # 1. Compute eta (test direction) — one-vs-mean
        x_ref_mean = np.mean(x_refs, axis=0)
        diff = x_test - x_ref_mean
        signs = np.ones(d, dtype=np.float64)
        signs[diff < 0] = -1.0

        eta = np.zeros((m + 1) * d, dtype=np.float64)
        eta[0:d] = signs
        eta[d:] = np.tile(-signs / m, m)

        X_flat = np.concatenate([x_test, x_refs.flatten()])

        # Format variables as column vectors for matrix math
        x = X_flat.reshape(-1, 1)
        etaj = eta.reshape(-1, 1)
        sigma_matrix = block_diag(*([sigma] * (m + 1)))

        # 1. Compute z_obs
        etajTx = etaj.T.dot(x)
        z_obs = float(etajTx.item())

        # 2. Compute variance
        etajTsigmaetaj = etaj.T.dot(sigma_matrix).dot(etaj)
        variance = float(etajTsigmaetaj.item())
        deviation = np.sqrt(variance)

        # 3. Compute a, b (parametric decomposition)
        b = sigma_matrix.dot(etaj).dot(np.linalg.inv(etajTsigmaetaj))
        a = (np.identity((m + 1) * d) - b.dot(etaj.T)).dot(x)

        # Flatten to 1D to ensure compatibility with vectorized sign constraints below
        b = b.flatten()
        a = a.flatten()

        # 4. Sign constraints — ensure sign(x_test - x_ref_mean) is preserved
        a_test = a[0:d]
        b_test = b[0:d]
        a_ref_mean = np.zeros(d, dtype=np.float64)
        b_ref_mean = np.zeros(d, dtype=np.float64)
        for k in range(m):
            a_ref_mean += a[(k + 1) * d : (k + 2) * d]
            b_ref_mean += b[(k + 1) * d : (k + 2) * d]
        a_ref_mean /= m
        b_ref_mean /= m

        a_diff = a_test - a_ref_mean
        b_diff = b_test - b_ref_mean

        # Use solve_linear_inequalities for sign constraints
        A_sign = np.where(diff >= 0, -a_diff, a_diff)
        B_sign = np.where(diff >= 0, -b_diff, b_diff)
        sign_itv = solve_linear_inequalities(A_sign, B_sign)

        itv = [-np.inf, np.inf]
        itv = intersect(itv, sign_itv)

        # 5. Parametrize input nodes
        self.x_test_node.parametrize(a=a[0:d].reshape(1, -1), b=b[0:d].reshape(1, -1))
        self.x_refs_node.parametrize(a=a[d:].reshape(m, -1), b=b[d:].reshape(m, -1))

        return eta, a, b, z_obs, variance, deviation, itv
