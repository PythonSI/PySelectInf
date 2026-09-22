import numpy as np
import numpy.typing as npt
from pythonsi.node import Data
from typing import Tuple, Optional
from pythonsi.util import solve_quadratic_inequality, intersect
from pythonsi.cnn import InferenceModel as CNNInferenceModel
from pythonsi.dnn import InferenceModel as DNNInferenceModel
import torch


class DeepSVDDAD:
    def __init__(
        self,
        model: object,
        R_squared: float,
        center: npt.NDArray[np.floating],
        img_shape: Optional[tuple] = None,
        device: str = "cpu",
        network_type: str = "cnn",
    ):
        self.x_node = None
        self.anomaly_node = Data(self)

        # Cast model to float64 to be consistent with the float64 inference pipeline
        self.model = model.to(device).double()
        self.network_type = network_type
        if self.network_type == "cnn":
            self.inference_model = CNNInferenceModel(
                model, device=device, img_shape=img_shape
            )
        elif self.network_type == "dnn":
            self.inference_model = DNNInferenceModel(model, device)
        else:
            raise ValueError("network_type must be either 'cnn' or 'dnn'")
        self.R_squared = R_squared
        self.center = np.asarray(center, dtype=np.float64).reshape(1, -1)
        self.img_shape = img_shape
        self.device = device

    def run(self, x: Data) -> Data:
        r"""Connect this method to an input data node.

        Parameters
        ----------
        x : Data
            Input data node.

        Returns
        -------
        Data
            Output anomaly node.
        """
        self.x_node = x
        return self.anomaly_node

    def forward(self, x: npt.NDArray) -> Tuple[npt.NDArray, npt.NDArray]:
        r"""Run forward pass to detect anomalies.

        Parameters
        ----------
        x : array-like, shape (n, d)
            Input data.dị thường

        Returns
        -------
        anomalies : array of int
            Sorted indices of detected anomalies.
        scores : array of float
            Deep SVDD scores for each sample.
        """
        x_tensor = torch.tensor(x, dtype=torch.float64, device=self.device)
        # Reshape flat (n, d) to (n, C, H, W) if img_shape is specified
        if self.img_shape is not None and x_tensor.ndim == 2:
            n = x_tensor.shape[0]
            x_tensor = x_tensor.reshape(n, *self.img_shape)
        with torch.no_grad():
            features = self.model(x_tensor).cpu().numpy()
        features = features.astype(np.float64)
        scores = np.sum((features - self.center) ** 2, axis=1)
        anomalies = np.sort(np.where(scores > self.R_squared)[0])
        return anomalies, scores

    def __call__(self):
        r"""Execute forward pass and update anomaly node."""
        x = self.x_node()
        anomalies, _ = self.forward(x)
        self.anomaly_node.update(anomalies)
        return anomalies

    def inference(self, z: float) -> list:
        x, u, v, itv_x = self.x_node.inference(z)

        p, q, itv_net = self.inference_model.forward(u, v, z)
        anomalies, scores = self.forward(x)

        # Start with network constraints
        final_itv = intersect(itv_net, itv_x)

        # Decision constraints for each sample (quadratic)
        n = p.shape[0]
        Oz_set = set(anomalies.flatten().astype(int))

        for j in range(n):
            p_j = p[j : j + 1].astype(np.float64)
            q_j = q[j : j + 1].astype(np.float64)
            c = self.center

            diff = p_j - c
            # Quadratic: ||p + q*z - c||^2 = w*z^2 + v_coef*z + u_coef
            u_coef = float(np.sum(diff**2) - self.R_squared)
            v_coef = float(2.0 * np.sum(diff * q_j))
            w_coef = float(np.sum(q_j**2))

            if j in Oz_set:
                # Anomaly: score > R^2 → -(w*z^2 + v*z + u) <= 0
                dec = solve_quadratic_inequality(-w_coef, -v_coef, -u_coef, z)
            else:
                # Normal: score <= R^2 → w*z^2 + v*z + u <= 0
                dec = solve_quadratic_inequality(w_coef, v_coef, u_coef, z)

            final_itv = intersect(final_itv, dec)

        # Update output node
        self.anomaly_node.parametrize(data=anomalies)
        return final_itv
