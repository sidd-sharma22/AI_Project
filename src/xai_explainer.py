"""
Explainable AI (XAI) Module using Kernel SHAP.
Implements Section 3.3 and Algorithm 1 from Kalutharage et al. (2023):
- Ranks feature-wise reconstruction errors |a_i - a'_i|.
- Computes SHAP values using Kernel SHAP on the Autoencoder with a background set of benign samples.
- Extracts Top-R most influential features for each anomalous network flow instance.
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import shap
import sys
from pathlib import Path

# Add project root to sys.path so 'from src....' always works
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.autoencoder import AutoencoderDetector


class XAIExplainer:
    """
    Computes explainability attributions using Kernel SHAP on the Autoencoder anomaly detector.
    """

    def __init__(
        self,
        detector: AutoencoderDetector,
        feature_names: List[str],
        background_data: np.ndarray,
        num_background: int = 50,
        random_state: int = 42,
    ):
        self.detector = detector
        self.feature_names = feature_names

        # Subsample background benign flows for Kernel SHAP to maintain high speed and accuracy
        np.random.seed(random_state)
        if len(background_data) > num_background:
            idx = np.random.choice(len(background_data), size=num_background, replace=False)
            self.background = background_data[idx]
        else:
            self.background = background_data

        # Predict function for scalar reconstruction error
        def error_predict_fn(x: np.ndarray) -> np.ndarray:
            reconstructed = self.detector.reconstruct(x)
            # Row-wise sum of squared differences L(A, A')
            return np.sum(np.square(x - reconstructed), axis=1)

        self.error_predict_fn = error_predict_fn
        self.explainer = shap.KernelExplainer(self.error_predict_fn, self.background)

    def get_top_reconstruction_features(
        self, x_instance: np.ndarray, top_k: int = 15
    ) -> List[Tuple[str, float]]:
        """
        Ranks features by absolute reconstruction error |a_i - a'_i| as described in the paper:
        |a_1 - a'_1| > |a_2 - a'_2| > ... > |a_n - a'_n|
        """
        x_2d = x_instance.reshape(1, -1)
        reconstructed = self.detector.reconstruct(x_2d)[0]
        abs_diff = np.abs(x_instance - reconstructed)
        ranked_idx = np.argsort(abs_diff)[::-1][:top_k]

        return [(self.feature_names[i], float(abs_diff[i])) for i in ranked_idx]

    def explain_instance(
        self, x_instance: np.ndarray, nsamples: int = 100
    ) -> Dict[str, float]:
        """
        Computes SHAP attribution values for a single anomalous flow instance.
        Returns a dictionary mapping feature_name -> shap_value.
        """
        x_2d = x_instance.reshape(1, -1)
        shap_vals = self.explainer.shap_values(x_2d, nsamples=nsamples, l1_reg="auto")

        # In shap 0.50+, KernelExplainer returns 2D array or 1D array depending on output shape
        if isinstance(shap_vals, list):
            sv = shap_vals[0]
        else:
            sv = shap_vals

        if sv.ndim > 1:
            sv = sv[0]

        feature_shap_dict = {
            self.feature_names[i]: float(sv[i]) for i in range(len(self.feature_names))
        }
        return feature_shap_dict

    def get_top_r_influential_features(
        self, feature_shap_dict: Dict[str, float], top_r: int = 10
    ) -> List[Tuple[str, float]]:
        """
        Selects top-R features based on highest SHAP attribution magnitude.
        As shown in Figures 4-8 of the paper.
        """
        sorted_features = sorted(
            feature_shap_dict.items(), key=lambda item: abs(item[1]), reverse=True
        )
        return sorted_features[:top_r]


if __name__ == "__main__":
    from src.data_loader import USBIDSDataLoader

    loader = USBIDSDataLoader()
    X_train, _ = loader.load_benign_train(sample_size=300)
    detector = AutoencoderDetector(input_dim=len(loader.feature_names))
    detector.fit(X_train, epochs=2, batch_size=64, verbose=False)

    explainer = XAIExplainer(detector, loader.feature_names, X_train, num_background=10)
    sample_anomaly = X_train[0]
    top_diff = explainer.get_top_reconstruction_features(sample_anomaly, top_k=5)
    print("Top reconstruction diffs:", top_diff)
    shaps = explainer.explain_instance(sample_anomaly, nsamples=30)
    top_r = explainer.get_top_r_influential_features(shaps, top_r=5)
    print("Top-R SHAP features:", top_r)
