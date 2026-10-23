"""
DDoS Identification & Domain Feature Mapping Module.
Implements Section 3.4, Section 3.5, Equation (3), and Algorithm 2 from Kalutharage et al. (2023):
- Defines the 21 informative DDoS features across 3 categories:
    1. TCP State Exhaustion
    2. Volumetric
    3. Application-Layer
- Implements F1-score threshold tuning (Eq. 3): T = argmax(F1).
- Maps Top-R influential SHAP features with the 21 DDoS features (Algorithm 2).
- Classifies flows with high certainty and provides human-interpretable explanations.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class DDoSIdentifier:
    """
    Manages domain-guided DDoS attack identification using the paper's 21 DDoS features.
    """

    # Mapping paper feature names to exact USB-IDS-1 column names
    DDOS_FEATURES_MAP = {
        # --- State Exhaustion Features (Table 1) ---
        "fwd-TCP-num": "Total Fwd Packet",
        "fwd-ACK-num": "ACK Flag Count",
        "fwd-max-ACK": "Fwd IAT Max",
        "fwd-SYN-num": "SYN Flag Count",
        "fwd-SYN-rate": "SYN Flag Count",  # Normalized rate or count
        # --- Application Layer Features (Table 2) ---
        "protocol": "Protocol",
        # --- Volumetric Features (Table 3) ---
        "flow-duration": "Flow Duration",
        "tot-fwd-pkts": "Total Fwd Packet",
        "tot-bwd-pkts": "Total Bwd packets",
        "tot-len-bwd-pkts": "Total Length of Bwd Packet",
        "fwd-pkts-len-min": "Fwd Packet Length Min",
        "Fwd-Packet-Length-Std": "Fwd Packet Length Std",
        "bwd-pkts-len-max": "Bwd Packet Length Max",
        "flow-packets/s": "Flow Packets/s",
        "bwd-IAT-tot": "Bwd IAT Total",
        "bwd-IAT-mean": "Bwd IAT Mean",
        "fwd-psh-num": "PSH Flag Count",
        "Bwd-Packets/s": "Bwd Packets/s",
        "fwd-ent-int": "Flow IAT Std",
    }

    def __init__(self, available_columns: List[str]):
        self.available_columns = available_columns
        # Match features that are actually in the dataset
        self.ddos_columns: Set[str] = set()
        for paper_name, col_name in self.DDOS_FEATURES_MAP.items():
            if col_name in available_columns:
                self.ddos_columns.add(col_name)

        # Fallback check for case-insensitive matching
        col_lookup = {c.lower(): c for c in available_columns}
        for paper_name, col_name in self.DDOS_FEATURES_MAP.items():
            if col_name.lower() in col_lookup:
                self.ddos_columns.add(col_lookup[col_name.lower()])

        self.ddos_columns_list = sorted(list(self.ddos_columns))
        self.feature_thresholds: Dict[str, float] = {}

    def tune_thresholds(
        self,
        X_df: pd.DataFrame,
        y_true: np.ndarray,
        num_candidates: int = 50,
        verbose: bool = True,
    ) -> Dict[str, float]:
        """
        Tunes the threshold value for each DDoS feature using Equation (3):
        T = argmax(F1)
        """
        if verbose:
            print(f"Tuning F1 thresholds across {len(self.ddos_columns_list)} DDoS features...")

        for col in self.ddos_columns_list:
            vals = X_df[col].values
            # Filter finite positive values
            finite_vals = vals[np.isfinite(vals)]
            if len(finite_vals) == 0:
                continue

            q_low, q_high = np.percentile(finite_vals, [5, 99])
            candidates = np.linspace(q_low, q_high, num_candidates)

            best_t = float(candidates[0])
            best_f1 = -1.0

            for cand in candidates:
                preds = (vals > cand).astype(int)
                score = f1_score(y_true, preds, zero_division=0)
                if score > best_f1:
                    best_f1 = score
                    best_t = float(cand)

            self.feature_thresholds[col] = best_t
            if verbose and col in ["Flow Packets/s", "Bwd Packets/s", "Total Fwd Packet", "Fwd Packet Length Std"]:
                print(f"  Feature '{col}': Optimal Threshold = {best_t:.2f} (F1 = {best_f1:.4f})")

        return self.feature_thresholds

    def map_influential_ddos_features(
        self, top_r_features: List[Tuple[str, float]]
    ) -> List[Tuple[str, float]]:
        """
        Implements Algorithm 2 from the paper:
        Finds the intersection between the Top-R SHAP influential features and the 21 DDoS features.
        """
        influential_ddos = []
        for feat_name, shap_val in top_r_features:
            if feat_name in self.ddos_columns:
                influential_ddos.append((feat_name, shap_val))
        return influential_ddos

    def evaluate_ddos_attack(
        self,
        x_raw_row: pd.Series,
        top_r_features: List[Tuple[str, float]],
    ) -> Tuple[bool, str, List[Dict[str, Union[str, float, bool]]]]:
        """
        Section 3.5: If the most influential DDoS features exceed their threshold,
        classify as DDoS attack with an explanation leading to attack certainty.
        """
        influential_ddos = self.map_influential_ddos_features(top_r_features)
        explanations = []
        is_ddos = False

        for feat_name, shap_val in influential_ddos:
            actual_val = float(x_raw_row.get(feat_name, 0.0))
            threshold = self.feature_thresholds.get(feat_name, np.inf)
            exceeds = actual_val > threshold

            explanations.append(
                {
                    "feature": feat_name,
                    "actual_value": actual_val,
                    "threshold": threshold,
                    "shap_value": shap_val,
                    "exceeds_threshold": exceeds,
                }
            )
            if exceeds:
                is_ddos = True

        if is_ddos:
            exceeded_feats = [e["feature"] for e in explanations if e["exceeds_threshold"]]
            summary = (
                f"CONFIRMED DDOS ATTACK: Influential feature(s) {exceeded_feats} "
                f"exceeded domain security threshold(s)."
            )
        else:
            summary = "BENIGN / UNCONFIRMED: None of the influential DDoS features exceeded threshold."

        return is_ddos, summary, explanations


if __name__ == "__main__":
    from src.data_loader import USBIDSDataLoader

    loader = USBIDSDataLoader()
    loader.load_benign_train(sample_size=100)
    _, y_val, df_val = loader.load_eval_scenario("Hulk-NoDefense", split="VALIDATION", sample_benign=500, sample_attack=500)
    identifier = DDoSIdentifier(loader.feature_names)
    print("DDoS features identified in dataset:", len(identifier.ddos_columns_list))
    identifier.tune_thresholds(df_val, y_val)
