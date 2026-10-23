"""
Data Loader & Preprocessing Module for USB-IDS-1 Dataset.
Handles loading parquet splits, cleaning infinite/missing values,
filtering benign traffic for unsupervised autoencoder training,
and applying Min-Max normalization as described in Kalutharage et al. (2023).
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.preprocessing import MinMaxScaler


class USBIDSDataLoader:
    """
    Data loader for the USB-IDS-1 dataset in Parquet format.
    Provides methods for benign extraction, min-max scaling, and scenario evaluation sets.
    """

    def __init__(self, data_dir: Union[str, Path] = "dataset"):
        self.data_dir = Path(data_dir)
        self.train_path = self.data_dir / "USB-IDS-1-TRAIN.parquet"
        self.val_path = self.data_dir / "USB-IDS-1-VALIDATION.parquet"
        self.test_path = self.data_dir / "USB-IDS-1-TEST.parquet"

        self.feature_names: List[str] = []
        self.scaler: Optional[MinMaxScaler] = None
        self._init_schema()

    def _init_schema(self) -> None:
        """Inspect schema to identify feature columns."""
        if not self.test_path.exists():
            raise FileNotFoundError(f"Test dataset not found at {self.test_path}")
        schema = pq.read_schema(self.test_path)
        self.feature_names = [col for col in schema.names if col != "Label"]

    @staticmethod
    def clean_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
        """
        Clean infinite, NaN, or extreme numeric values.
        Replaces infs with column max/finite bounds and fills NaNs with median/0.
        """
        df = df.copy()
        for col in feature_cols:
            vals = df[col].to_numpy()
            finite_mask = np.isfinite(vals)
            if not np.all(finite_mask):
                max_finite = vals[finite_mask].max() if np.any(finite_mask) else 0.0
                min_finite = vals[finite_mask].min() if np.any(finite_mask) else 0.0
                vals = np.where(vals == np.inf, max_finite, vals)
                vals = np.where(vals == -np.inf, min_finite, vals)
                vals = np.nan_to_num(vals, nan=0.0)
                df[col] = vals
        return df

    def load_benign_train(
        self, sample_size: Optional[int] = 50000, random_state: int = 42
    ) -> Tuple[np.ndarray, MinMaxScaler]:
        """
        Loads benign traffic only from the training split for unsupervised Autoencoder training.
        Fits and returns the MinMaxScaler.
        """
        print(f"Loading Benign training data from {self.train_path}...")
        # Read Label column first to identify Benign row indices
        tab_label = pq.read_table(self.train_path, columns=["Label"])
        labels = tab_label["Label"].to_pandas()
        benign_indices = np.where(labels == "Benign")[0]

        if sample_size is not None and sample_size < len(benign_indices):
            np.random.seed(random_state)
            chosen_indices = np.random.choice(benign_indices, size=sample_size, replace=False)
            chosen_indices.sort()
        else:
            chosen_indices = benign_indices

        print(f"Reading {len(chosen_indices):,} Benign training flows (Total available: {len(benign_indices):,})...")
        tab = pq.read_table(self.train_path)
        df_train = tab.take(chosen_indices).to_pandas()
        df_train = self.clean_features(df_train, self.feature_names)

        X_benign = df_train[self.feature_names].values.astype(np.float32)

        # Fit Min-Max Scaler on Benign training traffic
        print("Fitting MinMaxScaler on Benign traffic...")
        self.scaler = MinMaxScaler()
        X_benign_scaled = self.scaler.fit_transform(X_benign)

        return X_benign_scaled, self.scaler

    def load_eval_scenario(
        self,
        attack_type: str = "Hulk-NoDefense",
        split: str = "TEST",
        sample_benign: Optional[int] = 10000,
        sample_attack: Optional[int] = 20000,
        random_state: int = 42,
    ) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Creates an evaluation dataset consisting of Benign flows and a specific Attack type.
        Returns scaled features X, binary labels y (0=Benign, 1=Attack), and the unscaled DataFrame.
        """
        if self.scaler is None:
            raise ValueError("MinMaxScaler has not been fitted yet! Call load_benign_train() first.")

        file_path = self.test_path if split.upper() == "TEST" else self.val_path
        print(f"\nLoading evaluation scenario '{attack_type}' from {file_path.name}...")

        tab_label = pq.read_table(file_path, columns=["Label"])
        labels = tab_label["Label"].to_pandas()

        benign_idx = np.where(labels == "Benign")[0]
        attack_idx = np.where(labels == attack_type)[0]

        if len(attack_idx) == 0:
            raise ValueError(f"Attack '{attack_type}' not found in {file_path.name}. Available labels: {labels.unique()}")

        np.random.seed(random_state)
        if sample_benign and sample_benign < len(benign_idx):
            benign_idx = np.random.choice(benign_idx, size=sample_benign, replace=False)
        if sample_attack and sample_attack < len(attack_idx):
            attack_idx = np.random.choice(attack_idx, size=sample_attack, replace=False)

        selected_idx = np.concatenate([benign_idx, attack_idx])
        selected_idx.sort()

        tab = pq.read_table(file_path)
        df_eval = tab.take(selected_idx).to_pandas()
        df_eval = self.clean_features(df_eval, self.feature_names)

        X_raw = df_eval[self.feature_names].values.astype(np.float32)
        X_scaled = self.scaler.transform(X_raw)

        # 0 for Benign, 1 for Attack
        y = (df_eval["Label"] == attack_type).values.astype(int)

        print(f"Scenario '{attack_type}': {len(benign_idx):,} Benign flows, {len(attack_idx):,} {attack_type} flows.")
        return X_scaled, y, df_eval


if __name__ == "__main__":
    loader = USBIDSDataLoader()
    print("Features count:", len(loader.feature_names))
    X_train, scaler = loader.load_benign_train(sample_size=1000)
    print("X_train shape:", X_train.shape)
    X_test, y_test, df_test = loader.load_eval_scenario("Hulk-NoDefense", sample_benign=500, sample_attack=500)
    print("X_test shape:", X_test.shape, "y_test balance:", np.bincount(y_test))
