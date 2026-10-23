"""
Baseline Supervised Models Module.
Replicates Table 4 from Kalutharage et al. (2023):
- Decision Tree (DT)
- Random Forest (RF)
- Deep Neural Network (DNN / MLPClassifier)
Demonstrates why traditional supervised models collapse on evasive attacks (Hulk-Evasive),
while the unsupervised Autoencoder + XAI framework remains resilient.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))


class BaselineBenchmark:
    """
    Trains and evaluates the baseline supervised models mentioned in Table 4 of the paper.
    """

    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        self.models = {
            "Decision Tree (DT)": DecisionTreeClassifier(random_state=random_state, max_depth=15),
            "Random Forest (RF)": RandomForestClassifier(n_estimators=50, random_state=random_state, n_jobs=-1, max_depth=15),
            "Deep Neural Network (DNN)": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=20, random_state=random_state),
        }
        self.trained_models = {}

    def fit_baselines(self, X_train: np.ndarray, y_train: np.ndarray, verbose: bool = True) -> None:
        """
        Fits DT, RF, and DNN on labeled training traffic (Benign vs Normal Hulk Attack).
        """
        for name, model in self.models.items():
            if verbose:
                print(f"Training {name} on {len(X_train):,} samples...")
            model.fit(X_train, y_train)
            self.trained_models[name] = model

    def evaluate_model(
        self, model_name: str, X_test: np.ndarray, y_test: np.ndarray
    ) -> Dict[str, float]:
        """
        Evaluates a trained model and returns accuracy, precision, recall, and f1.
        """
        model = self.trained_models.get(model_name)
        if model is None:
            raise ValueError(f"Model '{model_name}' has not been trained yet!")

        y_pred = model.predict(X_test)
        return {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }

    def run_table4_comparison(
        self,
        eval_scenarios: Dict[str, Tuple[np.ndarray, np.ndarray]],
        proposed_results: Optional[Dict[str, float]] = None,
    ) -> pd.DataFrame:
        """
        Generates the comparative table matching Table 4 in the paper across scenarios:
        - Hulk No Defense
        - Hulk Evasive
        - Hulk Reqtimeout
        """
        rows = []
        for name in self.models.keys():
            row = {"Detection Method": name}
            for scenario_name, (X_scen, y_scen) in eval_scenarios.items():
                metrics = self.evaluate_model(name, X_scen, y_scen)
                row[scenario_name] = round(metrics["accuracy"], 2)
            rows.append(row)

        if proposed_results:
            prop_row = {"Detection Method": "Proposed Method (Autoencoder + XAI)"}
            prop_row.update(proposed_results)
            rows.append(prop_row)

        df_comparison = pd.DataFrame(rows)
        return df_comparison


if __name__ == "__main__":
    from src.data_loader import USBIDSDataLoader

    loader = USBIDSDataLoader()
    loader.load_benign_train(sample_size=100)
    X_train_nd, y_train_nd, _ = loader.load_eval_scenario("Hulk-NoDefense", split="TRAIN", sample_benign=500, sample_attack=500)

    bench = BaselineBenchmark()
    bench.fit_baselines(X_train_nd, y_train_nd)

    X_test_nd, y_test_nd, _ = loader.load_eval_scenario("Hulk-NoDefense", split="TEST", sample_benign=200, sample_attack=200)
    X_test_ev, y_test_ev, _ = loader.load_eval_scenario("Hulk-Evasive", split="TEST", sample_benign=200, sample_attack=200)

    scenarios = {"Hulk No Defense": (X_test_nd, y_test_nd), "Hulk Evasive": (X_test_ev, y_test_ev)}
    df_res = bench.run_table4_comparison(scenarios, proposed_results={"Hulk No Defense": 0.98, "Hulk Evasive": 1.00})
    print("\nBenchmark Results:")
    print(df_res.to_string(index=False))
