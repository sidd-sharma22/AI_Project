"""
Evaluation & Visualization Module.
Reproduces:
- Figures 4 to 8: Bar charts of top influential SHAP values for sample anomalous instances.
- Figure 9: Explanation of DDoS detection features (multi-bar plot of SHAP feature values across instances).
- Figures 10 & 11: Distribution of Flow Packets/s (Benign vs Attack state).
- Figures 12 & 13: Distribution of Bwd Packets/s (Benign vs Attack state).
- Generates Table 4 comparative metrics summary.
"""

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
try:
    import seaborn as sns
except ImportError:
    sns = None

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Professional publication plotting style
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.size"] = 11
plt.rcParams["axes.titlesize"] = 13
plt.rcParams["axes.labelsize"] = 11


class Visualizer:
    """
    Generates and saves the exact figures from Kalutharage et al. (2023).
    """

    def __init__(self, output_dir: Union[str, Path] = "output/figures"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_instance_shap_bar(
        self,
        instance_id: Union[int, str],
        top_features: List[Tuple[str, float]],
        save_name: Optional[str] = None,
    ) -> Path:
        """
        Replicates Figures 4-8: Bar plot of influential features for a single anomalous instance.
        """
        feats, values = zip(*top_features)
        # Convert values to positive magnitude for visual comparison matching the paper
        vals = [abs(v) for v in values]

        fig, ax = plt.subplots(figsize=(8, 4.5))
        bars = ax.bar(feats, vals, color="#2b5c8f", width=0.6, edgecolor="black", linewidth=0.7)

        ax.set_title(f"Figure: Anomalous Instance {instance_id}", fontweight="bold", pad=12)
        ax.set_ylabel("SHAP Value (Feature Influence)", fontweight="bold")
        ax.set_xticks(range(len(feats)))
        ax.set_xticklabels(feats, rotation=45, ha="right", fontsize=9)
        ax.set_ylim(0, max(vals) * 1.18 if vals else 1.0)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        # Value annotations
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )

        plt.tight_layout()
        fname = save_name or f"fig_instance_{instance_id}.png"
        save_path = self.output_dir / fname
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path.name}")
        return save_path

    def plot_figure_9_ddos_explanation(
        self,
        instances_data: Dict[str, Dict[str, float]],
        save_name: str = "fig9_ddos_feature_explanation.png",
    ) -> Path:
        """
        Replicates Figure 9: Multi-bar chart comparing SHAP values of key DDoS detection features
        (Flow Packets/s, Bwd Packets/s, Fwd Packet Length Std, Bwd Packet Length Max)
        across anomalous instances.
        """
        instance_names = list(instances_data.keys())
        features_to_plot = [
            "Flow Packets/s",
            "Bwd Packets/s",
            "Fwd Packet Length Std",
            "Bwd Packet Length Max",
        ]
        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]

        x = np.arange(len(instance_names))
        width = 0.18

        fig, ax = plt.subplots(figsize=(10, 5))

        for i, feat in enumerate(features_to_plot):
            vals = [abs(instances_data[inst].get(feat, 0.0)) for inst in instance_names]
            offset = (i - 1.5) * width
            ax.bar(x + offset, vals, width, label=feat, color=colors[i], edgecolor="black", linewidth=0.6)

        ax.set_title("Figure 9: Explanation of DDoS Detection Features Across Anomalous Instances", fontweight="bold", pad=12)
        ax.set_xlabel("Anomalous Instances", fontweight="bold")
        ax.set_ylabel("Feature Value (SHAP) for each instance", fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(instance_names)
        ax.legend(frameon=True, facecolor="white", edgecolor="none")
        ax.set_ylim(0, 0.65)
        ax.grid(axis="y", linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path.name}")
        return save_path

    def plot_feature_distribution_strip(
        self,
        values: np.ndarray,
        feature_name: str,
        state_label: str,
        max_xlim: float,
        save_name: str,
    ) -> Path:
        """
        Replicates Figures 10, 11, 12, 13: Horizontal point/strip distribution plots.
        """
        fig, ax = plt.subplots(figsize=(8, 2.5))

        # Sample for clean rendering if dataset is huge
        if len(values) > 3000:
            sample_vals = np.random.choice(values, size=3000, replace=False)
        else:
            sample_vals = values

        # Scatter points along a single horizontal axis with jitter
        y_jitter = np.random.normal(0, 0.02, size=len(sample_vals))
        ax.scatter(sample_vals, y_jitter, alpha=0.4, color="#1c3b5e", s=12, edgecolors="none")

        ax.set_title(f"Figure: {feature_name} — {state_label} Feature Value", fontweight="bold", pad=10)
        ax.set_xlabel(feature_name, fontweight="bold")
        ax.set_yticks([])
        ax.set_xlim(-100, max_xlim)
        ax.set_ylim(-0.15, 0.15)
        ax.grid(axis="x", linestyle="--", alpha=0.7)

        plt.tight_layout()
        save_path = self.output_dir / save_name
        plt.savefig(save_path, dpi=300)
        plt.close(fig)
        print(f"Saved: {save_path.name}")
        return save_path


if __name__ == "__main__":
    viz = Visualizer()

    # Test Figure 4-8 style
    dummy_top = [
        ("Flow Packets/s", 0.58),
        ("Bwd Packets/s", 0.49),
        ("Packet Length Max", 0.23),
        ("Avg Packet Size", 0.05),
        ("Bwd Packet Length Std", 0.04),
    ]
    viz.plot_instance_shap_bar("384574", dummy_top)

    # Test Figure 9
    dummy_instances = {
        "124930": {"Flow Packets/s": 0.55, "Bwd Packets/s": 0.48, "Fwd Packet Length Std": 0.0, "Bwd Packet Length Max": 0.0},
        "602902": {"Flow Packets/s": 0.59, "Bwd Packets/s": 0.51, "Fwd Packet Length Std": 0.0, "Bwd Packet Length Max": 0.01},
        "384574": {"Flow Packets/s": 0.57, "Bwd Packets/s": 0.49, "Fwd Packet Length Std": 0.0, "Bwd Packet Length Max": 0.0},
        "686625": {"Flow Packets/s": 0.54, "Bwd Packets/s": 0.47, "Fwd Packet Length Std": 0.05, "Bwd Packet Length Max": 0.02},
        "718029": {"Flow Packets/s": 0.56, "Bwd Packets/s": 0.49, "Fwd Packet Length Std": 0.04, "Bwd Packet Length Max": 0.02},
    }
    viz.plot_figure_9_ddos_explanation(dummy_instances)
