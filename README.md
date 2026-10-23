# Explainable AI-Based DDoS Attack Identification for IoT Networks

Implementation and experimental reproduction of the research paper:  
> **"Explainable AI-Based DDOS Attack Identification Method for IoT Networks"**  
> *Chathuranga Sampath Kalutharage, Xiaodong Liu, Christos Chrysoulas, Nikolaos Pitropakis, Pavlos Papadopoulos*  
> **MDPI Computers 2023, 12(2), 32.** DOI: [10.3390/computers12020032](https://doi.org/10.3390/computers12020032)  
>
> **Course:** CSE311 — Artificial Intelligence, IIIT Kottayam  
> **Group Members:** Siddharth Sharma, Rohan Kumar, Alok Kumar, Aqueel Ahmad 

---

## 🌟 Executive Summary

Traditional Machine Learning Intrusion Detection Systems (IDS) rely heavily on **supervised learning** (Random Forests, Decision Trees, DNNs). However, they suffer from three fatal flaws in real-world security:
1. **Labeled Data Scarcity:** Training requires massive labeled attack traffic, which is scarce and legally difficult to acquire.
2. **Collapse on Evasive Attacks:** In the paper's benchmark on the USB-IDS-1 dataset, when an attacker employs evasion techniques (`Hulk-Evasive`), Decision Tree accuracy collapses to **6%** and Random Forest collapses to **0%**!
3. **Black-Box Nature:** Deep neural networks provide no explanation for why an alert was triggered, leading to alert fatigue and low trust.

### The Paper's Core Framework:
- **Unsupervised Anomaly Detection:** A lightweight Deep Autoencoder (32 and 10 hidden neurons) trained **strictly on Benign traffic**. When an attack flow passes through, the autoencoder fails to reconstruct it, resulting in a spike in reconstruction error ($L(A, A') = \sum (a_i - a'_i)^2$).
- **Local Explainable AI (Kernel SHAP):** Dissects the reconstruction error by calculating SHAP attribution values for the Top-$R$ influential features.
- **Domain Knowledge Integration (21 DDoS Features):** Maps influential features to 21 informative DDoS features across three categories:
  - **TCP State Exhaustion** (e.g., SYN rate, ACK intervals)
  - **Volumetric** (e.g., Packets/s, Flow Duration, Packet Lengths)
  - **Application-Layer** (e.g., Ports, Protocols)
- **F1 Threshold Optimization & Attack Certainty:** Features are tuned using $T = \arg\max(F_1)$ (Eq. 3). If influential DDoS features exceed their thresholds, the flow is classified as a confirmed DDoS attack with an explicit, interpretable explanation.

---

## 📂 Repository Structure

```
5AIP/
├── 1. Paper.pdf                                # Original MDPI paper
├── requirements.txt                            # Python dependencies
├── README.md                                   # Comprehensive documentation
├── dataset/                                    # Official USB-IDS-1 Parquet splits
│   ├── USB-IDS-1-TRAIN.parquet                 # 70% split (3.36M flows)
│   ├── USB-IDS-1-VALIDATION.parquet            # 15% split (721K flows)
│   └── USB-IDS-1-TEST.parquet                  # 15% split (722K flows)
├── notebooks/                                  # Jupyter Notebooks
│   └── DDoS_XAI_Paper_Implementation.ipynb     # Master interactive reproduction notebook
├── src/                                        # Modular Python Implementation
│   ├── __init__.py
│   ├── data_loader.py                          # Fast Parquet loader, cleaning & MinMaxScaler
│   ├── autoencoder.py                          # PyTorch Autoencoder (77 -> 32 -> 10 -> 32 -> 77)
│   ├── xai_explainer.py                        # Kernel SHAP attribution & Top-R ranking (Alg. 1)
│   ├── ddos_identifier.py                      # 21 DDoS features & F1 threshold tuning (Alg. 2, Eq. 3)
│   ├── baselines.py                            # Supervised baselines (DT, RF, DNN)
│   ├── evaluate_and_plot.py                    # Generation of Figures 4–13 & metrics
│   └── main.py                                 # Automated end-to-end pipeline runner
├── output/
│   ├── models/                                 # Trained model weights (autoencoder_best.pt)
│   ├── figures/                                # Replicated Figures 4–13 (PNG format)
│   └── table4_comparison.csv                   # Table 4 benchmark output
└── presentation/
    ├── presentation_slides.md                  # 12 slide deck with verbatim speaker scripts
    └── viva_questions_and_answers.md           # 12 tough faculty Q&A answers
```

---

## * Quick Start & How to Run

### 1. Prerequisites
Ensure Python 3.10+ is installed. Install required packages:
```bash
pip install -r requirements.txt
```

### 2. Run the Automated Pipeline
To execute the complete end-to-end reproduction:
```bash
python src/main.py
```
This script will automatically:
1. Load and normalize benign training flows.
2. Train the PyTorch Autoencoder for 40 epochs.
3. Calibrate the benign anomaly threshold.
4. Tune F1 thresholds for the 21 DDoS features using the validation split.
5. Benchmark against Decision Tree, Random Forest, and DNN on `Hulk-NoDefense`, `Hulk-Evasive`, and `Hulk-Reqtimeout` (reproducing Table 4).
6. Generate Kernel SHAP explainability bar plots (Figures 4–8) and Figure 9.
7. Generate traffic distribution strip plots (Figures 10–13).

### 3. Run the Interactive Jupyter Notebook
To run or present the step-by-step notebook:
```bash
jupyter DDoS_XAI_Paper_Implementation.ipynb
```

---

## 📊 Experimental Results & Paper Replication

### 1. Table 4: State-of-the-Art Model Comparison (HULK Attack)
Evaluation on the USB-IDS-1 test set comparing the Proposed Method against baseline classifiers:

| Detection Method | Hulk No Defense | Hulk Evasive | Hulk Reqtimeout |
| :--- | :---: | :---: | :---: |
| **Decision Tree (DT)** | 0.97 | 0.06 | 0.97 |
| **Random Forest (RF)** | 0.98 | 0.00 | 0.98 |
| **Deep Neural Network (DNN)** | 0.67 | 0.05 / 0.33 | 0.66 |
| **Proposed Method (Autoencoder + XAI)** | **0.98** | **0.95 – 1.00** | **0.98 – 1.00** |

> **Key Finding:** Supervised models overfit to specific attack characteristics, causing them to collapse to near **0% accuracy on `Hulk-Evasive`**. The unsupervised Autoencoder + XAI model only learns benign traffic, remaining resilient at **~95%–100% accuracy** across all scenarios.

### 2. Explainability & Visualizations Replicated
All figures are generated in `output/figures/`:
- **Figures 4–8:** Bar plots of Top-$R$ influential SHAP values for sample anomalous instances (`384574`, `602902`, `686625`, `718029`, `124930`), showing `Flow Packets/s` and `Bwd Packets/s` as the dominant attack vectors.
- **Figure 9:** Multi-bar grouped plot comparing SHAP values of key DDoS features across instances.
- **Figures 10 & 11:** Distribution of `Flow Packets/s` (Benign: 0–5,000 pkts/s vs. Attack: up to 22,000 pkts/s).
- **Figures 12 & 13:** Distribution of `Bwd Packets/s` (Benign: 0–3,500 pkts/s vs. Attack: up to 10,000 pkts/s).

---
*Created as part of the CSE311 - Artificial Intelligence Coursework*
