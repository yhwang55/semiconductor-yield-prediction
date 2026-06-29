# Semiconductor Yield Prediction & Root-Cause Factor Identification

<p align="center">
  <a href="https://yhwang55.github.io/semiconductor-yield-prediction/" target="_blank">
    <img src="https://img.shields.io/badge/🔬%20Live%20Dashboard-Open%20Interactive%20Demo-1428A0?style=for-the-badge&logoColor=white" alt="Live Dashboard" />
  </a>
</p>

> Semiconductor yield **predictive modeling** and **statistical root-cause factor identification** — a 5-phase incremental ML pipeline on the UCI SECOM dataset

---

## Key Results

| Metric | Value |
|--------|-------|
| **Final PR-AUC** | 0.227 (XGBoost + Optuna, StratifiedKFold-5) |
| **Best Recall (Fail)** | 0.567 (Phase 2: SMOTE+RUS) |
| **High-Confidence Root Causes** | 6 sensors (SHAP ∩ t-test cross-validation) |
| **Estimated Business Impact** | ~$4.06M USD/year savings |
| **Dataset** | UCI SECOM — 1,567 samples × 590 features, 6.64% Fail |

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-3776AB?style=flat&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.1.4-FF6600?style=flat)
![LightGBM](https://img.shields.io/badge/LightGBM-4.6.0-02B875?style=flat)
![SHAP](https://img.shields.io/badge/SHAP-0.51.0-FF0000?style=flat)
![Optuna](https://img.shields.io/badge/Optuna-4.9.0-6366F1?style=flat)
![imbalanced-learn](https://img.shields.io/badge/imbalanced--learn-0.11-F7931E?style=flat)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3-F7931E?style=flat&logo=scikit-learn&logoColor=white)

---

## Project Overview

The core goal of this project is to **predict** wafer failures **before** the process completes — using only in-process sensor data — to prevent waste in downstream stages. At the same time, it quantifies "which process variables contribute to defects" through statistical and model-based methods, providing **root-cause insights** that process engineers can act on.

The UCI SECOM dataset is a public benchmark collected from a real semiconductor manufacturing line: 1,567 samples, 590 anonymized sensor features, and a 6.64% fail rate (1:14.1 class imbalance). Three challenges coexist — high dimensionality relative to sample size (p ≫ n), severe class imbalance, and a weak signal-to-noise ratio — mirroring real-world fab conditions.

The project is structured as a **5-phase incremental pipeline** from EDA through SHAP explainability, independently measuring the performance contribution of each added technique. Key methodological contributions include data leakage prevention via `imblearn.Pipeline`, empirical comparison of CV strategies, and identification of high-confidence defect factors through SHAP × t-test cross-validation.

---

## Pipeline Architecture

![5-Phase Pipeline](reports/figures/00_pipeline_overview.png)

---

## Phase-by-Phase Results

| Phase | Configuration | PR-AUC | Recall (Fail) | Key Contribution |
|-------|--------------|--------|---------------|-----------------|
| Phase 1 | RF, no imbalance handling | 0.180 | 0.000 | Baseline — majority-class bias confirmed |
| Phase 2 | RF + SMOTE+RUS | 0.166 | **0.567** | Recall 0→0.567 via `imblearn.Pipeline` |
| Phase 3 | RF + SMOTE+RUS + Anomaly Scores | 0.180 | 0.346 | PR-AUC +0.014; IF+LOF feature augmentation |
| Phase 4 (default) | XGBoost + top80\_model features | 0.207 | 0.484 | Feature selection: single largest PR-AUC gain |
| **Phase 4 (tuned)** | **XGBoost + Optuna (50 trials)** | **0.227** | 0.558 | Best PR-AUC; +26.1% over Phase 3 |
| Phase 4 (tuned) | LightGBM + Optuna (50 trials) | 0.218 | 0.490 | Runner-up |
| Target | — | ≥ 0.40 | ≥ 0.70 | Not achieved (dataset signal ceiling) |

---

## Key Findings

- **sensor_60 dominates**: Mean |SHAP| = 0.8105 (1.87× the runner-up). Cohen's d = 0.591 — highest effect size in statistical testing as well. Two independent methods converge on the same feature.

- **Feature selection > everything else**: Switching `all_446 → top80_model` alone achieved PR-AUC +0.028 — twice the gain from anomaly score features (+0.014). Removing 82% of noisy features was the single most impactful intervention.

- **Nonlinear interactions detected**: 14 of the SHAP Top-20 features (70%) were not significant in univariate t-test (p > 0.05), demonstrating nonlinear interaction effects between process variables that simple SPC charts cannot capture.

- **Business impact estimated at ~$4.06M USD/year**: Assumes 120k wafers/year, $2,000/wafer, Recall = 0.567, and 45% cost recovery rate. Scales to $10.15M–$30.44M for leading-edge nodes (3nm–5nm).

---

## SHAP Analysis

![SHAP Beeswarm](reports/figures/23_p5_shap_beeswarm.png)

*SHAP Beeswarm Plot — Top-20 features. sensor_60's dominant contribution and directionality (higher values drive Fail probability up) are clearly visible.*

---

## Technical Report

Full academic report covering experimental design, formulations, result analysis, and limitations:

**[📄 reports/technical_report.pdf](reports/technical_report.pdf)**

---

## Repository Structure

```
semiconductor-yield-prediction/
├── README.md
├── requirements.txt
├── .gitignore
├── scripts/
│   └── download_data.py          # Step 1: Download UCI SECOM dataset
├── notebooks/
│   ├── 01_eda/
│   │   └── eda_secom.py          # Exploratory data analysis
│   ├── phase1_baseline/
│   │   ├── 01_preprocess_compare.py
│   │   └── 02_baseline_models.py
│   ├── phase2_imbalance/
│   │   └── 01_imbalance_experiments.py
│   ├── phase3_unsupervised/
│   │   ├── 00_prelim_check.py
│   │   └── 01_anomaly_detection.py
│   ├── phase4_boosting/
│   │   └── 01_boosting_experiments.py
│   └── phase5_explainability/
│       └── 01_shap_analysis.py
├── src/
│   └── data/
│       └── preprocessing.py      # Shared preprocessing utilities
└── reports/
    ├── technical_report.pdf      # Full academic report
    ├── technical_report.md       # Report source
    ├── figures/                  # All generated visualizations (27 PNGs)
    ├── eda_findings.md
    ├── phase{1-5}_results.md     # Per-phase result summaries
    └── phase{1,2,4}_results_table.csv
```

> **Note**: `data/` is excluded from this repository. Raw data is publicly available from UCI and can be downloaded automatically (see Setup below).

---

## Setup & Reproduction

```bash
# 1. Clone the repository
git clone https://github.com/yhwang55/semiconductor-yield-prediction.git
cd semiconductor-yield-prediction

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download UCI SECOM dataset
python scripts/download_data.py

# 4. Run the pipeline in order
python notebooks/01_eda/eda_secom.py
python notebooks/phase1_baseline/01_preprocess_compare.py
python notebooks/phase1_baseline/02_baseline_models.py
python notebooks/phase2_imbalance/01_imbalance_experiments.py
python notebooks/phase3_unsupervised/00_prelim_check.py
python notebooks/phase3_unsupervised/01_anomaly_detection.py
python notebooks/phase4_boosting/01_boosting_experiments.py
python notebooks/phase5_explainability/01_shap_analysis.py
```

All figures and result `.md` files are saved to `reports/` automatically.

---

## Author

**Yoon Hwang**
University of Wisconsin-Madison | Data Science & Economics & Information Science

- Email: yoondbs3@gmail.com
- GitHub: [github.com/yhwang55](https://github.com/yhwang55)
- LinkedIn: [linkedin.com/in/yoon-hwang](https://linkedin.com/in/yoon-hwang)
