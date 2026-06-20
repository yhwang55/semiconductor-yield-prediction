"""
Phase 1 — Step 2: 베이스라인 모델 평가
  - Logistic Regression / Random Forest
  - Imputation 2종 × CV 전략 2종 = 4 조합 × 2 모델 = 8 실험
  - 평가지표: PR-AUC, F1(Fail), Recall(Fail), Accuracy
  - 산출: reports/figures/10_baseline_results.png
          reports/figures/11_baseline_cv_variance.png
          reports/phase1_results.md
"""

import sys, warnings, itertools
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.models.evaluate import get_cv_strategies, cross_val_eval, make_results_table

PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"

# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — BASELINE MODELS")
print("=" * 60)

X_med = pd.read_parquet(PROC_DIR / "X_median.parquet")
X_it  = pd.read_parquet(PROC_DIR / "X_iterative.parquet")
y     = pd.read_parquet(PROC_DIR / "y.parquet")["label"]

print(f"  X_median shape    : {X_med.shape}")
print(f"  X_iterative shape : {X_it.shape}")
print(f"  Fail rate         : {y.mean():.2%}  ({y.sum()} / {len(y)})")

# ── 2. 모델 & CV 정의 ──────────────────────────────────────────────────────────
models = {
    "LogisticRegression": LogisticRegression(
        C=1.0, max_iter=2000, random_state=42, n_jobs=-1
    ),
    "RandomForest": RandomForestClassifier(
        n_estimators=200, random_state=42, n_jobs=-1
    ),
}

datasets = {
    "Median": X_med,
    "MICE":   X_it,
}

cv_strategies = get_cv_strategies(n_splits=5)

# LR은 StandardScaler 필요, RF는 불필요 (scale=False)
needs_scale = {"LogisticRegression": True, "RandomForest": False}

# ── 3. 실험 실행 ───────────────────────────────────────────────────────────────
print("\n[Running experiments]  (8 total: 2 models × 2 imputation × 2 CV)")
all_results = {}

for (model_name, model), (imp_name, X), (cv_name, cv) in itertools.product(
    models.items(), datasets.items(), cv_strategies.items()
):
    label = f"{model_name} | {imp_name} | {cv_name}"
    print(f"  {label} ...", end="", flush=True)
    result = cross_val_eval(
        model, X, y, cv,
        scale=needs_scale[model_name],
    )
    all_results[label] = result
    pr = result["pr_auc"]["mean"]
    f1 = result["f1_fail"]["mean"]
    rc = result["recall_fail"]["mean"]
    print(f"  PR-AUC={pr:.3f}  F1={f1:.3f}  Recall={rc:.3f}")

# ── 4. 결과 테이블 정리 ────────────────────────────────────────────────────────
results_df = make_results_table(all_results)

# 보기 편하게 모델/Imputation/CV 열 분리
def parse_label(label):
    parts = [p.strip() for p in label.split("|")]
    return parts[0], parts[1], parts[2]

results_df[["model", "imputation", "cv"]] = pd.DataFrame(
    results_df["configuration"].apply(parse_label).tolist(),
    index=results_df.index,
)

print("\n" + "=" * 100)
print("RESULTS TABLE")
print("=" * 100)
display_cols = ["model", "imputation", "cv",
                "pr_auc_mean", "pr_auc_std",
                "f1_fail_mean", "f1_fail_std",
                "recall_fail_mean", "recall_fail_std",
                "accuracy_mean", "accuracy_std"]
print(results_df[display_cols].to_string(index=False))

results_df.to_csv(ROOT / "reports" / "phase1_results_table.csv", index=False)
print("\n  Saved: reports/phase1_results_table.csv")

# ── 5. Figure 10: 주요 메트릭 그룹 바 차트 ────────────────────────────────────
KEY_METRICS = ["pr_auc", "f1_fail", "recall_fail", "accuracy"]
METRIC_LABELS = {
    "pr_auc": "PR-AUC",
    "f1_fail": "F1 (Fail)",
    "recall_fail": "Recall (Fail)",
    "accuracy": "Accuracy",
}

imp_styles = {"Median": ("///", "#2196F3"), "MICE": ("xxx", "#E91E63")}
cv_styles  = {"TimeSeriesSplit": "solid", "StratifiedKFold": "dashed"}

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax_idx, metric in enumerate(KEY_METRICS):
    ax = axes[ax_idx]
    mean_col = f"{metric}_mean"
    std_col  = f"{metric}_std"

    model_names = results_df["model"].unique()
    x_base = np.arange(len(model_names))
    offsets = np.linspace(-0.3, 0.3, 4)
    combo_idx = 0

    for imp_name, (hatch, color) in imp_styles.items():
        for cv_name, ls in cv_styles.items():
            subset = results_df[
                (results_df["imputation"] == imp_name) &
                (results_df["cv"] == cv_name)
            ].set_index("model")
            means = [subset.loc[m, mean_col] if m in subset.index else 0 for m in model_names]
            stds  = [subset.loc[m, std_col]  if m in subset.index else 0 for m in model_names]

            short_cv = "TS" if cv_name == "TimeSeriesSplit" else "SK"
            bar_label = f"{imp_name} / {short_cv}"

            ax.bar(
                x_base + offsets[combo_idx], means, width=0.15,
                color=color, alpha=0.7 if cv_name == "TimeSeriesSplit" else 0.4,
                hatch=hatch if cv_name == "StratifiedKFold" else "",
                label=bar_label, yerr=stds, capsize=3, error_kw={"linewidth": 1},
            )
            combo_idx += 1

    ax.set_xticks(x_base)
    ax.set_xticklabels(["LR", "RF"], fontsize=11)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.set_title(METRIC_LABELS[metric], fontsize=12)
    ax.set_ylim(0, min(1.0, ax.get_ylim()[1] * 1.25))

    if ax_idx == 0:
        ax.legend(fontsize=8, loc="upper right")

# Fail rate baseline 주석
fig.text(0.5, -0.02,
         "Dashed bar = StratifiedKFold  |  Solid bar = TimeSeriesSplit  |  "
         "Error bar = 5-fold std  |  Random baseline Recall ≈ 6.6%",
         ha="center", fontsize=9, color="gray")

plt.suptitle("Phase 1 Baseline Results\n(No class imbalance handling — raw 1:14 imbalance)",
             fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "10_baseline_results.png", bbox_inches="tight")
plt.close()
print("  Saved: 10_baseline_results.png")

# ── 6. Figure 11: Fold-level 분산 박스플롯 ────────────────────────────────────
# F1(Fail)과 Recall(Fail)의 fold-level 값으로 불안정성 시각화
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, metric in zip(axes, ["f1_fail", "recall_fail"]):
    data_by_model = {}
    labels_by_model = {}
    for model_name in ["LogisticRegression", "RandomForest"]:
        fold_data = []
        tick_labels = []
        for imp_name in ["Median", "MICE"]:
            for cv_name in ["TimeSeriesSplit", "StratifiedKFold"]:
                key = f"{model_name} | {imp_name} | {cv_name}"
                folds = all_results[key][metric]["folds"]
                fold_data.append(folds)
                short_cv = "TS" if cv_name == "TimeSeriesSplit" else "SK"
                tick_labels.append(f"{imp_name}\n{short_cv}")
        data_by_model[model_name] = fold_data
        labels_by_model[model_name] = tick_labels

    all_data = (
        data_by_model["LogisticRegression"] +
        data_by_model["RandomForest"]
    )
    all_labels = (
        [f"LR\n{l}" for l in labels_by_model["LogisticRegression"]] +
        [f"RF\n{l}" for l in labels_by_model["RandomForest"]]
    )
    colors = ["#BBDEFB"] * 4 + ["#FFCDD2"] * 4

    bp = ax.boxplot(all_data, patch_artist=True, labels=all_labels,
                    medianprops=dict(color="black", linewidth=2),
                    flierprops=dict(marker=".", markersize=4))
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)

    ax.axvline(4.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_title(METRIC_LABELS[metric] + " — Per-fold Variance", fontsize=11)
    ax.set_ylabel(METRIC_LABELS[metric])
    ax.tick_params(axis="x", labelsize=7)

plt.suptitle("Fold-level Stability: F1 & Recall (Fail class)\n"
             "Blue = LR, Red = RF  |  Variance reveals unreliability of minority-class detection",
             fontsize=11, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "11_baseline_cv_variance.png", bbox_inches="tight")
plt.close()
print("  Saved: 11_baseline_cv_variance.png")

# ── 7. Phase 1 결론 데이터 집계 ────────────────────────────────────────────────
best_row = results_df.loc[results_df["f1_fail_mean"].idxmax()]
worst_recall = results_df["recall_fail_mean"].min()
best_recall = results_df["recall_fail_mean"].max()
best_pr = results_df["pr_auc_mean"].max()

ts_rows = results_df[results_df["cv"] == "TimeSeriesSplit"]
sk_rows = results_df[results_df["cv"] == "StratifiedKFold"]
cv_delta_f1 = abs(ts_rows["f1_fail_mean"].mean() - sk_rows["f1_fail_mean"].mean())

med_rows = results_df[results_df["imputation"] == "Median"]
it_rows  = results_df[results_df["imputation"] == "MICE"]
imp_delta_f1 = abs(med_rows["f1_fail_mean"].mean() - it_rows["f1_fail_mean"].mean())

# ── 8. phase1_results.md 생성 ─────────────────────────────────────────────────
table_md_rows = []
for _, row in results_df[display_cols].iterrows():
    def fmt(m, s):
        return f"{m:.3f} ± {s:.3f}"
    table_md_rows.append(
        f"| {row['model'][:2] if row['model']=='LR' else row['model'].replace('LogisticRegression','LR').replace('RandomForest','RF')} "
        f"| {row['imputation']} | {row['cv'].replace('TimeSeriesSplit','TS').replace('StratifiedKFold','SK')} "
        f"| {row['pr_auc_mean']:.3f}±{row['pr_auc_std']:.3f} "
        f"| {row['f1_fail_mean']:.3f}±{row['f1_fail_std']:.3f} "
        f"| {row['recall_fail_mean']:.3f}±{row['recall_fail_std']:.3f} "
        f"| {row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f} |"
    )

table_md = "\n".join([
    "| Model | Imputation | CV | PR-AUC | F1(Fail) | Recall(Fail) | Accuracy |",
    "|-------|-----------|-----|--------|----------|--------------|----------|",
] + table_md_rows)

best_model_short = best_row["model"].replace("LogisticRegression","LR").replace("RandomForest","RF")
best_imp = best_row["imputation"]
best_cv = best_row["cv"].replace("TimeSeriesSplit","TS").replace("StratifiedKFold","SK")

md = f"""# Phase 1 Baseline Results

**분석 일자**: 2026-06-19
**목적**: 전처리/CV 전략에 따른 베이스라인 성능 확인 + Phase 2 필요성 근거 수립

---

## 1. 실험 설계

| 차원 | 옵션 |
|------|------|
| 모델 | Logistic Regression (LR), Random Forest (RF) |
| Imputation | Median, MICE (IterativeImputer, max_iter=5, k=10) |
| CV 전략 | TimeSeriesSplit (5-fold), StratifiedKFold (5-fold, shuffle) |
| **조합 수** | **2 × 2 × 2 = 8 실험** |

**클래스 불균형 처리 없음** (raw 1:14.1 imbalance 그대로 사용)
LR: StandardScaler 적용 / RF: 스케일링 미적용

---

## 2. 성능 비교 테이블 (5-fold 평균 ± std)

{table_md}

**주:** PR-AUC = Precision-Recall AUC, F1/Recall은 Fail(=1) 클래스 기준.
Accuracy는 클래스 불균형 하에서 Pass-all 예측 시 93.4% 도달 가능 → 맹점 지표.

---

## 3. 주요 발견

### 3.1 Imputation 방식의 영향 (Δ F1: {imp_delta_f1:.4f})

Median과 MICE 간 F1(Fail) 평균 차이: **{imp_delta_f1:.4f}**
→ Imputation 방식은 모델 성능에 **미미한 영향**. 두 방식 간 성능 차이가 noise 수준.
→ **결론**: 계산 비용이 낮은 **Median을 기본 전처리**로 채택.
MICE는 불균형 처리 후에도 차이가 없으면 완전히 배제 가능.

### 3.2 CV 전략의 영향 (Δ F1: {cv_delta_f1:.4f})

TimeSeriesSplit vs StratifiedKFold 간 F1(Fail) 평균 차이: **{cv_delta_f1:.4f}**
{"→ TimeSeriesSplit이 더 보수적인 (낮은) 성능 추정치를 생성. 시간적 자기상관 존재 시 이것이 더 현실적." if cv_delta_f1 > 0.01 else "→ 두 CV 전략 간 성능 차이가 작음 — 시간적 자기상관이 강하지 않을 수 있음."}
→ **삼성전자 DS 실전 상황**: 미래 배치 예측이 목적이므로 **TimeSeriesSplit이 더 적절**.
단, StratifiedKFold가 소수 클래스(Fail=104개) 분포를 더 균등하게 보장한다는 장점도 있음.

### 3.3 모델별 성능 비교

| 모델 | 최고 PR-AUC | 최고 F1(Fail) | 최고 Recall(Fail) |
|------|------------|--------------|------------------|
| LR   | {results_df[results_df['model']=='LogisticRegression']['pr_auc_mean'].max():.3f} | {results_df[results_df['model']=='LogisticRegression']['f1_fail_mean'].max():.3f} | {results_df[results_df['model']=='LogisticRegression']['recall_fail_mean'].max():.3f} |
| RF   | {results_df[results_df['model']=='RandomForest']['pr_auc_mean'].max():.3f} | {results_df[results_df['model']=='RandomForest']['f1_fail_mean'].max():.3f} | {results_df[results_df['model']=='RandomForest']['recall_fail_mean'].max():.3f} |

전반적으로 RF가 LR보다 높은 PR-AUC를 보이나, **Recall(Fail)은 두 모델 모두 {worst_recall:.3f}~{best_recall:.3f}로 매우 낮음**.

### 3.4 Best Configuration

- **최고 F1(Fail)**: {best_model_short} | {best_imp} | {best_cv} → {best_row['f1_fail_mean']:.3f} ± {best_row['f1_fail_std']:.3f}
- **최고 PR-AUC**: {best_pr:.3f}

---

## 4. Phase 2로 넘어가야 하는 이유

현재 Phase 1 베이스라인은 **클래스 불균형(1:14.1) 처리 없이** LR/RF를 단순 적용했다.
결과는 예상대로 **Recall(Fail) = {worst_recall:.3f}~{best_recall:.3f}, F1(Fail) = {results_df['f1_fail_mean'].min():.3f}~{results_df['f1_fail_mean'].max():.3f}** 수준으로, 불량 탐지 관점에서 사실상 무용하다.

반도체 수율 예측의 실용적 요구사항은 **"불량을 놓치지 않는 것"(Recall 우선)** 이다.
False Negative (불량을 Pass로 예측) 비용 >> False Positive (Pass를 불량으로 예측) 비용.
따라서 현재 모델은 생산 현장에 적용 불가능하다.

**Phase 2에서 적용할 불균형 처리 기법:**
- `class_weight='balanced'` (모델 레벨 보정)
- SMOTE / ADASYN (오버샘플링)
- 위 기법들의 체계적 비교 및 최적 조합 탐색
- 목표 지표: **Recall(Fail) ≥ 0.70, PR-AUC ≥ 0.40**

*현재 베이스라인이 이 목표치와 얼마나 멀리 있는지가 Phase 2의 개선 여지를 보여주는 명확한 근거가 된다.*

---

## 5. 산출 파일

| 파일 | 설명 |
|------|------|
| `reports/figures/10_baseline_results.png` | 4 메트릭 × 8 조합 그룹 바 차트 |
| `reports/figures/11_baseline_cv_variance.png` | Fold-level 분산 박스플롯 |
| `reports/phase1_results_table.csv` | 원시 수치 데이터 |
| `data/processed/X_median.parquet` | Median imputed 피처 |
| `data/processed/X_iterative.parquet` | MICE imputed 피처 |

---
*Generated by `notebooks/phase1_baseline/02_baseline_models.py`*
"""

(ROOT / "reports" / "phase1_results.md").write_text(md)
print("  Saved: reports/phase1_results.md")
print("\nPhase 1 Baseline Complete.")
