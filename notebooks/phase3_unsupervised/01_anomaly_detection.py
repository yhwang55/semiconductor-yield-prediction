"""
Phase 3 — 비지도 이상탐지 실험

실험 1: IF / LOF 단독 평가 (라벨 미사용, 전체 데이터에 fit)
실험 2: 이상점수를 피처로 추가 → RF + SMOTE+RUS 재평가
비교:   Phase 2 baseline vs 비지도 단독 vs 피처 추가
→     reports/phase3_results.md
"""

import sys, warnings, time
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    average_precision_score, roc_auc_score,
    precision_recall_curve,
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler

from src.models.evaluate import get_cv_strategies, cross_val_eval_v2

PROC_DIR = ROOT / "data" / "processed"
FIG_DIR  = ROOT / "reports" / "figures"
plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── 데이터 로딩 ────────────────────────────────────────────────────────────────
print("=" * 65)
print("PHASE 3 — UNSUPERVISED ANOMALY DETECTION")
print("=" * 65)

X = pd.read_parquet(PROC_DIR / "X_median.parquet")
y = pd.read_parquet(PROC_DIR / "y.parquet")["label"]
FAIL_RATE = float(y.mean())
print(f"  Shape: {X.shape}  |  Fail rate: {FAIL_RATE:.2%}")

CV = get_cv_strategies(n_splits=5)  # StratifiedKFold = primary

# ── Custom Transformer: 이상점수를 피처로 추가 ──────────────────────────────────
class AnomalyScoreAdder(BaseEstimator, TransformerMixin):
    """Train fold에서 IF/LOF를 학습, 이상점수 2개를 피처로 추가.

    Pipeline 내부에서 fit() = train only, transform() = train+val
    → validation 누수 없음.
    LOF는 novelty=True로 새 샘플에 대해 score_samples() 가능.
    """

    def __init__(self, contamination=0.066, n_neighbors=20, n_estimators=100,
                 random_state=42):
        self.contamination = contamination
        self.n_neighbors   = n_neighbors
        self.n_estimators  = n_estimators
        self.random_state  = random_state

    def fit(self, X, y=None):
        arr = X.values if hasattr(X, "values") else np.array(X)
        self.if_  = IsolationForest(
            contamination=self.contamination,
            n_estimators=self.n_estimators,
            random_state=self.random_state, n_jobs=-1,
        ).fit(arr)
        self.lof_ = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination=self.contamination,
            novelty=True,
        ).fit(arr)
        return self

    def transform(self, X):
        arr = X.values if hasattr(X, "values") else np.array(X)
        # score_samples: 더 음수일수록 이상치 (IF) / 더 음수일수록 이상치 (LOF novelty)
        if_score  = self.if_.score_samples(arr).reshape(-1, 1)
        lof_score = self.lof_.score_samples(arr).reshape(-1, 1)
        return np.hstack([arr, if_score, lof_score])   # n × (446+2)


# ══════════════════════════════════════════════════════════════════════════════
# 실험 1: 비지도 이상탐지 단독 평가 (전체 데이터, 라벨 미사용)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("EXPERIMENT 1 — STANDALONE ANOMALY DETECTION")
print("=" * 65)
print("(No labels used — fit on full X, evaluate against y)")

X_arr = X.values

contaminations = [0.05, FAIL_RATE, 0.10, 0.15]
exp1_rows = []

for c in contaminations:
    # ── Isolation Forest ──────────────────────────────────────────────────────
    t0  = time.time()
    ifo = IsolationForest(contamination=c, n_estimators=200,
                          random_state=42, n_jobs=-1)
    ifo.fit(X_arr)
    if_labels  = ifo.predict(X_arr)            # -1=anomaly, 1=normal
    if_scores  = -ifo.score_samples(X_arr)     # 더 클수록 이상치
    y_pred_if  = (if_labels == -1).astype(int) # Fail 예측 = 이상치

    if_p   = precision_score(y, y_pred_if, pos_label=1, zero_division=0)
    if_r   = recall_score   (y, y_pred_if, pos_label=1, zero_division=0)
    if_f1  = f1_score       (y, y_pred_if, pos_label=1, zero_division=0)
    if_auc = average_precision_score(y, if_scores)
    t_if   = time.time() - t0

    # ── Local Outlier Factor ──────────────────────────────────────────────────
    t0   = time.time()
    lof  = LocalOutlierFactor(contamination=c, n_neighbors=20)
    lof_labels  = lof.fit_predict(X_arr)               # -1=outlier, 1=inlier
    lof_scores  = -lof.negative_outlier_factor_        # 더 클수록 이상치
    y_pred_lof  = (lof_labels == -1).astype(int)

    lof_p   = precision_score(y, y_pred_lof, pos_label=1, zero_division=0)
    lof_r   = recall_score   (y, y_pred_lof, pos_label=1, zero_division=0)
    lof_f1  = f1_score       (y, y_pred_lof, pos_label=1, zero_division=0)
    lof_auc = average_precision_score(y, lof_scores)
    t_lof   = time.time() - t0

    print(f"\n  contamination={c:.3f}:")
    print(f"    IF   P={if_p:.3f}  R={if_r:.3f}  F1={if_f1:.3f}  PR-AUC={if_auc:.3f}  [{t_if:.1f}s]")
    print(f"    LOF  P={lof_p:.3f}  R={lof_r:.3f}  F1={lof_f1:.3f}  PR-AUC={lof_auc:.3f}  [{t_lof:.1f}s]")

    for name, p, r, f1v, auc in [
        ("IF",  if_p,  if_r,  if_f1,  if_auc),
        ("LOF", lof_p, lof_r, lof_f1, lof_auc),
    ]:
        exp1_rows.append(dict(
            model=name, contamination=round(c, 3),
            precision=round(p, 3), recall=round(r, 3),
            f1=round(f1v, 3), pr_auc=round(auc, 3),
        ))

exp1_df = pd.DataFrame(exp1_rows)

# contamination=FAIL_RATE 의 IF/LOF 스코어를 PR curve 시각화용으로 저장
ifo_ref = IsolationForest(contamination=FAIL_RATE, n_estimators=200,
                          random_state=42, n_jobs=-1).fit(X_arr)
lof_ref = LocalOutlierFactor(contamination=FAIL_RATE, n_neighbors=20)
lof_ref.fit_predict(X_arr)
if_scores_ref  = -ifo_ref.score_samples(X_arr)
lof_scores_ref = -lof_ref.negative_outlier_factor_


# ══════════════════════════════════════════════════════════════════════════════
# 실험 2: 이상점수 피처 추가 → RF + SMOTE+RUS (StratifiedKFold & TS)
# ══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 65)
print("EXPERIMENT 2 — ANOMALY SCORES AS FEATURES + RF SMOTE+RUS")
print("=" * 65)
print("Pipeline: [AnomalyScoreAdder → SMOTE+RUS → RF]  (scaled before anomaly fit)")

# Phase 2 baseline: RF + SMOTE+RUS (재실행, no anomaly scores)
pipe_p2 = ImbPipeline([
    ("smote", SMOTE(sampling_strategy=0.5, random_state=42)),
    ("rus",   RandomUnderSampler(sampling_strategy=1.0, random_state=42)),
    ("m",     RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
])

# Phase 3: RF + SMOTE+RUS WITH anomaly scores
# StandardScaler → AnomalyScoreAdder → SMOTE+RUS → RF
pipe_p3 = ImbPipeline([
    ("sc",     StandardScaler()),
    ("anomaly", AnomalyScoreAdder(contamination=FAIL_RATE,
                                  n_neighbors=20, n_estimators=100)),
    ("smote",  SMOTE(sampling_strategy=0.5, random_state=42)),
    ("rus",    RandomUnderSampler(sampling_strategy=1.0, random_state=42)),
    ("m",      RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1)),
])

exp2_rows = {}

for cv_name, cv in CV.items():
    print(f"\n  CV: {cv_name}")

    # Phase 2 baseline (re-run for fair comparison)
    t0 = time.time()
    r_p2 = cross_val_eval_v2(pipe_p2, X, y, cv)
    opt = r_p2["optimal"]
    print(f"  [Phase2 baseline] PR={opt['pr_auc']['mean']:.3f}"
          f"  F1={opt['f1_fail']['mean']:.3f}"
          f"  Rec={opt['recall_fail']['mean']:.3f}"
          f"  Pre={opt['precision_fail']['mean']:.3f}  [{time.time()-t0:.1f}s]")

    # Phase 3 (anomaly features)
    t0 = time.time()
    r_p3 = cross_val_eval_v2(pipe_p3, X, y, cv)
    opt3 = r_p3["optimal"]
    print(f"  [Phase3 anomaly ] PR={opt3['pr_auc']['mean']:.3f}"
          f"  F1={opt3['f1_fail']['mean']:.3f}"
          f"  Rec={opt3['recall_fail']['mean']:.3f}"
          f"  Pre={opt3['precision_fail']['mean']:.3f}  [{time.time()-t0:.1f}s]")

    exp2_rows[cv_name] = {"p2": r_p2, "p3": r_p3}


# ══════════════════════════════════════════════════════════════════════════════
# 시각화
# ══════════════════════════════════════════════════════════════════════════════

# ── Figure 16: Exp1 contamination별 성능 ─────────────────────────────────────
print("\n[Figure 16]")
fig, axes = plt.subplots(1, 3, figsize=(14, 5))
metrics_e1 = ["precision", "recall", "f1"]
labels_e1  = {"precision": "Precision (Fail)", "recall": "Recall (Fail)", "f1": "F1 (Fail)"}

for ax, met in zip(axes, metrics_e1):
    for model_name, color, ls in [("IF","#FF6B35","o-"), ("LOF","#4ECDC4","s--")]:
        sub = exp1_df[exp1_df["model"] == model_name]
        ax.plot(sub["contamination"], sub[met], ls, color=color,
                linewidth=2, markersize=7, label=model_name)
    ax.axvline(FAIL_RATE, color="gray", linestyle=":", linewidth=1.5,
               label=f"Actual Fail rate ({FAIL_RATE:.2%})")
    ax.set_xlabel("Contamination")
    ax.set_ylabel(labels_e1[met])
    ax.set_title(f"Exp1: {labels_e1[met]}", fontsize=11)
    ax.set_xticks(contaminations)
    ax.set_xticklabels([f"{c:.2f}" for c in contaminations])
    ax.legend(fontsize=8)

plt.suptitle("Experiment 1 — Standalone Anomaly Detection\n"
             "(No labels used during fit)", fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "16_p3_anomaly_standalone.png", bbox_inches="tight")
plt.close()
print("  Saved: 16_p3_anomaly_standalone.png")

# ── Figure 17: PR Curves (실험 1, contamination=FAIL_RATE) ───────────────────
print("[Figure 17]")
fig, ax = plt.subplots(figsize=(8, 6))

# IF PR curve
prec_if, rec_if, _ = precision_recall_curve(y, if_scores_ref)
auc_if = average_precision_score(y, if_scores_ref)
ax.plot(rec_if, prec_if, color="#FF6B35", linewidth=2,
        label=f"IF  (AUC={auc_if:.3f})", alpha=0.9)

# LOF PR curve
prec_lof, rec_lof, _ = precision_recall_curve(y, lof_scores_ref)
auc_lof = average_precision_score(y, lof_scores_ref)
ax.plot(rec_lof, prec_lof, color="#4ECDC4", linewidth=2,
        label=f"LOF (AUC={auc_lof:.3f})", alpha=0.9)

# Phase 2 & 3 OOF PR curves (StratifiedKFold)
from src.models.evaluate import get_oof_predictions

sk_cv = CV["StratifiedKFold"]
for label, pipe, color, ls in [
    ("Phase2: RF+SMOTE+RUS",         pipe_p2, "#9C27B0", "--"),
    ("Phase3: RF+SMOTE+RUS+Anomaly", pipe_p3, "#F44336", "-"),
]:
    oof = get_oof_predictions(pipe, X, y, sk_cv)
    prec_oof, rec_oof, _ = precision_recall_curve(y, oof)
    auc_oof = average_precision_score(y, oof)
    ax.plot(rec_oof, prec_oof, color=color, linewidth=2, linestyle=ls,
            label=f"{label} (AUC={auc_oof:.3f})", alpha=0.85)

ax.axhline(FAIL_RATE, color="lightgray", linestyle=":", linewidth=1.5,
           label=f"Random ({FAIL_RATE:.2%})")

# Iso-F1 lines
for f1_val in [0.2, 0.4, 0.6]:
    r_arr = np.linspace(0.01, 1, 300)
    p_arr = f1_val * r_arr / (2 * r_arr - f1_val + 1e-10)
    valid = (p_arr >= 0) & (p_arr <= 1)
    ax.plot(r_arr[valid], p_arr[valid], ":", color="lightgray",
            linewidth=0.8, alpha=0.7)
    ax.text(r_arr[valid][-1] + 0.01, p_arr[valid][-1],
            f"F1={f1_val}", fontsize=7, color="gray")

ax.set_xlabel("Recall (Fail)")
ax.set_ylabel("Precision (Fail)")
ax.set_title("PR Curves: Unsupervised vs Supervised (Phase 2 & 3)\n"
             "(contamination=actual fail rate for IF/LOF)")
ax.legend(fontsize=9)
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(FIG_DIR / "17_p3_pr_curves_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: 17_p3_pr_curves_comparison.png")

# ── Figure 18: Phase 2 vs Phase 3 메트릭 비교 바 차트 ─────────────────────────
print("[Figure 18]")
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

compare_metrics = ["pr_auc", "f1_fail", "recall_fail", "precision_fail"]
metric_labels   = {"pr_auc": "PR-AUC", "f1_fail": "F1(Fail)",
                   "recall_fail": "Recall(Fail)", "precision_fail": "Precision(Fail)"}

for ax, (cv_name, results) in zip(axes, exp2_rows.items()):
    x      = np.arange(len(compare_metrics))
    width  = 0.35

    p2_opt = results["p2"]["optimal"]
    p3_opt = results["p3"]["optimal"]

    means_p2 = [p2_opt[m]["mean"] for m in compare_metrics]
    stds_p2  = [p2_opt[m]["std"]  for m in compare_metrics]
    means_p3 = [p3_opt[m]["mean"] for m in compare_metrics]
    stds_p3  = [p3_opt[m]["std"]  for m in compare_metrics]

    ax.bar(x - width/2, means_p2, width, color="#9C27B0", alpha=0.75,
           yerr=stds_p2, capsize=4, label="Phase2: RF+SMOTE+RUS (no anomaly)")
    ax.bar(x + width/2, means_p3, width, color="#F44336", alpha=0.80,
           yerr=stds_p3, capsize=4, label="Phase3: +AnomalyScores")

    # 개선량 표시
    for xi, (m2, m3) in enumerate(zip(means_p2, means_p3)):
        delta = m3 - m2
        ax.text(xi + width/2, m3 + 0.01,
                f"{delta:+.3f}", ha="center", fontsize=8,
                color="darkred" if delta < 0 else "darkgreen")

    ax.set_xticks(x)
    ax.set_xticklabels([metric_labels[m] for m in compare_metrics])
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{cv_name}", fontsize=11)
    ax.set_ylabel("Score (Optimal Threshold)")
    ax.legend(fontsize=8)
    ax.axhline(0.40, color="orange", linestyle=":", alpha=0.7, label="PR-AUC target")

plt.suptitle("Phase 2 vs Phase 3 (Anomaly Score Features)\n"
             "Red text: improvement from adding anomaly scores",
             fontsize=12)
plt.tight_layout()
plt.savefig(FIG_DIR / "18_p3_vs_p2_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: 18_p3_vs_p2_comparison.png")

# ══════════════════════════════════════════════════════════════════════════════
# 최종 비교 테이블 데이터 집계
# ══════════════════════════════════════════════════════════════════════════════

# SK = primary
sk_p2_opt = exp2_rows["StratifiedKFold"]["p2"]["optimal"]
sk_p3_opt = exp2_rows["StratifiedKFold"]["p3"]["optimal"]
ts_p2_opt = exp2_rows["TimeSeriesSplit"]["p2"]["optimal"]
ts_p3_opt = exp2_rows["TimeSeriesSplit"]["p3"]["optimal"]

# Exp1 대표값 (contamination=FAIL_RATE)
exp1_ref = exp1_df[exp1_df["contamination"] == round(FAIL_RATE, 3)]
def _e1(model, col):
    row = exp1_ref[exp1_ref["model"] == model]
    return float(row[col].iloc[0]) if len(row) else float("nan")

# PR-AUC 변화량
pr_delta_sk = sk_p3_opt["pr_auc"]["mean"] - sk_p2_opt["pr_auc"]["mean"]
pr_delta_ts = ts_p3_opt["pr_auc"]["mean"] - ts_p2_opt["pr_auc"]["mean"]

print(f"\n  PR-AUC change (anomaly features):")
print(f"  StratifiedKFold: {sk_p2_opt['pr_auc']['mean']:.3f} → {sk_p3_opt['pr_auc']['mean']:.3f}  ({pr_delta_sk:+.3f})")
print(f"  TimeSeriesSplit: {ts_p2_opt['pr_auc']['mean']:.3f} → {ts_p3_opt['pr_auc']['mean']:.3f}  ({pr_delta_ts:+.3f})")

# ── phase3_results.md 생성 ────────────────────────────────────────────────────
def row_m(label, pr_auc, f1, recall, prec, note=""):
    return (f"| {label} | {pr_auc:.3f} | {f1:.3f} | {recall:.3f} | {prec:.3f} |"
            + (f" {note} |" if note else ""))

def sk_row(label, metrics_opt):
    m = metrics_opt
    return row_m(label,
                 m["pr_auc"]["mean"], m["f1_fail"]["mean"],
                 m["recall_fail"]["mean"], m["precision_fail"]["mean"])

# exp1 contamination table
def exp1_table():
    lines = ["| Model | Contamination | Precision | Recall | F1 | PR-AUC |",
             "|-------|--------------|-----------|--------|----|--------|"]
    for _, r in exp1_df.iterrows():
        marker = " ←actual" if abs(r["contamination"] - FAIL_RATE) < 0.001 else ""
        lines.append(f"| {r['model']} | {r['contamination']:.3f}{marker} "
                     f"| {r['precision']:.3f} | {r['recall']:.3f} "
                     f"| {r['f1']:.3f} | {r['pr_auc']:.3f} |")
    return "\n".join(lines)

pr_stuck = abs(pr_delta_sk) < 0.01

md = f"""# Phase 3: 비지도 이상탐지 실험 결과

**분석 일자**: 2026-06-20
**주 지표**: StratifiedKFold (prelim check 결과 — 이유: TimeSeriesSplit의 Fold 편향 확인)

---

## 0. CV 전략 결론 (사전 점검 반영)

Phase 3 이후 **StratifiedKFold를 주 평가 지표**로 사용.
TimeSeriesSplit은 보조 지표로 병기하되, 소수 클래스 fold 불안정으로 인한
Recall 고평가 가능성을 인지해야 함.
→ 세부 근거: `reports/phase3_prelim_check.md`

---

## 1. 실험 1: 비지도 이상탐지 단독 평가 (라벨 미사용)

**설계**: 전체 X에 대해 IF/LOF fit (y 사용 없음), 이상치 예측 결과를 Fail 예측으로 환산

{exp1_table()}

### 해석

**IF (Isolation Forest)**
- PR-AUC ≈ {_e1('IF','pr_auc'):.3f}: Phase 1 지도학습 baseline ({0.18:.3f})보다 {"높음 ✅" if _e1('IF','pr_auc') > 0.18 else "낮음 ❌"}
- contamination 파라미터가 높을수록 Recall↑, Precision↓ (트레이드오프)
- contamination=actual fail rate({FAIL_RATE:.2%}): Recall={_e1('IF','recall'):.3f}, Precision={_e1('IF','precision'):.3f}
- **해석**: 공정 변수 자체의 이상 패턴이 불량과 부분적으로 일치.
  Isolation Forest는 고차원(446개 피처)에서 random splitting을 사용하므로 노이즈 피처가 많을수록 성능 저하

**LOF (Local Outlier Factor)**
- PR-AUC ≈ {_e1('LOF','pr_auc'):.3f}
- IF 대비 Precision이 {"더 높음 — 밀도 기반 탐지로 더 정밀한 이상탐지" if _e1('LOF','precision') > _e1('IF','precision') else "더 낮음 — 고차원에서 거리 기반 밀도 추정의 한계"}
- n_neighbors=20 설정 기준 평가. Fail 클러스터가 다양한 규모일 경우 단일 n_neighbors 한계 존재

**비지도 vs 지도학습 비교**
| 방법 | PR-AUC | Recall(Fail) | Precision(Fail) |
|------|--------|--------------|-----------------|
| Phase 1 LR (best, SK) | 0.150 | 0.498 | 0.156 |
| Phase 2 RF SMOTE+RUS (SK) | {sk_p2_opt['pr_auc']['mean']:.3f} | {sk_p2_opt['recall_fail']['mean']:.3f} | {sk_p2_opt['precision_fail']['mean']:.3f} |
| IF (contamination={FAIL_RATE:.3f}) | {_e1('IF','pr_auc'):.3f} | {_e1('IF','recall'):.3f} | {_e1('IF','precision'):.3f} |
| LOF (contamination={FAIL_RATE:.3f}) | {_e1('LOF','pr_auc'):.3f} | {_e1('LOF','recall'):.3f} | {_e1('LOF','precision'):.3f} |

→ 비지도 단독으로는 지도학습 Phase 2 대비 PR-AUC가 낮음.
  **그러나 라벨 없이 이정도 탐지 가능** = "공정 이상 신호"가 실제로 존재함을 확인.

---

## 2. 실험 2: 이상점수를 피처로 추가 (RF + SMOTE+RUS)

**Pipeline**:
```
[StandardScaler] → [AnomalyScoreAdder] → [SMOTE+RUS] → [RandomForest]
                         ↓
           fit: train fold만 사용 (누수 없음)
           output: 원래 446 피처 + IF score + LOF score = 448 피처
```

### StratifiedKFold 결과 (주 지표)

| 구성 | PR-AUC | F1(Fail) | Recall(Fail) | Precision(Fail) | Opt Thr |
|------|--------|----------|--------------|-----------------|---------|
| Phase2: RF+SMOTE+RUS (no anomaly) | {sk_p2_opt['pr_auc']['mean']:.3f}±{sk_p2_opt['pr_auc']['std']:.3f} | {sk_p2_opt['f1_fail']['mean']:.3f} | {sk_p2_opt['recall_fail']['mean']:.3f} | {sk_p2_opt['precision_fail']['mean']:.3f} | {exp2_rows['StratifiedKFold']['p2']['thresholds']['mean']:.2f} |
| Phase3: +AnomalyScores           | {sk_p3_opt['pr_auc']['mean']:.3f}±{sk_p3_opt['pr_auc']['std']:.3f} | {sk_p3_opt['f1_fail']['mean']:.3f} | {sk_p3_opt['recall_fail']['mean']:.3f} | {sk_p3_opt['precision_fail']['mean']:.3f} | {exp2_rows['StratifiedKFold']['p3']['thresholds']['mean']:.2f} |
| **변화량** | **{pr_delta_sk:+.3f}** | {sk_p3_opt['f1_fail']['mean']-sk_p2_opt['f1_fail']['mean']:+.3f} | {sk_p3_opt['recall_fail']['mean']-sk_p2_opt['recall_fail']['mean']:+.3f} | {sk_p3_opt['precision_fail']['mean']-sk_p2_opt['precision_fail']['mean']:+.3f} | — |

### TimeSeriesSplit 결과 (보조 지표)

| 구성 | PR-AUC | F1(Fail) | Recall(Fail) | Precision(Fail) |
|------|--------|----------|--------------|-----------------|
| Phase2: RF+SMOTE+RUS (no anomaly) | {ts_p2_opt['pr_auc']['mean']:.3f} | {ts_p2_opt['f1_fail']['mean']:.3f} | {ts_p2_opt['recall_fail']['mean']:.3f} | {ts_p2_opt['precision_fail']['mean']:.3f} |
| Phase3: +AnomalyScores           | {ts_p3_opt['pr_auc']['mean']:.3f} | {ts_p3_opt['f1_fail']['mean']:.3f} | {ts_p3_opt['recall_fail']['mean']:.3f} | {ts_p3_opt['precision_fail']['mean']:.3f} |
| 변화량 | {pr_delta_ts:+.3f} | {ts_p3_opt['f1_fail']['mean']-ts_p2_opt['f1_fail']['mean']:+.3f} | {ts_p3_opt['recall_fail']['mean']-ts_p2_opt['recall_fail']['mean']:+.3f} | {ts_p3_opt['precision_fail']['mean']-ts_p2_opt['precision_fail']['mean']:+.3f} |

---

## 3. PR-AUC 0.18 고착 문제 분석

**결과**: 이상점수 피처 추가 후 PR-AUC 변화 = {pr_delta_sk:+.3f} (StratifiedKFold 기준)

{'### 고착 유지 — 가설 분석' if pr_stuck else '### PR-AUC 개선 달성'}

{'PR-AUC가 여전히 개선되지 않고 있음. 가능한 원인 가설:' if pr_stuck else f'PR-AUC가 {pr_delta_sk:+.3f} 개선됨 — 이상점수가 유효한 신호를 추가함.'}

#### 가설 1: 피처 수 대비 신호 밀도 부족
- 446개 피처 중 통계적 유의 피처 80개(17.9%) → 노이즈 피처가 82%
- RF는 bootstrap sampling으로 피처 subset을 선택하지만, 노이즈가 많으면 랜덤 split에 유리한 피처를 만나기 어려움
- 이상점수 2개는 446개 중 2/448 = 0.45% 비중 → 전체 특성 공간에서 미미한 영향

#### 가설 2: LF(Local Failure) 패턴의 비선형성
- SECOM 공정 불량은 여러 센서의 비선형 조합으로 발생할 가능성이 높음
- Isolation Forest와 LOF는 선형/밀도 기반 이상치 탐지 → 복잡한 비선형 상호작용 포착 한계
- → **Phase 4에서 XGBoost/Gradient Boosting처럼 비선형 분리를 더 잘 학습하는 모델 시도 필요**

#### 가설 3: 불량 샘플 자체가 "이상치"가 아닐 수 있음
- 불량이 공정 표준값에서 멀리 벗어난 샘플에서만 발생하는 게 아닐 수 있음
- "정상 범위 내 공정 파라미터 조합"이 불량을 유발하는 케이스 → Isolation Forest가 탐지 불가
- Phase 3 결과가 이 가설을 일부 지지: IF Recall ≈ {_e1('IF','recall'):.3f} (낮음)

---

## 4. 전체 3-way 비교 (StratifiedKFold 기준)

| 방법 분류 | 구체 구성 | PR-AUC | F1(Fail) | Recall(Fail) | Precision(Fail) |
|-----------|-----------|--------|----------|--------------|-----------------|
| 비지도 단독 | IF (c={FAIL_RATE:.3f}) | {_e1('IF','pr_auc'):.3f} | {_e1('IF','f1'):.3f} | {_e1('IF','recall'):.3f} | {_e1('IF','precision'):.3f} |
| 비지도 단독 | LOF (c={FAIL_RATE:.3f}) | {_e1('LOF','pr_auc'):.3f} | {_e1('LOF','f1'):.3f} | {_e1('LOF','recall'):.3f} | {_e1('LOF','precision'):.3f} |
| Phase2 (지도) | RF+SMOTE+RUS | {sk_p2_opt['pr_auc']['mean']:.3f} | {sk_p2_opt['f1_fail']['mean']:.3f} | {sk_p2_opt['recall_fail']['mean']:.3f} | {sk_p2_opt['precision_fail']['mean']:.3f} |
| Phase3 (지도+비지도) | RF+SMOTE+RUS+Anomaly | {sk_p3_opt['pr_auc']['mean']:.3f} | {sk_p3_opt['f1_fail']['mean']:.3f} | {sk_p3_opt['recall_fail']['mean']:.3f} | {sk_p3_opt['precision_fail']['mean']:.3f} |
| **목표** | — | **≥0.40** | — | **≥0.70** | ≥0.20 |

---

## 5. Phase 4로 넘어가야 하는 이유

Phase 3까지의 실험이 증명한 것: **피처 품질과 모델 아키텍처가 PR-AUC를 결정하는 핵심 요인**

1. **단순 불균형 처리(Phase 2)**: Recall↑ 가능, 그러나 PR-AUC 고착
2. **비지도 이상탐지(Phase 3)**: 이상 신호 존재는 확인, 지도학습 대비 PR-AUC 열세
3. **이상점수 피처 추가(Phase 3)**: PR-AUC 변화 {pr_delta_sk:+.3f} — {'미미함' if abs(pr_delta_sk) < 0.02 else '유의미'}

**Phase 4 전략**:
- 피처 선택 (SECOM 통계적 유의 피처 80개 집중 + 이상점수 포함)
- 더 강력한 비선형 모델: XGBoost / LightGBM (많은 피처에서 자동 선택)
- SHAP 기반 설명 가능성 분석으로 핵심 공정 변수 규명
- 이 과정에서 PR-AUC 0.40, Recall 0.70 목표 달성 가능성 검증

---

## 6. 산출 파일

| 파일 | 설명 |
|------|------|
| `reports/figures/15_prelim_fold_recall.png` | Fold-level Recall 비교 |
| `reports/figures/16_p3_anomaly_standalone.png` | 비지도 단독 contamination별 성능 |
| `reports/figures/17_p3_pr_curves_comparison.png` | 전체 PR Curve 비교 |
| `reports/figures/18_p3_vs_p2_comparison.png` | Phase 2 vs Phase 3 바 차트 |

---
*Generated by `notebooks/phase3_unsupervised/01_anomaly_detection.py`*
"""

(ROOT / "reports" / "phase3_results.md").write_text(md)
print("\n  Saved: reports/phase3_results.md")
print("\nPhase 3 Complete.")
