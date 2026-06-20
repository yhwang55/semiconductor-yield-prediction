"""
Phase 1 — Step 1: 전처리 비교 실험
  - 상수 피처(116개) 제거, 고결측 피처(28개) 제거 → 446 피처
  - Median vs IterativeImputer(MICE) 비교
  - 처리된 데이터셋 data/processed/에 저장
  - 시각화 reports/figures/08~10_*.png 저장
  - reports/imputation_comparison.md 생성
"""

import sys, time, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.data.preprocessing import (
    load_raw, filter_features,
    apply_median_imputer, apply_iterative_imputer,
)

PROC_DIR = ROOT / "data" / "processed"
FIG_DIR = ROOT / "reports" / "figures"
PROC_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({"figure.dpi": 150, "font.size": 10})

# ── 1. 로딩 및 피처 필터링 ─────────────────────────────────────────────────────
print("=" * 60)
print("PHASE 1 — PREPROCESSING")
print("=" * 60)

X_raw, y, timestamps = load_raw()
X_filt, const_feats, highmiss_feats = filter_features(X_raw)

print(f"  Raw features          : {X_raw.shape[1]}")
print(f"  Removed (zero-var)    : {len(const_feats)}")
print(f"  Removed (≥50% miss)   : {len(highmiss_feats)}")
print(f"  Remaining features    : {X_filt.shape[1]}")
print(f"  Remaining missing val : {X_filt.isnull().sum().sum():,}")

missing_before = X_filt.isnull().mean()  # per-feature missing rate (for comparison later)

# ── 2. Median Imputation ───────────────────────────────────────────────────────
print("\n[Median Imputation]")
t0 = time.time()
X_median = apply_median_imputer(X_filt)
print(f"  Done in {time.time()-t0:.1f}s  | NaN remaining: {X_median.isnull().sum().sum()}")

# ── 3. IterativeImputer (MICE) ─────────────────────────────────────────────────
print("\n[IterativeImputer / MICE]  (may take ~60s)")
t0 = time.time()
X_iter = apply_iterative_imputer(X_filt)
print(f"  Done in {time.time()-t0:.1f}s  | NaN remaining: {X_iter.isnull().sum().sum()}")

# ── 4. 저장 ────────────────────────────────────────────────────────────────────
print("\n[Saving processed datasets]")
X_median.to_parquet(PROC_DIR / "X_median.parquet", index=False)
X_iter.to_parquet(PROC_DIR / "X_iterative.parquet", index=False)
y.to_frame("label").to_parquet(PROC_DIR / "y.parquet", index=False)
timestamps.to_frame("timestamp").to_parquet(PROC_DIR / "timestamps.parquet", index=False)
pd.Series(X_filt.columns.tolist()).to_frame("feature").to_parquet(PROC_DIR / "features.parquet", index=False)
print("  Saved: X_median, X_iterative, y, timestamps, features")

# ── 5. 비교 통계 계산 ──────────────────────────────────────────────────────────
# 두 imputation 방식이 실제로 얼마나 다른지 → NaN 위치의 값만 비교
nan_mask = X_filt.isnull()
n_imputed_cells = nan_mask.sum().sum()

diff_arr = (X_median[nan_mask] - X_iter[nan_mask]).values.flatten()
diff_arr = diff_arr[~np.isnan(diff_arr)]

mae = np.abs(diff_arr).mean()
pct_agree = (np.abs(diff_arr) < 1e-6).mean() * 100

# 피처별 통계 비교
feat_stats = pd.DataFrame({
    "feature": X_filt.columns,
    "missing_pct": missing_before.values,
    "median_mean": X_median.mean().values,
    "iter_mean": X_iter.mean().values,
    "median_std": X_median.std().values,
    "iter_std": X_iter.std().values,
})
feat_stats["mean_diff_abs"] = np.abs(feat_stats["median_mean"] - feat_stats["iter_mean"])
feat_stats["std_ratio"] = feat_stats["iter_std"] / (feat_stats["median_std"] + 1e-12)

# label correlation 비교
corr_median = X_median.corrwith(y.astype(float)).abs()
corr_iter = X_iter.corrwith(y.astype(float)).abs()
corr_raw = pd.Series({
    col: X_filt[col].dropna().values @ y[X_filt[col].notna()].values /
         (len(X_filt[col].dropna()) * X_filt[col].std() * y[X_filt[col].notna()].std() + 1e-12)
    for col in X_filt.columns
}).abs()

print(f"\n  Imputed cells (NaN positions): {n_imputed_cells:,}")
print(f"  Median vs MICE MAE           : {mae:.4f}")
print(f"  Exact agreement              : {pct_agree:.1f}%")
print(f"  Features with >10% mean diff : {(feat_stats['mean_diff_abs'] / (feat_stats['median_mean'].abs() + 1e-10) > 0.1).sum()}")

# ── 6. Figure 08: 결측률 Top 10 피처 분포 비교 ────────────────────────────────
top_miss_feats = missing_before.sort_values(ascending=False).head(10).index.tolist()

fig, axes = plt.subplots(2, 5, figsize=(20, 7))
axes = axes.flatten()

for i, feat in enumerate(top_miss_feats):
    nan_m = X_filt[feat].isnull()
    obs = X_filt[feat].dropna()
    med_imp = X_median.loc[nan_m, feat]
    it_imp = X_iter.loc[nan_m, feat]

    axes[i].hist(obs, bins=25, alpha=0.45, color="gray", label="Observed", density=True)
    axes[i].hist(med_imp, bins=25, alpha=0.5, color="#2196F3", label="Median", density=True)
    axes[i].hist(it_imp, bins=25, alpha=0.5, color="#E91E63", label="MICE", density=True)

    miss_r = missing_before[feat]
    axes[i].set_title(f"{feat}\n(missing: {miss_r:.1%})", fontsize=8)
    axes[i].tick_params(labelsize=7)
    if i == 0:
        axes[i].legend(fontsize=7)

plt.suptitle(
    "Imputation Comparison: Distribution of Imputed Values\n"
    "Top 10 Features by Missing Rate — Median (blue) vs MICE (pink)",
    fontsize=11, y=1.01,
)
plt.tight_layout()
plt.savefig(FIG_DIR / "08_imputation_dist_comparison.png", bbox_inches="tight")
plt.close()
print("\n  Saved: 08_imputation_dist_comparison.png")

# ── 7. Figure 09: 피처별 평균 비교 & std 비율 ─────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 9-1: 평균 차이 산점도
axes[0].scatter(feat_stats["median_mean"], feat_stats["iter_mean"],
                c=feat_stats["missing_pct"], cmap="YlOrRd", s=10, alpha=0.6)
lim_lo = min(feat_stats["median_mean"].min(), feat_stats["iter_mean"].min())
lim_hi = max(feat_stats["median_mean"].max(), feat_stats["iter_mean"].max())
axes[0].plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", linewidth=0.8, label="y=x (identical)")
axes[0].set_xlabel("Median Imputed — Feature Mean")
axes[0].set_ylabel("MICE Imputed — Feature Mean")
axes[0].set_title("Per-Feature Mean: Median vs MICE\n(color = missing rate)")
sm = plt.cm.ScalarMappable(cmap="YlOrRd",
     norm=plt.Normalize(feat_stats["missing_pct"].min(), feat_stats["missing_pct"].max()))
plt.colorbar(sm, ax=axes[0], label="Missing Rate")
axes[0].legend(fontsize=8)

# 9-2: label 상관관계 비교
axes[1].scatter(corr_median, corr_iter, s=8, alpha=0.5, color="#4CAF50")
m = max(corr_median.max(), corr_iter.max())
axes[1].plot([0, m], [0, m], "k--", linewidth=0.8, label="y=x")
axes[1].set_xlabel("|Corr(feature, label)| — Median Imputed")
axes[1].set_ylabel("|Corr(feature, label)| — MICE Imputed")
axes[1].set_title("Signal Preservation: Label Correlation\n(Median vs MICE)")
axes[1].legend(fontsize=8)

plt.tight_layout()
plt.savefig(FIG_DIR / "09_imputation_stats_comparison.png", bbox_inches="tight")
plt.close()
print("  Saved: 09_imputation_stats_comparison.png")

# ── 8. KS test: 분포 보존 여부 정량화 ─────────────────────────────────────────
ks_results = []
for feat in top_miss_feats:
    nan_m = X_filt[feat].isnull()
    obs = X_filt[feat].dropna().values
    if len(obs) < 10:
        continue
    med_imp = X_median.loc[nan_m, feat].values
    it_imp = X_iter.loc[nan_m, feat].values
    ks_med = stats.ks_2samp(obs, med_imp).statistic
    ks_it = stats.ks_2samp(obs, it_imp).statistic
    ks_results.append({"feature": feat, "ks_median": ks_med, "ks_mice": ks_it,
                        "missing_pct": missing_before[feat]})

ks_df = pd.DataFrame(ks_results)
print(f"\n  KS statistic (lower = more similar to observed):")
print(f"  Median avg KS: {ks_df['ks_median'].mean():.3f}  |  MICE avg KS: {ks_df['ks_mice'].mean():.3f}")
mice_better = (ks_df["ks_mice"] < ks_df["ks_median"]).sum()
print(f"  MICE better at distribution preservation: {mice_better}/{len(ks_df)} top features")

corr_improvement = (corr_iter - corr_median).mean()
print(f"  Label correlation improvement (MICE vs Median): {corr_improvement:+.5f}")

# ── 9. imputation_comparison.md 생성 ──────────────────────────────────────────
ks_table = "\n".join(
    f"| {r['feature']} | {r['missing_pct']:.1%} | {r['ks_median']:.3f} | {r['ks_mice']:.3f} |"
    for _, r in ks_df.iterrows()
)

md_content = f"""# Imputation Comparison Report

**분석 일자**: 2026-06-19
**목적**: Median Imputation vs IterativeImputer(MICE) 비교 — Phase 1 전처리 전략 결정

---

## 1. 피처 필터링 결과

| 단계 | 피처 수 | 조치 |
|------|---------|------|
| 원본 | 590 | — |
| 상수 피처 제거 | −{len(const_feats)} | VarianceThreshold(0) |
| 고결측(≥50%) 제거 | −{len(highmiss_feats)} | — |
| **최종 유효 피처** | **{X_filt.shape[1]}** | 모델링에 사용 |

제거 후 잔존 결측 셀 수: **{n_imputed_cells:,}개** (전체의 {n_imputed_cells / (X_filt.shape[0]*X_filt.shape[1]):.1%})

---

## 2. Imputation 방법 비교

| 항목 | Median | MICE (IterativeImputer) |
|------|--------|------------------------|
| 전략 | 각 피처의 중앙값으로 대체 | 다른 피처로 회귀 예측 (5회 반복, 10개 최근접 피처) |
| 속도 | 즉시 | ~60초 |
| NaN 셀 간 차이(MAE) | — | {mae:.4f} |
| 정확히 일치하는 비율 | — | {pct_agree:.1f}% |
| 피처 간 상관관계 반영 | ✗ | ✓ |

---

## 3. 분포 보존 분석 (KS Test)

결측률 상위 10개 피처에 대해 관측값과 대체값의 분포 유사도를 KS 통계량으로 측정.
KS 통계량이 낮을수록 원본 분포를 더 잘 보존.

| 피처 | 결측률 | KS(Median) | KS(MICE) |
|------|--------|------------|----------|
{ks_table}

**평균 KS 통계량**: Median = {ks_df['ks_median'].mean():.3f} / MICE = {ks_df['ks_mice'].mean():.3f}
MICE가 더 나은 분포 보존: **{mice_better}/{len(ks_df)}** 피처

---

## 4. 신호 보존 분석 (Label Correlation)

각 피처와 Pass/Fail 레이블 간 절대 상관계수의 평균:

| Imputation 방법 | 평균 |Corr(feature, label)| |
|----------------|------|
| Median | {corr_median.mean():.5f} |
| MICE | {corr_iter.mean():.5f} |
| 차이 (MICE − Median) | {corr_improvement:+.5f} |

신호 보존 차이는 {abs(corr_improvement):.5f}로 매우 작음 → 두 방법의 정보량 차이는 미미.

---

## 5. 시각화

- `reports/figures/08_imputation_dist_comparison.png`: 결측률 상위 10 피처의 분포 비교
- `reports/figures/09_imputation_stats_comparison.png`: 피처 평균 산점도 & 레이블 상관관계 비교

---

## 6. 결론 및 Phase 1 전략

| 판단 기준 | 결과 | 선택 |
|-----------|------|------|
| 분포 보존 (KS) | MICE 우세: {mice_better}/{len(ks_df)} | MICE |
| 신호 보존 (corr) | 차이 거의 없음 ({corr_improvement:+.5f}) | 무관 |
| 계산 비용 | Median 압도적으로 빠름 | Median |

**Phase 1 전략**: 두 방식을 **모두 사용**해 2 × 2 조합으로 성능 비교.
→ 모델 성능 차이가 insignificant하면 Median을 기본 파이프라인으로 선택 (Occam's Razor).
→ 차이가 유의미하면 MICE를 채택하고 계산 비용을 감수.

> **다음 단계 시사점**: Imputation 방식보다 **클래스 불균형 처리** (Phase 2)가 모델 성능에 훨씬 큰 영향을 줄 것으로 예상됨.
> Fail class는 104개(6.6%)에 불과하여, 어떤 imputation을 써도 imbalanced baseline은 Recall(Fail) ≈ 0이 예상됨.
"""

(ROOT / "reports" / "imputation_comparison.md").write_text(md_content)
print("\n  Saved: reports/imputation_comparison.md")
print("\nPhase 1 Preprocessing Complete.")
