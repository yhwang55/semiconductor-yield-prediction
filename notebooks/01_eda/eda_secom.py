"""
SECOM EDA — Pass/Fail 분포, 결측치 패턴, 분산 분석, 그룹 간 차이
출력: reports/figures/*.png
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── 경로 설정 ──────────────────────────────────────────────────────────────────
ROOT = Path(__file__).parent.parent.parent
DATA_DIR = ROOT / "data" / "raw"
FIG_DIR = ROOT / "reports" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})
PALETTE = {"Pass": "#2196F3", "Fail": "#F44336"}


# ── 1. 데이터 로딩 ─────────────────────────────────────────────────────────────
print("=" * 60)
print("1. DATA LOADING")
print("=" * 60)

X = pd.read_csv(DATA_DIR / "secom.data", sep=" ", header=None)
y_raw = pd.read_csv(DATA_DIR / "secom_labels.data", sep=" ", header=None)

X.columns = [f"sensor_{i+1}" for i in range(X.shape[1])]
y = y_raw.iloc[:, 0].map({-1: "Pass", 1: "Fail"})
timestamp = pd.to_datetime(y_raw.iloc[:, 1])

print(f"  X shape       : {X.shape}  ({X.shape[0]} samples × {X.shape[1]} features)")
print(f"  y distribution: {y.value_counts().to_dict()}")
print(f"  Time range    : {timestamp.min()} → {timestamp.max()}")

n_samples, n_features = X.shape
n_pass = (y == "Pass").sum()
n_fail = (y == "Fail").sum()
fail_rate = n_fail / n_samples
imbalance_ratio = n_pass / n_fail

print(f"\n  Fail rate     : {fail_rate:.2%}")
print(f"  Imbalance     : 1 : {imbalance_ratio:.1f}  (Fail : Pass)")


# ── 2. 클래스 불균형 시각화 ────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("2. CLASS IMBALANCE VISUALIZATION")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(10, 4))

# 2-1. Pie chart
counts = y.value_counts()
axes[0].pie(
    counts,
    labels=counts.index,
    autopct="%1.1f%%",
    colors=[PALETTE[k] for k in counts.index],
    startangle=90,
    wedgeprops={"edgecolor": "white", "linewidth": 2},
)
axes[0].set_title(f"Class Distribution\n(n={n_samples:,})")

# 2-2. Bar chart
bar_colors = [PALETTE[k] for k in counts.index]
bars = axes[1].bar(counts.index, counts.values, color=bar_colors, edgecolor="white", linewidth=1.5)
for bar, v in zip(bars, counts.values):
    axes[1].text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 5,
                 f"{v:,}", ha="center", va="bottom", fontweight="bold")
axes[1].set_ylabel("Count")
axes[1].set_title(f"Pass vs Fail\n(Imbalance Ratio 1 : {imbalance_ratio:.1f})")
axes[1].set_ylim(0, counts.max() * 1.15)

plt.tight_layout()
plt.savefig(FIG_DIR / "01_class_imbalance.png", bbox_inches="tight")
plt.close()
print("  Saved: 01_class_imbalance.png")


# ── 3. 결측치 분석 ─────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("3. MISSING VALUE ANALYSIS")
print("=" * 60)

missing_count = X.isnull().sum()
missing_pct = missing_count / n_samples * 100

n_all_missing = (missing_pct == 100).sum()
n_heavy_missing = ((missing_pct >= 50) & (missing_pct < 100)).sum()
n_partial_missing = ((missing_pct > 0) & (missing_pct < 50)).sum()
n_complete = (missing_pct == 0).sum()

print(f"  Features with 100% missing  : {n_all_missing}")
print(f"  Features with ≥50% missing  : {n_heavy_missing}")
print(f"  Features with 1–49% missing : {n_partial_missing}")
print(f"  Features with 0% missing    : {n_complete}")
print(f"\n  Sample-level: {X.isnull().any(axis=1).sum()} rows have at least one NaN")

# 결측률 분포 히스토그램
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(missing_pct[missing_pct > 0], bins=30, color="#607D8B", edgecolor="white")
axes[0].axvline(50, color="red", linestyle="--", label="50% threshold")
axes[0].set_xlabel("Missing Rate (%)")
axes[0].set_ylabel("Feature Count")
axes[0].set_title("Missing Rate Distribution\n(features with any missing)")
axes[0].legend()

# 결측률 상위 30개 센서
top_missing = missing_pct.sort_values(ascending=False).head(30)
colors_miss = ["#F44336" if v >= 50 else "#FF9800" if v >= 20 else "#FFC107"
               for v in top_missing.values]
axes[1].barh(range(len(top_missing)), top_missing.values, color=colors_miss)
axes[1].set_yticks(range(len(top_missing)))
axes[1].set_yticklabels(top_missing.index, fontsize=8)
axes[1].axvline(50, color="red", linestyle="--", alpha=0.7, label="50%")
axes[1].set_xlabel("Missing Rate (%)")
axes[1].set_title("Top 30 Features by Missing Rate")
axes[1].legend()

plt.tight_layout()
plt.savefig(FIG_DIR / "02_missing_values.png", bbox_inches="tight")
plt.close()
print("  Saved: 02_missing_values.png")

# 선행연구 비교: 28개 이상 결측 센서
threshold_28 = 90  # 통상 28개 센서가 결측률 90% 이상으로 알려져 있음
n_90pct_missing = (missing_pct >= threshold_28).sum()
print(f"\n  ≥90% missing features : {n_90pct_missing}  (prior literature ref: ~28)")


# ── 4. 분산 0 / 상수 피처 확인 ────────────────────────────────────────────────
print("\n" + "=" * 60)
print("4. ZERO VARIANCE / NEAR-ZERO VARIANCE FEATURES")
print("=" * 60)

feature_std = X.std()
zero_var = (feature_std == 0).sum()
near_zero = ((feature_std > 0) & (feature_std < 1e-4)).sum()

print(f"  Zero variance features      : {zero_var}")
print(f"  Near-zero variance (<1e-4)  : {near_zero}")
print(f"  Effectively usable features : {n_features - zero_var - n_all_missing} (excl. zero-var & 100%-missing)")

# 분산 분포
fig, ax = plt.subplots(figsize=(9, 4))
log_std = np.log10(feature_std[feature_std > 0])
ax.hist(log_std, bins=50, color="#4CAF50", edgecolor="white", alpha=0.8)
ax.axvline(np.log10(1e-4), color="orange", linestyle="--", label="near-zero threshold (1e-4)")
ax.axvline(np.log10(1e0), color="blue", linestyle="--", alpha=0.6, label="std = 1")
ax.set_xlabel("log₁₀(Standard Deviation)")
ax.set_ylabel("Feature Count")
ax.set_title("Feature Standard Deviation Distribution\n(log scale, excluding zero-std features)")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "03_variance_distribution.png", bbox_inches="tight")
plt.close()
print("  Saved: 03_variance_distribution.png")


# ── 5. Pass/Fail 그룹 간 평균 차이 Top 20 ──────────────────────────────────────
print("\n" + "=" * 60)
print("5. TOP 20 DISCRIMINATIVE FEATURES (Pass vs Fail)")
print("=" * 60)

# t-검정 기반 그룹 간 차이 정량화
pass_mask = (y == "Pass").values
fail_mask = (y == "Fail").values

results = []
for col in X.columns:
    g_pass = X.loc[pass_mask, col].dropna()
    g_fail = X.loc[fail_mask, col].dropna()
    if len(g_pass) < 5 or len(g_fail) < 5:
        continue
    if g_pass.std() == 0 and g_fail.std() == 0:
        continue
    # Welch's t-test
    t_stat, p_val = stats.ttest_ind(g_pass, g_fail, equal_var=False)
    # Cohen's d (effect size)
    pooled_std = np.sqrt((g_pass.std() ** 2 + g_fail.std() ** 2) / 2)
    cohen_d = abs(g_pass.mean() - g_fail.mean()) / (pooled_std + 1e-12)
    results.append({
        "feature": col,
        "pass_mean": g_pass.mean(),
        "fail_mean": g_fail.mean(),
        "mean_diff_abs": abs(g_pass.mean() - g_fail.mean()),
        "cohen_d": cohen_d,
        "t_stat": abs(t_stat),
        "p_value": p_val,
    })

results_df = pd.DataFrame(results).sort_values("cohen_d", ascending=False)
top20 = results_df.head(20)
sig_features = (results_df["p_value"] < 0.05).sum()
print(f"  Statistically significant (p<0.05) : {sig_features} / {len(results_df)} features")
print(f"\n  Top 5 by Cohen's d:")
print(top20[["feature", "pass_mean", "fail_mean", "cohen_d", "p_value"]].head(5).to_string(index=False))

# 시각화
fig, ax = plt.subplots(figsize=(11, 7))
colors_top = ["#EF5350" if d > 0.8 else "#FFA726" if d > 0.5 else "#66BB6A"
              for d in top20["cohen_d"]]
bars = ax.barh(range(len(top20)), top20["cohen_d"].values, color=colors_top, edgecolor="white")
ax.set_yticks(range(len(top20)))
ax.set_yticklabels(top20["feature"].values, fontsize=9)
ax.axvline(0.8, color="red", linestyle="--", alpha=0.7, label="Large effect (d=0.8)")
ax.axvline(0.5, color="orange", linestyle="--", alpha=0.7, label="Medium effect (d=0.5)")
ax.axvline(0.2, color="green", linestyle="--", alpha=0.7, label="Small effect (d=0.2)")
ax.set_xlabel("Cohen's d (Effect Size)")
ax.set_title("Top 20 Features: Pass vs Fail Group Difference\n(sorted by Cohen's d, Welch's t-test)")
ax.legend(loc="lower right")
ax.invert_yaxis()
plt.tight_layout()
plt.savefig(FIG_DIR / "04_top20_discriminative.png", bbox_inches="tight")
plt.close()
print("  Saved: 04_top20_discriminative.png")


# ── 6. Top 10 피처 박스플롯 ───────────────────────────────────────────────────
print("\n" + "=" * 60)
print("6. BOX PLOTS — TOP 10 DISCRIMINATIVE FEATURES")
print("=" * 60)

top10_features = top20["feature"].head(10).tolist()
plot_df = X[top10_features].copy()
plot_df["label"] = y.values

fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for i, feat in enumerate(top10_features):
    data_pass = plot_df.loc[plot_df["label"] == "Pass", feat].dropna()
    data_fail = plot_df.loc[plot_df["label"] == "Fail", feat].dropna()
    axes[i].boxplot(
        [data_pass, data_fail],
        labels=["Pass", "Fail"],
        patch_artist=True,
        boxprops=dict(facecolor="lightblue"),
        medianprops=dict(color="navy", linewidth=2),
        flierprops=dict(marker=".", markersize=2, alpha=0.3),
    )
    axes[i].set_title(feat, fontsize=9)
    axes[i].tick_params(labelsize=8)
    cohen_d_val = top20.loc[top20["feature"] == feat, "cohen_d"].values[0]
    p_val = top20.loc[top20["feature"] == feat, "p_value"].values[0]
    axes[i].set_xlabel(f"d={cohen_d_val:.2f}, p={p_val:.2e}", fontsize=7)

plt.suptitle("Box Plots: Pass vs Fail — Top 10 Discriminative Sensors", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig(FIG_DIR / "05_top10_boxplots.png", bbox_inches="tight")
plt.close()
print("  Saved: 05_top10_boxplots.png")


# ── 7. 결측치 × 클래스 교차 분석 ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("7. MISSING VALUE PATTERN BY CLASS")
print("=" * 60)

row_missing_rate = X.isnull().mean(axis=1)
miss_by_class = pd.DataFrame({
    "label": y.values,
    "row_missing_rate": row_missing_rate.values
})
summary = miss_by_class.groupby("label")["row_missing_rate"].describe()
print(summary.round(4).to_string())

fig, ax = plt.subplots(figsize=(7, 4))
for label, color in [("Pass", "#2196F3"), ("Fail", "#F44336")]:
    subset = miss_by_class.loc[miss_by_class["label"] == label, "row_missing_rate"]
    ax.hist(subset, bins=40, alpha=0.6, color=color, label=label, density=True)
ax.set_xlabel("Row-level Missing Rate")
ax.set_ylabel("Density")
ax.set_title("Missing Rate per Sample: Pass vs Fail")
ax.legend()
plt.tight_layout()
plt.savefig(FIG_DIR / "06_missing_by_class.png", bbox_inches="tight")
plt.close()
print("  Saved: 06_missing_by_class.png")


# ── 8. 시계열 수율 추이 ────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("8. TEMPORAL YIELD TREND")
print("=" * 60)

time_df = pd.DataFrame({"timestamp": timestamp, "label": y.values})
time_df["date"] = time_df["timestamp"].dt.date
daily = time_df.groupby("date")["label"].apply(
    lambda x: (x == "Fail").sum() / len(x) * 100
).reset_index()
daily.columns = ["date", "fail_rate_pct"]

fig, ax = plt.subplots(figsize=(12, 4))
ax.plot(daily["date"], daily["fail_rate_pct"], color="#E91E63", linewidth=1.5, alpha=0.8)
ax.fill_between(daily["date"], daily["fail_rate_pct"], alpha=0.15, color="#E91E63")
ax.axhline(fail_rate * 100, color="black", linestyle="--", alpha=0.5,
           label=f"Overall fail rate: {fail_rate*100:.1f}%")
ax.set_xlabel("Date")
ax.set_ylabel("Daily Fail Rate (%)")
ax.set_title("Temporal Yield Trend (Daily Fail Rate)")
ax.legend()
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig(FIG_DIR / "07_temporal_yield.png", bbox_inches="tight")
plt.close()
print("  Saved: 07_temporal_yield.png")


# ── 9. 요약 통계 저장 ─────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("9. SUMMARY STATISTICS EXPORT")
print("=" * 60)

summary_stats = {
    "n_samples": n_samples,
    "n_features": n_features,
    "n_pass": int(n_pass),
    "n_fail": int(n_fail),
    "fail_rate": round(float(fail_rate), 4),
    "imbalance_ratio": round(float(imbalance_ratio), 2),
    "n_all_missing_features": int(n_all_missing),
    "n_heavy_missing_features": int(n_heavy_missing),
    "n_zero_var_features": int(zero_var),
    "n_significant_features_p05": int(sig_features),
    "top20_discriminative": top20[["feature", "cohen_d", "p_value"]].to_dict("records"),
}

import json
with open(ROOT / "reports" / "eda_summary.json", "w") as f:
    json.dump(summary_stats, f, indent=2, default=str)

print("  Saved: reports/eda_summary.json")

print("\n" + "=" * 60)
print("EDA COMPLETE")
print("=" * 60)
print(f"  Figures saved to: {FIG_DIR}")

# 최종 요약 출력
print(f"""
┌─────────────────────────────────────────────┐
│  SECOM EDA SUMMARY                          │
├─────────────────────────────────────────────┤
│  Samples          : {n_samples:,}                   │
│  Features         : {n_features}                     │
│  Pass / Fail      : {n_pass:,} / {n_fail:,}             │
│  Fail rate        : {fail_rate:.2%}                 │
│  Imbalance ratio  : 1 : {imbalance_ratio:.1f}                │
│  100% NaN features: {n_all_missing}                      │
│  ≥50% NaN features: {n_heavy_missing}                      │
│  Zero-var features: {zero_var}                       │
│  Sig. features    : {sig_features} / {len(results_df)}               │
└─────────────────────────────────────────────┘
""")
