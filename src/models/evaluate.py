"""
모델 평가 유틸리티 — Phase 1 이후 모든 실험에서 재사용
"""

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    recall_score,
    accuracy_score,
    precision_score,
    roc_auc_score,
    precision_recall_curve,
)
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.preprocessing import StandardScaler


def get_cv_strategies(n_splits: int = 5) -> dict:
    """Phase 전체에서 통일하여 사용할 CV 전략 두 가지 반환."""
    return {
        "TimeSeriesSplit": TimeSeriesSplit(n_splits=n_splits),
        "StratifiedKFold": StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=42
        ),
    }


def cross_val_eval(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    cv,
    scale: bool = True,
) -> dict:
    """Cross-validation 수행 후 메트릭 집계.

    Fail(y=1)이 positive class.
    Returns: {metric: {mean, std, folds}}
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = pd.Series(y) if not isinstance(y, pd.Series) else y

    fold_rows = {
        k: []
        for k in [
            "pr_auc", "roc_auc", "f1_fail",
            "recall_fail", "precision_fail", "accuracy",
        ]
    }

    for train_idx, val_idx in cv.split(X, y):
        X_tr = X.iloc[train_idx].values
        X_va = X.iloc[val_idx].values
        y_tr = y.iloc[train_idx].values
        y_va = y.iloc[val_idx].values

        if scale:
            sc = StandardScaler()
            X_tr = sc.fit_transform(X_tr)
            X_va = sc.transform(X_va)

        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_va)
        y_prob = model.predict_proba(X_va)[:, 1]

        fold_rows["pr_auc"].append(average_precision_score(y_va, y_prob))
        fold_rows["roc_auc"].append(roc_auc_score(y_va, y_prob))
        fold_rows["f1_fail"].append(
            f1_score(y_va, y_pred, pos_label=1, zero_division=0)
        )
        fold_rows["recall_fail"].append(
            recall_score(y_va, y_pred, pos_label=1, zero_division=0)
        )
        fold_rows["precision_fail"].append(
            precision_score(y_va, y_pred, pos_label=1, zero_division=0)
        )
        fold_rows["accuracy"].append(accuracy_score(y_va, y_pred))

    return {
        k: {"mean": np.mean(v), "std": np.std(v), "folds": v}
        for k, v in fold_rows.items()
    }


def make_results_table(all_results: dict) -> pd.DataFrame:
    """중첩 결과 dict → 비교 테이블 DataFrame.

    all_results: {label: {metric: {mean, std}}}
    """
    rows = []
    for label, metrics in all_results.items():
        row = {"configuration": label}
        for metric, stats in metrics.items():
            row[f"{metric}_mean"] = round(stats["mean"], 4)
            row[f"{metric}_std"] = round(stats["std"], 4)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Phase 2 추가 함수 ──────────────────────────────────────────────────────────

def _optimal_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """PR curve에서 F1을 최대화하는 임계값 반환."""
    prec, rec, thr = precision_recall_curve(y_true, y_prob)
    # precision_recall_curve의 마지막 원소는 threshold 없음 → [:-1] 정렬
    f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-10)
    return float(thr[np.argmax(f1s)])


def _fold_metrics(y_va: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc":         average_precision_score(y_va, y_prob),
        "roc_auc":        roc_auc_score(y_va, y_prob),
        "f1_fail":        f1_score(y_va, y_pred, pos_label=1, zero_division=0),
        "recall_fail":    recall_score(y_va, y_pred, pos_label=1, zero_division=0),
        "precision_fail": precision_score(y_va, y_pred, pos_label=1, zero_division=0),
        "accuracy":       accuracy_score(y_va, y_pred),
    }


def _aggregate(folds: dict) -> dict:
    return {k: {"mean": np.mean(v), "std": np.std(v), "folds": v} for k, v in folds.items()}


def cross_val_eval_v2(pipeline, X: pd.DataFrame, y: pd.Series, cv) -> dict:
    """Phase 2용 통합 평가 함수.

    pipeline: sklearn/imblearn Pipeline (scaling + resampling + model 모두 내장).
              resampling은 Pipeline.fit() 내부에서만 적용 → validation leak 없음.
    각 fold마다 clone()으로 새 인스턴스 생성 → fold 간 상태 오염 방지.

    Returns:
        {
          'default':    {metric: {mean, std, folds}},  # threshold = 0.5
          'optimal':    {metric: {mean, std, folds}},  # PR-curve F1 최적 임계값
          'thresholds': {mean, std, folds},             # 선택된 최적 임계값들
        }
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = pd.Series(y)    if not isinstance(y, pd.Series)    else y

    METRIC_KEYS = ["pr_auc", "roc_auc", "f1_fail", "recall_fail", "precision_fail", "accuracy"]
    def_folds  = {k: [] for k in METRIC_KEYS}
    opt_folds  = {k: [] for k in METRIC_KEYS}
    thr_folds  = []

    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_va = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_va = y.iloc[train_idx].values, y.iloc[val_idx].values

        est = clone(pipeline)
        est.fit(X_tr, y_tr)                      # resampling은 여기서만
        y_prob = est.predict_proba(X_va)[:, 1]  # val은 원본 분포 그대로

        opt_thr = _optimal_threshold(y_va, y_prob)
        thr_folds.append(opt_thr)

        for k, v in _fold_metrics(y_va, y_prob, 0.5).items():
            def_folds[k].append(v)
        for k, v in _fold_metrics(y_va, y_prob, opt_thr).items():
            opt_folds[k].append(v)

    return {
        "default":    _aggregate(def_folds),
        "optimal":    _aggregate(opt_folds),
        "thresholds": {"mean": np.mean(thr_folds), "std": np.std(thr_folds), "folds": thr_folds},
    }


def get_oof_predictions(pipeline, X: pd.DataFrame, y: pd.Series, cv) -> np.ndarray:
    """PR curve 시각화용 OOF 확률값 반환. fold마다 clone() 사용."""
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X
    y = pd.Series(y)    if not isinstance(y, pd.Series)    else y

    oof = np.zeros(len(y))
    for train_idx, val_idx in cv.split(X, y):
        est = clone(pipeline)
        est.fit(X.iloc[train_idx], y.iloc[train_idx])
        oof[val_idx] = est.predict_proba(X.iloc[val_idx])[:, 1]
    return oof
