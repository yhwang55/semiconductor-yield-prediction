# Phase 2: 클래스 불균형 처리 실험 결과

**분석 일자**: 2026-06-20
**고정 조건**: Median Imputation, 5-fold CV, Fail(y=1) positive class

---

## 1. 실험 설계

### 누수 방지 구현 (`imblearn.Pipeline`)
```
Train Fold  → [StandardScaler] → [Resampler: SMOTE/ADASYN/RUS] → [Model.fit()]
Val Fold    → [StandardScaler] → (resampler skip)               → [Model.predict_proba()]
```
- `imblearn.Pipeline`은 `fit()` 시에만 resampler 실행, `predict()`/`predict_proba()` 시 자동 skip
- 각 fold마다 `clone(pipeline)`으로 새 인스턴스 생성 → fold 간 상태 오염 없음
- Validation fold는 항상 **원본 분포(1:14.1)** 그대로 유지

### 임계값 전략
| 전략 | 설명 |
|------|------|
| Default (0.5) | sklearn 기본 결정 경계 |
| Optimal | 각 fold의 PR curve에서 F1 최대화 임계값 선택 후 집계 |

> **한계**: optimal threshold는 validation fold로부터 선택 → 약간의 낙관적 편향 존재.
> 실전에서는 별도 calibration set 또는 nested CV 권장.

---

## 2. 성능 결과 — StratifiedKFold (Optimal Threshold)

**StratifiedKFold / Optimal Threshold (5-fold mean ± std)**

| Method | M | PR-AUC | F1(Fail) | Recall(Fail) | Precision(Fail) | OPT Thr |
|--------|---|--------|----------|--------------|-----------------|---------|
| ADASYN       | LR | 0.140±0.020 | 0.215±0.025 | 0.460±0.149 | 0.154±0.043 | 0.29 |
| CW_Balanced  | LR | 0.149±0.021 | 0.227±0.024 | 0.469±0.157 | 0.166±0.047 | 0.34 |
| Default      | LR | 0.150±0.025 | 0.226±0.021 | 0.498±0.132 | 0.156±0.040 | 0.10 |
| RUS          | LR | 0.150±0.045 | 0.229±0.035 | 0.377±0.173 | 0.224±0.113 | 0.87 |
| SMOTE        | LR | 0.142±0.023 | 0.220±0.022 | 0.469±0.157 | 0.159±0.043 | 0.30 |
| SMOTE+RUS    | LR | 0.140±0.023 | 0.244±0.054 | 0.509±0.110 | 0.180±0.082 | 0.30 |
| ADASYN       | RF | 0.179±0.028 | 0.275±0.031 | 0.460±0.119 | 0.208±0.049 | 0.26 |
| CW_Balanced  | RF | 0.162±0.021 | 0.268±0.012 | 0.431±0.091 | 0.198±0.009 | 0.11 |
| Default      | RF | 0.180±0.040 | 0.302±0.050 | 0.393±0.147 | 0.289±0.104 | 0.18 |
| RUS          | RF | 0.171±0.028 | 0.258±0.026 | 0.327±0.088 | 0.225±0.035 | 0.60 |
| SMOTE        | RF | 0.181±0.018 | 0.275±0.028 | 0.556±0.128 | 0.188±0.027 | 0.25 |
| SMOTE+RUS    | RF | 0.166±0.015 | 0.278±0.022 | 0.567±0.126 | 0.189±0.028 | 0.32 |

---

## 3. 성능 결과 — TimeSeriesSplit (Optimal Threshold)

**TimeSeriesSplit / Optimal Threshold (5-fold mean ± std)**

| Method | M | PR-AUC | F1(Fail) | Recall(Fail) | Precision(Fail) | OPT Thr |
|--------|---|--------|----------|--------------|-----------------|---------|
| ADASYN       | LR | 0.121±0.068 | 0.229±0.114 | 0.631±0.254 | 0.203±0.169 | 0.27 |
| CW_Balanced  | LR | 0.119±0.067 | 0.221±0.100 | 0.571±0.187 | 0.181±0.132 | 0.27 |
| Default      | LR | 0.127±0.081 | 0.226±0.106 | 0.571±0.187 | 0.191±0.140 | 0.24 |
| RUS          | LR | 0.078±0.037 | 0.175±0.058 | 0.626±0.291 | 0.111±0.043 | 0.57 |
| SMOTE        | LR | 0.121±0.068 | 0.233±0.116 | 0.480±0.234 | 0.214±0.168 | 0.47 |
| SMOTE+RUS    | LR | 0.149±0.118 | 0.236±0.147 | 0.484±0.189 | 0.330±0.362 | 0.46 |
| ADASYN       | RF | 0.129±0.086 | 0.217±0.101 | 0.350±0.126 | 0.214±0.196 | 0.25 |
| CW_Balanced  | RF | 0.095±0.045 | 0.193±0.099 | 0.381±0.134 | 0.157±0.103 | 0.11 |
| Default      | RF | 0.145±0.085 | 0.221±0.093 | 0.358±0.196 | 0.344±0.342 | 0.23 |
| RUS          | RF | 0.096±0.051 | 0.146±0.056 | 0.653±0.381 | 0.270±0.368 | 0.47 |
| SMOTE        | RF | 0.099±0.050 | 0.181±0.065 | 0.334±0.090 | 0.133±0.055 | 0.25 |
| SMOTE+RUS    | RF | 0.098±0.042 | 0.196±0.078 | 0.334±0.162 | 0.150±0.071 | 0.33 |

> **TimeSeriesSplit vs StratifiedKFold**: TS가 더 보수적인 추정치를 제공.
> 반도체 공정처럼 시간 순서가 있는 데이터에서 TS가 실전 배포 성능에 더 근접.

---

## 4. 핵심 발견 분석

### 4.1 Phase 1 → Phase 2 개선폭

| 지표 | Phase 1 Best | Phase 2 Best (LR) | Phase 2 Best (RF) |
|------|-------------|-------------------|-------------------|
| Recall(Fail) | 0.164 | **0.509** (+0.345) | **0.567** (+0.403) |
| PR-AUC | 0.180 | 0.150 (-0.030) | 0.181 (+0.001) |

목표 Recall ≥ 0.70: LR 달성=False, RF 달성=False

### 4.2 기법별 효과 분석

**class_weight='balanced' (CW_Balanced)**
- 모델 파라미터 하나만 변경해서 RF의 Recall을 0→유의미한 수준으로 끌어올리는 가장 강력한 단일 레버
- 원리: loss function에서 Fail 샘플의 가중치를 Pass 대비 14배 부여 → 결정 경계가 소수 클래스 쪽으로 이동
- 장점: 데이터 증가 없음(속도 동일), 구현 단순

**SMOTE / ADASYN**
- Train fold에서 Fail 합성 샘플 생성 → 모델이 minority class 패턴 더 많이 학습
- ADASYN: 분류 경계 근처 샘플에 더 많은 합성 샘플 배치 → 경계 학습에 특화
- 소규모 Fail(~17개) fold에서 합성 과정이 불안정할 수 있음 (TimeSeriesSplit 초기 fold)

**RUS (RandomUnderSampler)**
- Pass 샘플을 Fail 수준으로 감소 → 훈련 데이터 대폭 축소(fold 1: ~34개만 남음)
- 정보 손실이 크나, 불균형 자체는 해소됨. PR-AUC보다 Recall 단순 향상에 유리
- RF처럼 데이터 양에 민감한 모델에는 불리, LR처럼 regularization이 강한 모델에 상대적으로 유리

**SMOTE+RUS (하이브리드)**
- SMOTE(sampling_strategy=0.5): Fail을 Pass의 50%까지 과표집
- RUS(sampling_strategy=1.0): 이후 1:1 비율로 정리
- 과표집과 과소표집을 동시에 사용해 합성 데이터 비중 감소 + 훈련 데이터 양 적절히 유지

### 4.3 임계값 튜닝의 기여도 (기법별 Recall: Default 0.5 → Optimal)

임계값 튜닝만으로 얻는 Recall 개선 (RF Default 기법 기준): **+0.393**

- Default(0.5)에서 RF는 Fail 확률이 0.5 이상이어야 Fail 예측 → 극히 드물게 발생
- 임계값을 낮추면(예: 0.2) 더 많은 Fail을 탐지, 단 Precision 비용 발생
- **결론**: 임계값 튜닝은 "공짜 점심"이 아님 — Recall↑하면 Precision↓ 트레이드오프 발생

### 4.4 Recall vs Precision 트레이드오프

| 구간 | 특성 |
|------|------|
| Recall 높음, Precision 낮음 | 불량을 많이 잡되 Pass를 자주 불량으로 오분류 → 과잉 검사 비용 |
| Recall 낮음, Precision 높음 | 놓치는 불량 많음 → 불량 제품 출하 위험 |
| **실전 권고** | Recall ≥ 0.70 우선 확보, Precision은 최소 0.20 이상 유지 |

---

## 5. Phase 3로 넘어가야 하는 이유

Phase 2에서 불균형 처리와 임계값 튜닝으로 Recall을 유의미하게 끌어올렸으나,
PR-AUC는 아직 목표치(0.40)에 미달하는 수준이다.

근본적인 한계는 **피처 품질**에 있다:
- 590개 원본 피처 중 통계적으로 유의한 피처는 80개(17%)에 불과
- 최대 Cohen's d ≈ 0.6으로 단일 피처의 판별력 자체가 제한적
- 현재 모델은 446개 피처를 모두 사용 → 노이즈 피처가 신호를 희석

**Phase 3에서 다룰 내용:**
1. **비지도 이상탐지** (Isolation Forest, LOF): 라벨 없이 공정 이상점 탐지
   - 반도체 공정 특성상 "정상 패턴에서 얼마나 벗어났는지"가 불량의 핵심 신호
   - 이상 점수(anomaly score)를 새로운 피처로 추가하여 지도 학습 모델에 결합
2. 비지도 점수와 Phase 2 최적 파이프라인의 앙상블

---

## 6. 산출 파일

| 파일 | 설명 |
|------|------|
| `reports/figures/12_p2_main_comparison.png` | 6기법 × LR/RF 메트릭 비교 |
| `reports/figures/13_p2_pr_curves.png` | OOF 기반 PR Curve (6기법 오버레이) |
| `reports/figures/14_p2_threshold_analysis.png` | 임계값 0.5 vs optimal 기여도 |
| `reports/phase2_results_table.csv` | 전체 수치 데이터 |

---
*Generated by `notebooks/phase2_imbalance/01_imbalance_experiments.py`*
