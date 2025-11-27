네, 전처리 파이프라인을 엄밀하게 재구축했으므로, 이에 맞춰 **실험 현황표(Experiment Status Table)**와 **실험 세부 리스트**를 새롭게 정리해 드리겠습니다.

모든 실험은 **Stage 1 (Base Processing)** → **Stage 2 (Episode Generation)** → **Model Training** → **Analysis** 순서로 진행되어야 합니다.

---

### **최종 실험 현황표 (Experiment Status Table)**

현재 상태는 모두 **[대기 (Pending)]** 로 초기화되었습니다. 진행하면서 이 표를 업데이트하시면 됩니다.

| 단계 | ID | 실험명 | 의존성 (선행 작업) | 주요 스크립트 | 상태 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Phase 0** | **D0.1** | **Base Preprocessing** | - | `preprocess_base.py` | done|
| | **D0.2** | **Episode Generation (All Configs)** | D0.1 | `preprocess_episodes.py` | done |
| **Phase 1** | E1.1 | Gradient Alignment (MAML Justification) | D0.2 | `exp1_gradient_alignment.py` | ⬜ |
| (기반 검증) | E1.2 | HRP Structural Stability | D0.1 | `exp2_covariance_stability.py` | ⬜ |
| | E1.3 | HRP Learning Efficiency | D0.2 | `exp2b_learning_efficiency.py` | ⬜ |
| | E1.4 | Dominance Ratio (Prop 2) | D0.1 | `020_dominance_ratio.py` | ⬜ |
| **Phase 2** | **TR.1** | **Meta-Model Training (Core: K=4)** | D0.2 | `011_meta_trainer_epoch_save.py` | ⬜ |
| (성능 평가) | TR.2 | Pooled Model Training (Baseline) | D0.2 | `011b_pooled_trainer.py` | ⬜ |
| | TR.3 | Baselines Training (LSTM/TF) | D0.2 | `baselines.py` | ⬜ |
| | E2.1 | Comprehensive Backtest | TR.1, 2, 3 | `012b_backtester_ablation.py` | ⬜ |
| | E2.2 | Performance Metrics & Plots | E2.1 | `013_performance_analysis.py` | ⬜ |
| | E2.3 | Statistical Significance | E2.1 | `018_significance_test.py` | ⬜ |
| **Phase 3** | E3.1 | Architecture Ablation (No-Attn) | D0.2 (학습필요) | `016_ablation_studies.py` | ⬜ |
| (민감도) | E3.2 | Adaptation Diagnostics | TR.1 | `013_diagnostics.py` | ⬜ |
| | E3.3 | Sensitivity: K (Regime Count) | D0.2 (K=3,5,6 학습) | `024_bic_for_k.py` | ⬜ |
| | E3.4 | Sensitivity: Tau (Purity) | D0.2 (Tau 변형 학습) | (Backtester 활용) | ⬜ |
| | E3.5 | Sensitivity: Cost (Kappa) | E2.1 | (Backtester 활용) | ⬜ |
| | E3.6 | Crisis Period Analysis | E2.1 | `013_performance_analysis.py` | ⬜ |
| **Phase 4** | E4.1 | Chebyshev Geometry (Thm 3) | TR.1 | `019_chebyshev_geometry.py` | ⬜ |
| (심층 분석) | E4.2 | Noise Robustness (Thm 4) | TR.1 | `021_posterior_noise.py` | ⬜ |
| | E4.3 | Uncertainty Quantification | TR.1 | `026_conformal_...` | ⬜ |
| | E4.4 | Gradient Alignment Evolution | TR.1 (Epochs) | `exp1c_alignment_over...` | ⬜ |
| | E4.5 | Regime Definition (VIX vs KMeans) | D0.2 (VIX) | `exp1d_vix_vs_kmeans...` | ⬜ |
| **Phase 5** | S1 | HRP & Sector Structure | D0.1 | (New Script) | ⬜ |
| (섹터 분석) | S2 | Regime-Driven Sector Rotation | E2.1 | (New Script) | ⬜ |
| | S3 | Sector-Neutral Alpha | E2.1 | (New Script) | ⬜ |

---

### **실험 목록 상세 가이드 (Step-by-Step Guide)**

실험은 반드시 아래 순서대로 진행해야 데이터 의존성 문제가 발생하지 않습니다.

#### **Step 0: 데이터 준비 (Phase 0)**
가장 먼저 엄밀한 전처리를 수행합니다.
1.  `python preprocess_base.py` (Clean Panel, Scalers, Base Artifacts 생성)
2.  `python preprocess_episodes.py` (Core, Sensitivity, Sector 등 모든 에피소드 JSON 생성)

#### **Step 1: 기반 이론 검증 (Phase 1)**
데이터만 있으면 모델 학습 전에도 수행 가능한 실험들입니다.
1.  `python exp1_gradient_alignment.py` (이건 모델이 필요하므로, TR.1 이후 또는 Warm-up만 하고 수행) -> **수정: MAML 학습 전 초기 모델로 수행 가능**
2.  `python exp2_covariance_stability.py`
3.  `python exp2b_learning_efficiency.py` (간단한 학습 포함됨)
4.  `python 020_dominance_ratio.py`

#### **Step 2: 모델 학습 (Phase 2 - Training)**
메인 모델과 비교군들을 학습시킵니다. (GPU 시간 소요)
1.  **Core Model**: `python 011_meta_trainer_epoch_save.py` (K=4, Tau=0.60)
2.  **Pooled Model**: `python 011b_pooled_trainer.py`
3.  **Baselines**: `python baselines.py` (LSTM, Transformer)
4.  **Ablation**: `python 016_ablation_studies.py` (No-Attention 모델 등)
5.  **Sensitivity Models**: `preprocess_episodes.py`에서 생성한 K=3,5,6 및 Tau 변형 데이터셋들에 대해 `011_...` 스크립트를 인자만 바꿔서 각각 실행 (시간이 많이 걸리므로 Core 실험 후 순차 진행 권장).

#### **Step 3: 메인 성과 평가 (Phase 2 - Evaluation)**
학습된 모델들을 바탕으로 백테스트를 수행합니다.
1.  `python 012b_backtester_ablation.py` (모든 모델 로드 후 통합 백테스트)
2.  `python 013_performance_analysis.py` (결과 시각화)
3.  `python 018_significance_test.py` (통계 검정)

#### **Step 4: 심층 분석 및 보강 (Phase 3, 4, 5)**
리뷰어 방어 논리를 위한 깊이 있는 분석들입니다.
1.  **보강 실험**:
    *   `python exp1c_alignment_over_training.py` (학습 중 그래디언트 변화)
    *   `python exp1d_vix_vs_kmeans_regimes.py` (VIX 레짐 비교)
2.  **이론 검증**:
    *   `python 019_chebyshev_geometry.py`
    *   `python 021_posterior_noise.py`
    *   `python 026_conformal_prediction_adaptive.py`
3.  **섹터 분석**:
    *   (S1~S3에 해당하는 스크립트는 추후 작성 및 실행)
