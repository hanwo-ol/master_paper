# 실험 현황표

### 1. K값 확장에 따른 실험 현황 및 계획

| 실험 코드 | 실험 내용 | K=4 | K=3,5,6 | 비고 |
| :--- | :--- | :--- | :--- | :--- |
| **`005_preprocessing.py`** | 데이터 생성 | ✅ 완료 | 3만 실행함, **5-6 실행 필요** | **가장 먼저 수행** |
| **`011_meta_trainer.py`** | Meta-Model 학습 | ✅ 완료 | **🔄 재실행 필요** | K별로 모델 생성 |
| **`011b_pooled_trainer.py`** | Pooled Model 학습 | ✅ 완료 | ❌ 불필요 | Pooled는 K와 무관 |
| **`014_baselines.py`** | LSTM/TF 학습 | ✅ 완료 | ❌ 불필요 | Baseline은 K와 무관 |
| **`012_backtester.py`** | 최종 백테스트 | ✅ 완료 | **🔄 재실행 필요** | K별 최종 성과 비교 |
| **`013_performance_analysis.py`** | 성과 분석 | ✅ 완료 | **🔄 재실행 필요** | K별 결과 분석 |
| **`018_significance_test.py`** | 통계적 유의성 | ✅ 완료 | **🔄 재실행 필요** | K별 유의성 확인 |
| **`019_chebyshev_geometry.py`** | Chebyshev Geometry | ✅ 완료 | **🔄 재실행 필요** | K별 기하학적 구조 변화 |
| **`021_...` 시리즈** | Theorem 4 검증 | ✅ 완료 | **🔄 재실행 필요** | **K에 따른 Hard/Soft 변화가 핵심** |
| **`016_ablation_studies.py`** | Attention Ablation | ⏳ 실행중 | ❌ 불필요 | K=4에서만 비교해도 충분 |
| **`022_skip_connection.py`** | Skip Ablation | ⏳ 실행 예정 | ❌ 불필요 | K=4에서만 비교해도 충분 |
| **`exp1_...`, `exp2_...`** | 기초 검증 | ✅ 완료 | ❌ 불필요 | 논리의 근본 가정은 K=4로 증명 |
| **`020_dominance_ratio.py`** | Prop 2 검증 | ✅ 완료 | ❌ 불필요 | 이론 검증은 K와 무관 |

**실행 순서:**
1.  `005_preprocessing.py` (for K=3,5,6)
2.  `011_meta_trainer.py` (for K=3,5,6)
3.  `012_backtester.py` 및 `013_...`, `018_...` (for K=3,5,6)
4.  `019_...`, `021_...` 시리즈 (for K=3,5,6)

---

### 2. 논문의 깊이를 더할 추가 실험 제안 (분야별)

#### 📈 통계학 (Statistical Learning) 관점

1.  **Information Criterion for K Selection:**
    *   **아이디어:** "왜 K=4가 최적인가?"에 대한 통계적 근거 제시.
    *   **방법:** 각 K에 대해 Regime별 예측 오차의 분산-공분산 행렬($\Sigma_k$)을 계산하고, 이를 이용해 **BIC(Bayesian Information Criterion)** 계산.
    *   **수식:** $BIC(K) = -2 \cdot \ln(\mathcal{L}) + p \cdot \ln(N)$, 여기서 $\mathcal{L}$은 최대우도, $p$는 파라미터 수.
    *   **인사이트:** BIC가 가장 낮은 K가 통계적으로 가장 적절한 Regime 개수임을 보여줌.

2.  **Causal Inference (Granger Causality):**
    *   **아이디어:** Regime 전환이 실제로 포트폴리오 수익률에 "원인(Cause)"으로 작용하는지 검증.
    *   **방법:** Regime 전환 시점 더미 변수($D_t$)와 포트폴리오 수익률($R_t$) 간의 **Granger 인과관계 테스트** 수행.
    *   **수식:** $R_t = \sum \alpha_i R_{t-i} + \sum \beta_j D_{t-j} + \epsilon_t$. 귀무가설 $H_0: \beta_1 = ... = \beta_j = 0$.
    *   **인사이트:** Regime 변화가 미래 수익률 예측에 유의미한 정보를 제공함을 보여줌.

3.  **Conformal Prediction (Uncertainty Quantification):**
    *   **아이디어:** "내일 수익률은 0.1%일 것이다"가 아니라, **"내일 수익률은 90% 확률로 [-0.5%, 0.7%] 구간에 있을 것이다"**라고 예측.
    *   **방법:** Meta-Model의 예측 오차를 이용하여 **Conformal Prediction** 신뢰구간(Confidence Interval) 생성.
    *   **수식:** $C(X_{t+1}) = [\hat{y}_{t+1} - q, \hat{y}_{t+1} + q]$, 여기서 $q$는 Calibration Set 오차의 $(1-\alpha)$ 분위수.
    *   **인사이트:** 모델의 불확실성을 정량화하여, 리스크 관리 시스템과 연동할 수 있음을 보여줌.

#### 💰 금융학 & 퀀트 (Quantitative Finance) 관점

1.  **Factor Exposure Analysis:**
    *   **아이디어:** 우리 모델이 어떤 팩터(가치, 모멘텀, 사이즈 등)에 베팅하여 돈을 버는지 분석.
    *   **방법:** 백테스트 수익률을 Fama-French 5 Factors 등으로 **회귀분석**.
    *   **수식:** $R_t - R_{f,t} = \alpha + \beta_{mkt}(R_{m,t}-R_{f,t}) + \beta_{smb}SMB_t + ... + \epsilon_t$.
    *   **인사이트:** 모델의 알파($\alpha$)가 유의미한지, 특정 팩터에 과도하게 편중되지 않았는지 확인.

2.  **Regime-Conditional Factor Timing:**
    *   **아이디어:** 우리 모델이 Regime에 따라 팩터 베팅을 동적으로 조절하는지 확인.
    *   **방법:** 각 Regime(Bull, Bear 등)별로 수익률을 잘라서 팩터 회귀분석 수행.
    *   **수식:** 위 회귀분석을 Regime별로 따로 실행.
    *   **인사이트:** "Bear Market에서는 Quality 팩터에 대한 노출($\beta_{q}$)이 높아지고, Bull Market에서는 Momentum 팩터 노출($\beta_{mom}$)이 높아진다"와 같은 동적 전략을 보여줌.

3.  **Capacity Analysis (AUM-Adjusted Performance):**
    *   **아이디어:** "이 전략으로 얼마까지 운용할 수 있는가?" (운용자산(AUM)이 커지면 거래 비용, 시장 충격 비용 증가)
    *   **방법:** Turnover와 거래량 데이터를 이용해 **시장 충격 비용(Market Impact Cost)**을 모델링하고, AUM 규모에 따른 성능 하락 곡선 시뮬레이션.
    *   **수식:** $Cost = \text{const} \cdot \sigma \cdot (\frac{\text{TradeSize}}{\text{ADV}})^{\gamma}$, 여기서 ADV는 일평균거래량.
    *   **인사이트:** 전략의 확장성(Scalability)을 제시하여 학문적 연구를 넘어 실제 운용 가능성을 어필.

#### 🤖 모델링 (Modeling) 관점

1.  **Attention Map Visualization:**
    *   **아이디어:** 모델이 예측 시 어떤 자산/시간에 집중하는지 시각화.
    *   **방법:** U-Net의 Self-Attention Block에서 Attention Weight Matrix를 추출하여 히트맵으로 시각화.
    *   **인사이트:** "2008년 위기 직전, 모델은 금융 섹터 간의 상관관계에 집중했다"와 같은 해석 가능한 사례 연구(Case Study) 제시.

2.  **Feature Importance (SHAP or Permutation):**
    *   **아이디어:** 수십 개의 Feature 중 어떤 것이 예측에 가장 중요한지 분석.
    *   **방법:** **SHAP (SHapley Additive exPlanations)** 라이브러리를 적용하거나, 특정 Feature를 랜덤하게 섞었을 때(Permutation) 성능이 얼마나 떨어지는지 측정.
    *   **인사이트:** 모델이 VIX, 금리 등 특정 거시 변수에 민감하게 반응함을 보여주어, Regime 인식 메커니즘을 간접적으로 설명.

3.  **Meta-Parameter Sensitivity:**
    *   **아이디어:** Inner LR, Inner Step 수 등 Meta-Learning 하이퍼파라미터에 결과가 얼마나 민감한지 분석.
    *   **방법:** `inner_lr`을 $\{0.001, 0.01, 0.1\}$로, `n_inner_steps`를 $\{1, 3, 5, 10\}$으로 바꿔가며 Validation Set 성능 비교.
    *   **인사이트:** 모델이 넓은 범위의 하이퍼파라미터에 대해 강건함(Robust)을 보여주어, 튜닝의 어려움이 적음을 어필.
