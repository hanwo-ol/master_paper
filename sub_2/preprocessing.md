### **Formal Definition of Strict 2-Stage Preprocessing**

우리는 데이터 전처리 및 태스크 생성 과정을 계산 비용이 높은 **불변(Invariant) 단계**와 실험 설정에 따라 달라지는 **가변(Variant) 단계**로 분리한다.

#### **1. Notations (기호 정의)**

*   **Time & Assets**: $t \in \{1, \dots, T\}$는 시간, $n \in \{1, \dots, N\}$은 자산을 나타낸다.
*   **Data Split**: 전체 시간 집합 $\mathcal{T}$를 상호 배타적인 세 구간으로 엄격히 분할한다.
    $$ \mathcal{T} = \mathcal{T}_{\text{train}} \cup \mathcal{T}_{\text{val}} \cup \mathcal{T}_{\text{test}}, \quad \text{where } \sup(\mathcal{T}_{\text{train}}) < \inf(\mathcal{T}_{\text{val}}) $$
*   **Raw Data**: $\mathbf{X}^{\text{raw}} \in \mathbb{R}^{T \times N \times F}$는 입력 피처, $\mathbf{Y}^{\text{raw}} \in \mathbb{R}^{T \times N}$은 일별 수익률이다.

---

#### **Stage 1: Base Processing (Invariant Process)**

이 단계의 목표는 실험 설정($K, \tau$ 등)과 무관하게, **정보 누수(Data Leakage) 없는** 정제된 데이터 텐서를 구축하는 것이다.

**1.1. Strict Statistics Calculation (엄밀한 통계량 산출)**
모든 전처리 파라미터 $\Theta$는 오직 학습 기간 $\mathcal{T}_{\text{train}}$의 데이터만을 사용하여 추정한다.

$$
\Theta_{\text{imp}} = \text{Median}\left( \{ \mathbf{X}^{\text{raw}}_{t,n} \mid t \in \mathcal{T}_{\text{train}} \} \right) \quad (\text{Imputation})
$$
$$
\boldsymbol{\mu}_f, \boldsymbol{\sigma}_f = \text{Mean/Std}\left( \{ \mathbf{X}^{\text{raw}}_{t,n,f} \mid t \in \mathcal{T}_{\text{train}} \} \right) \quad (\text{Feature Scaling})
$$
$$
\mu_y, \sigma_y = \text{Mean/Std}\left( \{ \mathbf{Y}^{\text{raw}}_{t,n} \mid t \in \mathcal{T}_{\text{train}} \} \right) \quad (\text{Target Scaling})
$$

**1.2. Transformation (전체 변환)**
추정된 파라미터를 사용하여 전체 시계열 데이터를 변환한다. 이때 극단값 처리를 위한 클리핑 함수 $\phi_{\lambda}(\cdot)$를 적용한다.

$$
\mathbf{Z}_{t,n,f} = \frac{\mathbf{X}^{\text{raw}}_{t,n,f} - \boldsymbol{\mu}_f}{\boldsymbol{\sigma}_f}, \quad \tilde{Y}_{t,n} = \phi_{5\sigma}\left( \frac{\mathbf{Y}^{\text{raw}}_{t,n} - \mu_y}{\sigma_y} \right)
$$

*   $\mathbf{Z}$: 모델의 입력으로 사용될 정규화된 피처 텐서.
*   $\tilde{Y}$: 모델 학습(Loss 계산)에 사용될 정규화된 타겟.
*   $\mathbf{Y}^{\text{raw}}$: 백테스팅(P&L 계산)을 위해 **보존**된다.

**1.3. Market State Aggregation (시장 상태 요약)**
레짐 식별을 위해 횡단면(Cross-sectional) 정보를 집계하여 시장 상태 벡터 $\mathbf{m}_t \in \mathbb{R}^D$를 생성한다.
$$
\mathbf{m}_t = \left[ \frac{1}{N}\sum_n \mathbf{Y}^{\text{raw}}_{t,n}, \quad \text{Std}_n(\mathbf{Y}^{\text{raw}}_{t,n}), \quad \text{VIX}_t, \dots \right]^\top
$$

---

#### **Stage 2: Metadata & Task Generation (Variant Process)**

이 단계는 실험 변수 $\Omega = \{K, \tau, L, \mathcal{U}\}$에 따라 메타데이터(인덱스 집합)를 생성한다.

**2.1. Strict Regime Identification (학습 데이터 기반 레짐 정의)**
시장 레짐 $s_t$를 식별하기 위한 클러스터링 함수 $C$ 역시 $\mathcal{T}_{\text{train}}$ 내의 시장 상태 벡터들로만 학습된다.

$$
C^* = \arg\min_{C} \sum_{t \in \mathcal{T}_{\text{train}}} \sum_{k=1}^K \mathbb{1}(C(\mathbf{m}_t) = k) \cdot \| \mathbf{m}_t - \mathbf{c}_k \|^2
$$
전체 기간에 대한 레짐은 학습된 함수 $C^*$를 적용하여 얻는다.
$$
s_t = C^*(\mathbf{m}_t) \in \{1, \dots, K\}, \quad \forall t \in \mathcal{T}
$$

**2.2. Task(Episode) Construction**
메타러닝을 위한 태스크 $\mathcal{T}_i$는 룩백 윈도우 $L$과 순도 임계값 $\tau$, 자산 유니버스 $\mathcal{U}$에 의해 정의된다.

Support 기간 $W^{\text{sup}} = [t, t+L-1]$과 Query 기간 $W^{\text{qry}} = [t+L, t+2L-1]$에 대하여, 다음 조건을 만족하는 구간만을 에피소드로 채택한다.

1.  **Regime Consistency & Purity**:
    $$ \text{Mode}(s_{t'} \mid t' \in W^{\text{sup}}) = \text{Mode}(s_{t'} \mid t' \in W^{\text{qry}}) = k $$
    $$ \frac{1}{L} \sum_{t' \in W^{\text{sup}}} \mathbb{1}(s_{t'} = k) \ge \tau \quad \land \quad \frac{1}{L} \sum_{t' \in W^{\text{qry}}} \mathbb{1}(s_{t'} = k) \ge \tau $$

2.  **Data Integrity**: 구간 내 데이터가 유니버스 $\mathcal{U}$에 속하는 자산들로 구성됨.

최종적으로 생성된 태스크 집합 $\mathcal{E}(\Omega)$는 데이터 $\mathbf{Z}, \tilde{Y}$의 복사본이 아닌, **시간 인덱스와 레짐 라벨의 집합**으로 저장된다.

$$
\mathcal{E}(\Omega) = \{ (t_{\text{start}}^{(i)}, k^{(i)}) \}_{i=1}^{M}
$$

---

### **논문에서의 강조점**

1.  **엄밀성 (Rigor)**: 모든 통계적 파라미터($\mu, \sigma, C^*$)가 $\mathcal{T}_{\text{train}}$에서만 유도되었으므로, **Look-ahead Bias가 수학적으로 불가능함**을 보였습니다.
2.  **효율성 (Efficiency)**: $\mathbf{Z}$와 $\tilde{Y}$는 한 번만 계산되어 고정(Frozen)되고, 다양한 실험 조건($\Omega$)에 따라 가벼운 인덱스 집합 $\mathcal{E}$만 재생성되므로 실험 속도가 극대화됨을 수식으로 표현했습니다.
3.  **이원화된 타겟 (Dual Target)**: 학습 최적화를 위한 $\tilde{Y}$ (Scaled)와 투자 성과 평가를 위한 $\mathbf{Y}^{\text{raw}}$ (Raw)를 명시적으로 분리하여 정의했습니다.

이 노테이션을 논문의 Method 섹션이나 Appendix의 Experimental Setup 파트에 포함시키면, 실험 설계의 견고함을 심사위원들에게 강력하게 어필할 수 있습니다.
