

터미널 출력에 기록된 최적 손실 함수 값 **`0.172033`**이 산출된 수학적 공식과 사용된 기재(Notation)의 세부 명세는 다음과 같습니다.

최적 파라미터 집합 $\theta^* = (d^*, r^*, m^*) = (0.44\text{s}, 0.16, 0.015\text{m})$을 대입했을 때, 각 대상 시도(Trial)에 대한 손실값의 평균으로 계산됩니다.

---

### 1. 수학적 기호 정의 (Notation)

| 기호 | 물리적 정의 | 수식 / 코드 대응 |
| :--- | :--- | :--- |
| $M$ | 전체 유효 6m Walking 시도(Trial) 세트 | `len(trials)` = 233 |
| $m$ | 개별 시도 인덱스 ($m \in \{1, 2, \dots, M\}$) | `trial` |
| $\theta$ | 하이퍼파라미터 벡터 $\theta = (d, r, m_0)$ | `(min_dist_sec, prom_ratio, min_prominence_m)` |
| $T_{S, m}$ | $m$번째 시도에서 감지된 $S$측 발($S \in \{L, R\}$)의 Stride Time 시계열 | `L_strides`, `R_strides` |
| $N_{S, m}$ | $m$번째 시도에서 감지된 $S$측 Stride(보폭)의 총 개수 | `len(L_strides)`, `len(R_strides)` |
| $N_{HS, S, m}$ | $m$번째 시도에서 감지된 $S$측 Heel Strike의 총 개수 | `len(L_hs)`, `len(R_hs)` |
| $\mu(T_{S, m})$ | $m$번째 시도에서 $S$측 Stride Time의 평균값 | `np.mean(strides)` |
| $\sigma(T_{S, m})$ | $m$번째 시도에서 $S$측 Stride Time의 표준편차 | `np.std(strides)` |
| $CV(T_{S, m})$ | $m$번째 시도에서 $S$측 Stride Time의 변동계수(Coefficient of Variation) | `cv_L`, `cv_R` |

---

### 2. 단일 시도 손실 함수 $L_m(\theta)$ 수식
각 Trial $m$에 대한 개별 손실 함수 $L_m(\theta)$는 **(1) 주기성 손실**과 **(2) 생리학적 규제화 오차**의 합으로 정의됩니다.

$$L_m(\theta) = \text{Loss}_{\text{periodicity}, m}(\theta) + \text{Reg}_{\text{count}, m}(\theta) + \text{Reg}_{\text{bounds}, m}(\theta) + \text{Reg}_{\text{balance}, m}(\theta)$$

#### ① Stride 주기성 오차 ($\text{Loss}_{\text{periodicity}, m}$)
좌우 독립적 Stride Time 변동계수의 산술평균입니다. 걸음 수가 부족하여 변동성 추정이 불가능할 경우 벌점 상수 $P_{\text{fail}} = 2.0$을 부여합니다.

$$\text{Loss}_{\text{periodicity}, m}(\theta) = \frac{CV(T_{L, m}) + CV(T_{R, m})}{2}$$

$$\text{where}\quad CV(T_{S, m}) = \begin{cases} 
\frac{\sigma(T_{S, m})}{\mu(T_{S, m})} & \text{if } N_{S, m} \ge 3 \\
2.0 & \text{if } N_{S, m} < 3
\end{cases}$$

#### ② Stride 횟수 규제화 ($\text{Reg}_{\text{count}, m}$)
6m 보행 시 정상적인 보폭 범위인 $[3, 9]$를 벗어날 경우 부과되는 페널티입니다.
$$\text{Reg}_{\text{count}, m}(\theta) = \sum_{S \in \{L, R\}} \left( 0.5 \times \max(0, 3 - N_{S, m})^2 + 0.2 \times \max(0, N_{S, m} - 9)^2 \right)$$

#### ③ Stride 시간 범위 규제화 ($\text{Reg}_{\text{bounds}, m}$)
평균 보폭 주기(Stride Time)가 일반 생리적 범위인 $[0.7\text{s}, 2.0\text{s}]$를 탈출할 경우 부과되는 페널티입니다.
$$\text{Reg}_{\text{bounds}, m}(\theta) = \sum_{S \in \{L, R\}} \left( 1.0 \times \max(0, 0.7 - \mu(T_{S, m}))^2 + 1.0 \times \max(0, \mu(T_{S, m}) - 2.0)^2 \right)$$

#### ④ 좌우 대칭성 규제화 ($\text{Reg}_{\text{balance}, m}$)
왼발과 오른발의 검출된 발걸음 수가 불일치할 경우 대칭성 왜곡 벌점을 부과합니다.
$$\text{Reg}_{\text{balance}, m}(\theta) = 0.1 \times \left( N_{HS, L, m} - N_{HS, R, m} \right)^2$$

---

### 3. 최종 목적 함수 (Total Loss Objective) 및 결과 산출
그리드 서치가 최소화하고자 하는 최종 비용 함수 $J(\theta)$는 **전체 유효 시도 $M$개에 대한 $L_m(\theta)$의 산술평균**입니다.

$$J(\theta) = \frac{1}{M} \sum_{m=1}^{M} L_m(\theta)$$

* **최종 산출 계산 식**:
  $$J(\theta^*) = \frac{1}{233} \sum_{m=1}^{233} L_m(0.44, 0.16, 0.015) = 0.172033$$
  
이 결과값은 233개의 다각도 보행 데이터 전체에서 아웃라이어 피크나 가짜 발걸음을 완벽히 억제하고 가장 생리학적으로 대칭적이며 일관적인 Stride 주기를 찾아냈을 때의 평균 오차값입니다.
