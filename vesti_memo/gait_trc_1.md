# Biomechanical Evaluation of Spatiotemporal Gait Parameters and Initiation Hesitancy in Parkinson's Disease: A Standardized 6-Meter Walk-Through Study

---

## 1. Study Cohort & Dataset
본 연구는 다기관 협력 연구 데이터베이스(UNIST, 울산대병원, 전북대병원)에서 수집된 보행 시점 데이터를 기반으로 수행되었습니다. 임상 마스터 테이블(`cleaned _clinical _crf.csv`)의 임상 데이터 품질 통과 기준(Grade 'O') 및 모션 데이터 파일(.trc 및 .mot 파일의 쌍)이 온전하게 매칭되는 233개의 보행 데이터셋을 최종 분석 대상으로 확정하였습니다.

각 집단별 보행 데이터셋의 상세 분포는 다음과 같습니다:
* **Healthy Control (정상 대조군, HC)**: $n = 40$ trials
* **Parkinson's Disease (파킨슨병 환자군, PD)**: $n = 193$ trials
* **Total Dataset**: $N = 233$ trials

---

## 2. Methodology & Standardized Processing

### 2-1. Raw Marker Mapping (원시 마커 매핑 정의)
본 연구는 모션 캡처 시스템(OpenCap 등) 및 다기관 수집 환경에 따른 파일 내 마커 헤더명의 다양성을 통일하기 위해 아래와 같이 원시 마커 매핑 테이블을 정의하고, 이를 기반으로 생체역학 변수들을 도출하였습니다.

| 수식 기호 (Notation) | 생체역학적 정의 (Marker Definition) | 원본 TRC 파일 내 대응 헤더 패턴 (Target Header Pattern) | 역할 및 산출 목적 |
| :--- | :--- | :--- | :--- |
| $\vec{p} _{\text{pelvis}}(t)$ | 골반 중심 마커 (Pelvis Center) | `midHip`, `RHJC _study` | 신체 무게중심(COM) 위치 추정 및 보행 진행 방향축 정의 |
| $\vec{p} _{\text{heel, L}}(t)$ | 왼발 뒤꿈치 마커 (Left Calcaneus) | `L _calc _study`, `LHeel`, `L _calc` | 왼발 착지(Heel Strike) 시점 및 좌측 보폭 거리 산출 |
| $\vec{p} _{\text{heel, R}}(t)$ | 오른발 뒤꿈치 마커 (Right Calcaneus) | `r _calc _study`, `RHeel`, `r _calc` | 오른발 착지(Heel Strike) 시점 및 우측 보폭 거리 산출 |

* **데이터 추출**: 각 TRC 파일의 열(Columns) 중 패턴 매칭을 통해 해당하는 마커의 X, Y, Z 시계열 좌표(Nx3 행렬)를 자동으로 추출하여 연산에 대입합니다.

### 2-2. Data Standardization and Spatial Boundary Filtering
본 연구는 골반 중심의 공간적 6미터 보행 구간과 시작/종료 경계 조건에 대한 수학적 정의를 아래와 같이 정립하여 분석을 수행하였습니다.

```
       [출발선] 0m                                    [종료선] 6m
       P _start (t _start)                             P _target (t _cross)
          |                                             |
          V                                             V
---------[■ (골반 중심)]---------------------------------[■]----------> 보행 진행 방향
          \                                               \
          (첫 발자국 HS _1 검출)                           (6m 경계 통과 후 첫 HS _last)
```

1. **보행 진행 방향축 ($\vec{v} _{\text{forward}}$) 정의**:
   피험자가 걸어가는 방향은 모션 캡처 카메라의 고정된 X, Y, Z 축과 일치하지 않습니다. 따라서 골반 중심 마커 $\vec{p} _{\text{pelvis}}(t)$의 3D 시계열 데이터에서 수평면(X-Z 평면) 상의 좌표를 추출하여 피험자 고유의 보행 축을 정의합니다.
   * 골반의 수평 좌표 시계열을 $\vec{p}(t) = [X _{\text{pelvis}}(t), Z _{\text{pelvis}}(t)]$라고 할 때, 보행 시작 구간(첫 18프레임)의 중앙값과 보행 종료 구간(마지막 18프레임)의 중앙값을 잇는 벡터를 정규화하여 **보행 진행 단위 벡터 $\vec{v} _{\text{forward}}$**를 계산합니다.

$$\vec{v} _{\text{forward}} = \text{normalize}(\text{median}(\vec{p}(t _{\text{end window}})) - \text{median}(\vec{p}(t _{\text{start window}})))$$

2. **골반 전방 변위 ($P _{\text{pelvis}}(t)$) 정의**:
   피험자가 걷는 동안 골반이 보행 진행 방향으로 나아간 1차원 전방 위치는 골반 수평 좌표 $\vec{p}(t)$를 보행 진행 축 $\vec{v} _{\text{forward}}$에 내적(Projection)하여 구합니다.

$$P _{\text{pelvis}}(t) = \vec{p}(t) \cdot \vec{v} _{\text{forward}}$$

3. **물리적 6m 공간 경계선 설정 ($P _{\text{start}}$, $P _{\text{target}}$)**:
   * **시작 기준 위치 ($P _{start}$)**: 보행 신호 입력 시작 후 피험자가 대기 중인 극초반 5프레임 동안의 골반 평균 위치로 정의합니다.

$$P _{\text{start}} = \text{median}(P _{\text{pelvis}}(1 \dots 5))$$

   * **종료 타겟 위치 ($P _{target}$)**: 시작 기준 위치로부터 정확히 6.0미터를 이동한 지점입니다.

$$P _{\text{target}} = P _{\text{start}} + 6.0\text{ m}$$

   * **시간 경계 ($t _{\text{start}}$, $t _{\text{end}}$)**: 골반 위치 $P _{\text{pelvis}}(t)$가 출발점 $P _{\text{start}}$를 벗어나는 첫 프레임의 시간을 $t _{\text{start}}$로, 6m 지점인 $P _{\text{target}}$에 도달하는 순간의 프레임 시간을 $t _{\text{cross}}$로 정의합니다.

4. **Walk-Through 주기 정합 (Step Alignment)**:
   골반이 6.0m 지점을 넘는 순간인 $t _{\text{cross}}$에 데이터 분석을 칼같이 종료하면, 마지막 발걸음이 공중에 떠 있는 상태에서 끊겨 보폭이나 속도 계산에 오류가 생깁니다. 이를 방지하기 위해 **골반이 6m를 통과한 직후 땅에 딛는 첫 번째 발자국(Heel Strike, $HS _{last}$)의 타임스탬프인 $t _{\text{end}}$까지 분석 범위를 연장**합니다. 이렇게 해야 6미터 내에 포함된 모든 걸음이 쪼개지지 않고 온전한 주기(Full Cycle)로 계산됩니다.

---

### 2-3. 발걸음(Heel Strike) 검출 및 무감독 파라미터 최적화

#### ① 발꿈치 착지(Heel Strike, HS) 이벤트 검출 원리
발이 땅에 닿는 순간(HS)을 감지하기 위해 생체역학 표준인 **Zeni et al. (2008) 알고리즘**을 사용합니다.
* **상대적 뒤꿈치 거리 ($D _{\text{heel}}(t)$)**: 뒤꿈치 마커의 전방 투영 위치에서 골반 중심의 전방 위치를 뺀 값입니다.
  * 왼발: $D _{\text{heel, L}}(t) = (\vec{p} _{\text{heel, L}}(t) \cdot \vec{v} _{\text{forward}}) - P _{\text{pelvis}}(t)$
  * 오른발: $D _{\text{heel, R}}(t) = (\vec{p} _{\text{heel, R}}(t) \cdot \vec{v} _{\text{forward}}) - P _{\text{pelvis}}(t)$
* **생체역학적 배경**: 발을 앞으로 디뎌 땅에 닿는 순간(HS)은 골반(몸통)에 비해 발이 가장 멀리 앞으로 뻗어 나가 있으므로 $D _{\text{heel}}(t)$ 그래프가 **극댓값(Local Maximum, 피크)**을 이룹니다. 반대로 디뎠던 발을 떼는 순간(Toe Off)은 발이 골반보다 가장 뒤에 위치하므로 **극솟값(Local Minimum)**을 이룹니다.
* **노이즈 스무딩**: 캡처 장비의 미세한 흔들림 노이즈를 억제하기 위해 $D _{\text{heel}}(t)$ 시계열 데이터에 **0.08초 크기의 이동 평균 필터**를 적용하여 매끄럽게 보정합니다.
* **검출 제약 조건**: 가짜 피크를 방지하기 위해, 데이터 기반 최적화를 통해 도출된 세 가지 임계값 조건을 통과한 피크만 진짜 발자국으로 인정합니다.
  1. **최소 거리 제약 (`min _distance`)**: 인접한 걸음 간 최소 시간 간격인 **0.44초** 이하로 발생하는 피크는 무시합니다. (정상 대조군 전용 검출 시에는 그룹별 최적값인 **0.38초**를 적용)
  2. **피크 뚜렷함 제약 (`prominence _ratio`)**: 피크의 솟아오른 높이가 피험자 개별 보행 수평 가동 범위의 **16%** 이상이어야 합니다.
  3. **절대 최소 높이 제약 (`min _prominence _m`)**: 궤적 노이즈를 배제하기 위해 피크의 절대 최소 높이가 **1.5cm** ($0.015\text{ m}$) 이상이어야 합니다.

#### ② 파라미터 최적화의 무감독 설계 (Unsupervised Optimization)
본 연구는 학계에서 흔히 쓰는 임의의 임계값 대신, **"안정적인 보행은 일정한 리듬을 갖는다"**는 자연 법칙을 이용하여 임계값들을 스스로 찾아내게 만들었습니다.

* **좌우 분리 주기성 검증**: 파킨슨병 환자는 절뚝거리는 등 좌우 비대칭 보행(Asymmetry)을 보입니다. 좌우를 합친 '걸음(Step)' 시간으로 일정성을 평가하면 이 비대칭성 때문에 오차가 생기므로, 본 연구에서는 **왼발과 오른발을 완전히 분리하여 각각 연속된 보폭 시간(Stride Time: 왼발-왼발, 오른발-오른발)의 변동계수(CV, 변동성)를 최소화**하도록 설계했습니다.
* **생리학적 벌점 (Physiological Regularization)**: 프로그램이 에러를 피하기 위해 걸음 수를 0개로 찾거나 수십 개의 노이즈를 다 걸음으로 인정해버리는 상황을 막기 위해, 6m 내에 검출된 보폭 수가 상식적인 범위를 벗어나거나 보폭 시간이 인간의 한계를 넘으면 벌점을 주는 수식을 추가했습니다. 이를 통해 우리 연구실 데이터에 완전히 최적화된 전역 임계값($min\ _distance = 0.44\text{s}$, $ratio = 0.16$, $min\ _prom = 0.015\text{m}$)을 도출했습니다.

---

### 2-4. 보행 개시 지연 시간 (Gait Initiation Hesitancy)의 정의

보행 개시 지연은 환자가 뇌에서 걷겠다는 명령을 내린 뒤, 실제로 첫 발걸음이 바닥에 닿기까지 걸리는 시간적 지연(Anticpatory Postural Adjustments, APA 지연)을 평가합니다.

1. **움직임 개시 시점 ($t _{\text{onset}}$)**: 정지 상태에 있던 피험자가 걷기 위해 몸을 앞으로 기울여 골반 위치 $P _{\text{pelvis}}(t)$가 최초 기준점 $P _{\text{start}}$ 대비 **$2\text{ cm}$ ($0.02\text{ m}$) 이상 전방으로 최초 돌파하는 프레임의 시간**으로 정의합니다.
2. **첫 발자국 안착 시점 ($t _{\text{HS} _1}$)**: 좌/우 구분 없이 최초로 검출된 1번째 Heel Strike의 발생 시간입니다.
3. **지연 시간 산출 공식**:

   $$\text{Initiation Delay} = t _{\text{HS} _1} - t _{\text{onset}} \quad (\text{단위: 초})$$

---

## 3. Hypothesis Testing & Statistical Comparison

본 연구는 확립된 최적화 임계값과 공간 필터를 일괄 적용하여 233개의 보행 시도로부터 총 6개의 Spatiotemporal 특징 변수를 추출하였습니다. 이후 정상군(HC, $n=40$)과 파킨슨군(PD, $n=193$) 간에 유의미한 생체역학적 차이가 존재하는지 통계 검정을 실시하였습니다.

* **가설 검정 방법**: 모수 검정인 독립표본 **Welch's t-test**(두 집단의 샘플 수 차이 및 이분산성 반영)와 비모수 검정인 **Mann-Whitney U test**를 병행하여 두 집단의 통계적 유의차를 다각도로 평가하였습니다. 유의 수준은 $p < 0.05$로 설정하였습니다.

### Table 1. Statistical Comparison of Gait Metrics between HC and PD
| 보행 변수 (Gait Metric) | 정상 대조군 (HC)<br>($n=40$ trials) | 파킨슨 환자군 (PD)<br>($n=193$ trials) | Welch's t-test<br>($p$-value) | Mann-Whitney U<br>($p$-value) | 귀무가설 ($H _0$) 기각 여부<br>(Rejection at $\alpha=0.05$) |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Stride Time (보폭 주기 시간, s)** | $1.053 \pm 0.120$ | $1.043 \pm 0.090$ | $0.6289$ | $0.8629$ | **기각 실패 (집단 간 차이 없음)** |
| **Stride Time CV (보폭 주기 변동성)** | $0.085 \pm 0.130$ | $0.068 \pm 0.049$ | $0.4374$ | $0.5768$ | **기각 실패 (집단 간 차이 없음)** |
| **Step Length (걸음 길이, m)** | $0.663 \pm 0.078$ | $0.575 \pm 0.098$ | **$5.738 \times 10^{-8}$** | **$1.088 \times 10^{-6}$** | **기각 (통계적으로 매우 유의함)** |
| **Gait Speed (보행 속도, m/s)** | $1.221 \pm 0.269$ | $1.085 \pm 0.206$ | **$0.0043$** | **$0.0001$** | **기각 (통계적으로 유의함)** |
| **Cadence (분당 걸음 수, steps/min)** | $114.561 \pm 9.905$ | $115.038 \pm 9.717$ | $0.7845$ | $0.9158$ | **기각 실패 (집단 간 차이 없음)** |
| **Initiation Delay (보행 개시 지연, s)**| $0.707 \pm 0.786$ | $0.865 \pm 0.941$ | $0.2721$ | $0.1797$ | **기각 실패 (집단 간 차이 없음)** |

---

## 4. Biomechanical & Clinical Discussion (Discussion)

본 분석 결과는 파킨슨병 보행 특징의 **"공간적 결함과 시간적 보존성"**을 매우 선명하게 보여줍니다.

### 4-1. 공간적(Spatial) 결함의 생체역학적 기전
* **Step Length ($p < 10^{-5}$)** 및 **Gait Speed ($p < 0.005$)** 변수에서 극도로 강력한 통계적 유의차가 관찰되었습니다.
* 기저핵(Basal Ganglia)의 도파민성 신경 세포 소실은 파킨슨병의 핵심 증상인 **근강직(Rigidity)**과 **운동완서(Bradykinesia)**를 야기합니다. 이는 보행을 위해 다리를 들어 올리고 앞으로 뻗는 고관절 및 무릎의 관절 가동범위(ROM)를 기계적으로 위축시킵니다.
* 결과적으로 파킨슨 환자는 정상인에 비해 걸음 길이가 평균 **약 8.8cm 짧은 특징적인 종종걸음(Shuffling Gait)**을 보이며, 공간적 추진력 부족으로 인해 최종 보행 속도가 느려지게 됩니다. 

### 4-2. 시간적(Temporal) 제어 및 개시 시간의 통계적 대등성
* 반면, **Stride Time ($p=0.6289$)**, **Cadence ($p=0.7845$)**, **Initiation Delay ($p=0.2721$)**에서는 두 집단 간 **통계적 차이가 전혀 발견되지 않았습니다 ($p \ge 0.05$).**
* 이는 파킨슨 환자들이 걷는 발걸음의 박자(Cadence)와 보행 주기 템포를 조절하는 척수-기저핵 수준의 시간 동기화(CPG, Central Pattern Generator) 기능은 비교적 정상 범주로 유지하려고 보상한다는 임상적 증거입니다.
* 또한, **보행 시작 지연(Initiation Delay)**의 경우 파킨슨 환자군($0.865$초)이 정상 대조군($0.707$초)에 비해 절대적인 지연 시간 평균값은 다소 길게 측정되었으나, 집단 내 편차(표준편차 SD가 각각 0.78초, 0.94초로 매우 큼)가 커서 통계적인 유의수준에는 미치지 못했습니다 ($p \ge 0.05$). 이는 본 실험에 참여한 환자들의 L-dopa 약물 복용 시점이나 개별 기능 상태의 스펙트럼이 매우 다양하기 때문으로 파악되며, 단순 6m 보행의 정적 출발만으로는 개시 지연 변수를 신뢰성 있는 단독 스크리닝 바이오마커로 정의하기에는 임상적으로 한계가 있음을 시사합니다.
* **Stride Time CV (보행 주기 변동성, $p = 0.4374$)** 역시 유의차가 없었습니다. 일반적으로 파킨슨 환자의 보행이 더 불안정(CV가 큼)할 것으로 예상되나 차이가 없는 것은, 6m 직선 보행 프로토콜 특성상 가속과 감속 단계가 정상인에게도 동일한 수준의 주기 변동성을 일시적으로 유발하여 정상상태(Steady-state)의 순수 보행 리듬 변동성 차이를 가려버린(Masking Effect) 결과로 보입니다.

---

## 5. Conclusion (결론)
본 연구를 통해 6m 공간 범위 제약 및 Walk-Through 주기 정합을 거친 표준화된 마커리스 보행 분석 파이프라인의 진단 타당성을 교차 입증하였습니다. 임상에서 파킨슨 환자를 스크리닝할 때, **시간적 속성(보행 템포, 주기 시간, 시작 지연 시간)은 집단 간 차이가 불분명하므로 단독 바이오마커로 채택하기에 부적합**하며, 공간적 제약인 **보폭(Step Length) 단축과 이로 인한 속도(Gait Speed) 저하가 환자의 보행 병리를 변별해내는 정량적이고 핵심적인 진단 지표**로 사용되어야 합니다.
