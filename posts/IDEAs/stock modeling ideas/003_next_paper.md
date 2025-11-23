---
title: "next my paper idea"
date: 2025-11-23       
description: "졸업 논문 주제 - 이후 발전 가능성"
categories: [MetaLeanring, Paper, Idea, MoE]
author: "김한울"
---

다음 논문 아이디어:

# **MoE 기반 금융 시계열 전문가 모델 구성**

MoE(Mixture of Experts)는 **각 전문가가 서로 다른 금융 패턴/시장 구조를 담당**하도록 설계할 수 있어.
따라서 "다른 inductive bias를 가진 모델들"을 조합하는 것이 핵심이야.

---

#  1. **Statistical/Linear Expert (시간-불변 구조 담당)**

이류 모델은 금융에서 baseline으로 강력하고, MoE에 반드시 들어가야 하는 전문가.

###  (1) Linear AR/VAR Expert

* AR(p), VAR(p)
* 이유: 금융 시계열의 **단기 자기상관 / 선형 구조** 담당
* 매우 가벼워서 MoE 전문가로 사용하기 좋고, Regime 0,1같은 안정 국면에 강함.

###  (2) State Space Model / Kalman Expert

* ML의 SSM 버전(서브 스페이스 ID 포함)
* 이유: **저잡음 구간 (low-volatility), 금리/스프레드 같은 macro feature)**에 강함.

→ 통계 기반 expert들은 **smooth한 regime**에 최적화됨.

---

#  2. **Lightweight Convolution Experts (Local/Medium Patterns)**

U-Net 전체를 쓰기에는 heavy하니까, 경량화된 conv는 최고의 선택.

###  (3) **TCN (Temporal Convolution Network)**

* Dilated conv 기반
* LSTM보다 빠르고, 장기 의존성 처리 가능
* 금융의 **볼륨 급증, 변동성 스파이크, rolling 패턴** 포착에 적합

###  (4) **1D MobileNet / EfficientNet-Lite**

* Conv 기반 경량 CNN
* 실시간 inference 가능
* 장점: 주가 series 형태의 local shape 기반 패턴을 아주 잘 잡음
* 퀀트 실무에서 convolution-based price_shape encoding 자주 사용됨

###  (5) **Temporal Depthwise Separable Conv Expert**

* 1D Conv + Depthwise + Pointwise
* 패턴: **short-term burst, microstructure-like data**, ETF/선물에 강함

---

#  3. **Recurrent Experts (Long-range dynamics)**

MoE에서는 중량 좋은 Recurrent 모델 딱 하나 넣어도 강력함.

###  (6) GRU Expert

* LSTM보다 너무 가볍고
* 변동성 clustering & market drift 모델링에 좋음
* 단기적 비정상성 처리에 강함

###  (7) MinimalRNN / SRU

* High-speed RNN
* 변동성 예측에 특히 잘 맞음
* 장점: compute-efficient → 전문가 모델 8개 정도 넣어도 가벼움

---

#  4. **Attention-based Experts (각기 다른 시계열 dependency)**

경량 attention 계열은 U-Net의 multi-scale role 일부를 대체해.

###  (8) Performer / Linformer Expert

* O(N) self-attention
* long-range dependency 담당
* 대형 Transformer가 필요 없는 금융 시계열에 딱 맞음

###  (9) Nystromformer (경량 Transformer)

* nonstationary + long-range 구조 담당
* Macro-driven regime에서 성능이 좋음
* 시장 레짐 전환 직전의 “weak signals”을 잘 포착

---

#  5. **Hybrid & Feature-specific Experts**

금융에서는 multi-modal feature가 많기 때문에, 특정 feature domain을 담당하는 전문가가 효과적.

###  (10) Volatility-only Expert

* Input = realized vol, implied vol indices (VIX), volume, ATR
* Model = small MLP or TCN
* 역할: **Regime shift detection 강화**

###  (11) Macro Expert

* Yield spread / Fed rate / CPI 등만 입력
* Expert model: tiny GRU or Kalman
* Global macro regime에서 Sharpe↑

###  (12) Cross-sectional embedding expert

* 수백 종목을 embedding하는
  lightweight transformer or PCA-MLP
* 역할: Factor structure (value, momentum, quality)

---

#  6. **신형 경량 시계열 모델들 (2023~2024 최신 연구)**

###  (13) TimesNet Expert

* Microseasonality + season-decompose 기반
* 금융에서도 보이는 multi-frequency pattern들을 잘 잡음

###  (14) Mamba/MambaTime (SSM 기반)

* 2024년 neural SSM
* 장기 horizon 예측 + 계산량 매우 적음
* "U-Net보다 10배 가벼우면서 더 정확"이라는 결과 다수

###  (15) Hyena Operator Expert

* FFT-like long convolution
* 매우 긴 시계열에 안정적
* macro shocks, stress regimes 처리에 유리

---

# MoE 전문화 방식: 추천 구조

아래처럼 전문가를 묶어주는 게 금융에서 가장 잘 작동한다.

### **(1) Short-horizon experts**

* Depthwise CNN
* MobileNet-1D
* AR Expert

### **(2) Medium-horizon experts**

* TCN
* GRU
* TimesNet

### **(3) Long-horizon experts**

* Performer
* Nystromformer
* Mamba

### **(4) Feature-domain experts**

* Macro expert
* Volatility expert
* Cross-sectional expert

Total 8~12명이면 딱 좋아.

---

#  MoE 추가 아이디어 (다음 논문 컨트리뷰션 강화 포인트)

###  1. Gating Network = Regime classifier

현재 MAML이 regime adaptation 기반이므로
MoE gating이 regime 구조를 반영하도록 설계하면 자연스러움.

###  2. Expert Dropout

각 expert는 특정 조건에서만 활성화되므로
regularization 효과가 생기고 overfitting 줄어듦.

###  3. Energy-based uncertainty gating

불확실성이 크면
→ attention expert 활성화,
안정적이면
→ AR/TCN 같은 lightweight expert 활성화.

---

# 수식으로

## 0. 공통 표기

* 시계열 입력:

  * 단변량: $x_t \in \mathbb{R}$
  * 다변량: $\mathbf{x}_t \in \mathbb{R}^d$
* 예측 대상(다음 수익률 등): $y_t \in \mathbb{R}$ 혹은 $\mathbf{y}_t \in \mathbb{R}^{d_y}$
* 과거 윈도우: $\mathbf{X}*{t-L+1:t} = (\mathbf{x}*{t-L+1}, \dots, \mathbf{x}_t)$
* 파라미터 집합: $\theta$

---

## 1. 선형 계열 전문가 (AR / VAR / 상태공간)

### 1.1 AR(p) Expert

단변량 자기회귀 모형:

$$
y_t
= \phi_0 + \sum_{i=1}^p \phi_i, y_{t-i} + \varepsilon_t,
\qquad
\varepsilon_t \sim \mathcal{N}(0, \sigma^2).
$$

* **예측 함수 형태**:
  $$
  \hat{y}*t = f*{\text{AR}}(y_{t-1},\dots,y_{t-p}; \theta)
  = \phi_0 + \sum_{i=1}^p \phi_i, y_{t-i}.
  $$

---

### 1.2 VAR(p) Expert

$d$차원 다변량 벡터 $\mathbf{y}_t \in \mathbb{R}^d$에 대해:

$$
\mathbf{y}*t
= \mathbf{c} + \sum*{i=1}^p A_i \mathbf{y}_{t-i} + \boldsymbol{\varepsilon}_t,
\qquad
\boldsymbol{\varepsilon}_t \sim \mathcal{N}(\mathbf{0}, \Sigma).
$$

* 예측:
  $$
  \hat{\mathbf{y}}*t
  = f*{\text{VAR}}\big(\mathbf{y}*{t-1},\dots,\mathbf{y}*{t-p}; \theta\big)
  = \mathbf{c} + \sum_{i=1}^p A_i \mathbf{y}_{t-i}.
  $$

---

### 1.3 상태공간 / Kalman Expert

**숨은 상태**: $\mathbf{z}_t \in \mathbb{R}^k$
**관측**: $\mathbf{y}_t \in \mathbb{R}^d$

상태방정식:

$$
\mathbf{z}*t = F \mathbf{z}*{t-1} + \mathbf{w}_t,
\qquad
\mathbf{w}_t \sim \mathcal{N}(\mathbf{0}, Q)
$$

관측방정식:

$$
\mathbf{y}_t = H \mathbf{z}_t + \mathbf{v}_t,
\qquad
\mathbf{v}_t \sim \mathcal{N}(\mathbf{0}, R)
$$

**Kalman 필터 업데이트** (prediction + correction):

* 예측 단계:
  $$
  \hat{\mathbf{z}}*{t|t-1} = F \hat{\mathbf{z}}*{t-1|t-1}, \qquad
  P_{t|t-1} = F P_{t-1|t-1} F^\top + Q
  $$
* 보정 단계:
  $$
  K_t = P_{t|t-1} H^\top (H P_{t|t-1} H^\top + R)^{-1}
  $$
  $$
  \hat{\mathbf{z}}*{t|t}
  = \hat{\mathbf{z}}*{t|t-1} + K_t (\mathbf{y}*t - H \hat{\mathbf{z}}*{t|t-1})
  $$

예측값:

$$
\hat{\mathbf{y}}*{t+1}
= H \hat{\mathbf{z}}*{t+1|t}.
$$

---

## 2. Conv 기반 전문가 (로컬/중기 패턴)

### 2.1 1D CNN Expert

길이 $L$ 시계열 $\mathbf{X}_{t-L+1:t} \in \mathbb{R}^{L \times d}$,
커널 폭 $k$, 채널 $C$일 때, 각 필터 $c$에 대해

$$
h_{t}^{(c)} = \sigma\left(
\sum_{i=0}^{k-1} W^{(c)}*{i} \mathbf{x}*{t-i} + b^{(c)}
\right),
$$

* 여기서 $\sigma$는 비선형 함수(ReLU 등).

최종 예측:

$$
\hat{y}*t = W*{\text{out}} \mathbf{h}*t + b*{\text{out}},
\qquad
\mathbf{h}_t = [h_t^{(1)}, \dots, h_t^{(C)}]^\top.
$$

---

### 2.2 Depthwise Separable Conv Expert

입력 채널 수 $d$, 커널 크기 $k$.

1. **Depthwise Convolution** (채널별):

$$
\tilde{x}*t^{(j)} =
\sum*{i=0}^{k-1} w^{(j)}*i, x*{t-i}^{(j)} \quad (j=1,\dots,d)
$$

2. **Pointwise Convolution** (채널 혼합, $1\times 1$ conv):

$$
h_t^{(c)} = \sigma\left(
\sum_{j=1}^d v^{(c)}_j, \tilde{x}_t^{(j)} + b^{(c)}
\right)
$$

예측은 CNN과 동일하게:

$$
\hat{y}*t = W*{\text{out}} \mathbf{h}*t + b*{\text{out}}.
$$

---

### 2.3 TCN (Temporal Convolutional Network) Expert

**dilation** $d_\ell$을 가진 1D conv:

$$
h_t^{(\ell)} = \sigma\left(
\sum_{i=0}^{k-1} W^{(\ell)}*{i} h*{t - d_\ell \cdot i}^{(\ell-1)} + b^{(\ell)}
\right),
$$

* 레이어 깊이 $\ell = 1,\dots,L$
* 보통 $d_\ell = 2^{\ell-1}$ (지수적 dilation)

**Residual block**:

$$
\tilde{h}_t^{(\ell)}
= h_t^{(\ell)} + h_t^{(\ell-1)}
$$

최종:

$$
\hat{y}*t = W*{\text{out}} \tilde{h}*t^{(L)} + b*{\text{out}}.
$$

---

## 3. RNN 계열 전문가 (장기 의존성)

### 3.1 GRU Expert

입력 $\mathbf{x}_t \in \mathbb{R}^d$, hidden state $\mathbf{h}_t \in \mathbb{R}^m$:

**업데이트 게이트**:

$$
\mathbf{z}_t = \sigma\left( W_z \mathbf{x}*t + U_z \mathbf{h}*{t-1} + \mathbf{b}_z \right)
$$

**리셋 게이트**:

$$
\mathbf{r}_t = \sigma\left( W_r \mathbf{x}*t + U_r \mathbf{h}*{t-1} + \mathbf{b}_r \right)
$$

**candidate hidden**:

$$
\tilde{\mathbf{h}}_t
= \tanh\left( W_h \mathbf{x}_t + U_h (\mathbf{r}*t \odot \mathbf{h}*{t-1}) + \mathbf{b}_h \right)
$$

**최종 hidden**:

$$
\mathbf{h}_t
= (1 - \mathbf{z}*t) \odot \mathbf{h}*{t-1}

* \mathbf{z}_t \odot \tilde{\mathbf{h}}_t
  $$

예측:

$$
\hat{y}*t = W*{\text{out}} \mathbf{h}*t + b*{\text{out}}.
$$

---

### 3.2 Minimal RNN / SRU 스타일 Expert (간단 버전)

입력 투영:

$$
\mathbf{u}_t = W_x \mathbf{x}_t
$$

게이트:

$$
\mathbf{f}_t = \sigma(W_f \mathbf{x}_t + \mathbf{b}_f), \quad
\mathbf{r}_t = \sigma(W_r \mathbf{x}_t + \mathbf{b}_r)
$$

상태 업데이트:

$$
\mathbf{c}_t = \mathbf{f}*t \odot \mathbf{c}*{t-1} + (1 - \mathbf{f}_t) \odot \mathbf{u}_t
$$

출력:

$$
\mathbf{h}_t = \mathbf{r}_t \odot \phi(\mathbf{c}_t)
+ (1 - \mathbf{r}_t) \odot \mathbf{x}_t
$$

---

## 4. Attention 계열 전문가

### 4.1 기본 Scaled Dot-Product Self-Attention

입력 시퀀스 $X \in \mathbb{R}^{T \times d}$:

질의/키/값:

$$
Q = X W_Q, \quad K = X W_K, \quad V = X W_V,
\qquad
W_Q,W_K,W_V \in \mathbb{R}^{d \times d_k}
$$

Self-attention:

$$
\text{Attention}(Q,K,V)
= \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right) V
$$

multi-head:

$$
\text{MHA}(X)
= \text{Concat}(H_1,\dots,H_H) W_O,
$$

$$
H_h = \text{Attention}(X W_Q^{(h)}, X W_K^{(h)}, X W_V^{(h)}).
$$

---

### 4.2 Performer 스타일 (Kernelized Attention) Expert

키/쿼리에 대한 feature map $\phi: \mathbb{R}^{d_k} \to \mathbb{R}^{m}$ 사용:

$$
\text{Attention}_{\text{Performer}}(Q,K,V)
\approx
D^{-1} \left( \phi(Q) \left( \phi(K)^\top V \right) \right)
$$

여기서

$$
D = \text{diag}\big( \phi(Q) \phi(K)^\top \mathbf{1} \big)
$$

* $\mathbf{1}$: all-ones 벡터
* **복잡도**: $O(T m)$ (일반 $O(T^2)$ 대비 경량)

---

### 4.3 Linformer 스타일 Expert

키/값에 **저차원 투영** $E_K, E_V \in \mathbb{R}^{T \times r}$:

$$
K' = E_K^\top K, \quad V' = E_V^\top V
$$

$$
\text{Attention}_{\text{Linformer}}(Q,K,V)
= \text{softmax}\left( \frac{Q (K')^\top}{\sqrt{d_k}} \right) V'
$$

* 시간축을 $T \to r$로 줄여 $O(Tr)$

---

### 4.4 Nyströmformer 스타일 Expert (Nyström Approximation)

대표 타임스텝(landmarks) 인덱스 집합 $\mathcal{I}$ 선택:

$$
K_{\text{full}} = \text{softmax}\left( \frac{QK^\top}{\sqrt{d_k}} \right)
$$

Nyström 근사:

$$
K_{\text{full}} \approx C W^{\dagger} C^\top
$$

* $C$: 행/열 일부를 뽑은 행렬 (landmark 관련)
* $W$: landmark-block
* $W^{\dagger}$: pseudo-inverse

Attention:

$$
\text{Attention}_{\text{Nyström}}(Q,K,V)
\approx C W^{\dagger} (C^\top V).
$$

---

## 5. Feature-domain Experts (Vol / Macro / Cross-sectional)

### 5.1 Volatility Expert (MLP)

입력 피처 벡터:

$$
\mathbf{v}*t =
\big(
\text{RV}*{20,t}, \text{RV}_{60,t}, \text{VIX}_t, \text{ATR}_t, \dots
\big) \in \mathbb{R}^{d_v}
$$

2-layer MLP:

$$
\mathbf{h}_t = \phi(W_1 \mathbf{v}_t + \mathbf{b}_1)
$$

$$
\hat{y}_t = W_2 \mathbf{h}_t + \mathbf{b}_2
$$

---

### 5.2 Macro Expert (작은 GRU)

입력: $\mathbf{m}_t \in \mathbb{R}^{d_m}$
(예: 금리, 스프레드, CPI YoY 등)

GRU 업데이트는 앞의 GRU 수식과 동일, 단 입력을 $\mathbf{m}_t$로 사용:

$$
\mathbf{h}^{\text{macro}}_t = \text{GRU}(\mathbf{m}*t, \mathbf{h}^{\text{macro}}*{t-1})
$$

예측:

$$
\hat{y}^{\text{macro}}*t = W*{\text{macro}} \mathbf{h}^{\text{macro}}*t + \mathbf{b}*{\text{macro}}.
$$

---

### 5.3 Cross-sectional Expert (Embedding + MLP/Attention)

종목 임베딩 $\mathbf{e}*i \in \mathbb{R}^{d_e}$,
시점 $t$에서 종목 $i$의 특징 $\mathbf{x}*{i,t}$:

임베딩 결합:

$$
\tilde{\mathbf{x}}*{i,t} = [\mathbf{x}*{i,t}; \mathbf{e}_i] \in \mathbb{R}^{d + d_e}
$$

간단한 cross-sectional attention:

$$
q_{i,t} = W_q \tilde{\mathbf{x}}*{i,t}, \quad
k*{j,t} = W_k \tilde{\mathbf{x}}*{j,t}, \quad
v*{j,t} = W_v \tilde{\mathbf{x}}_{j,t}
$$

$$
\alpha_{i,j,t} =
\frac{
\exp\left( \frac{q_{i,t}^\top k_{j,t}}{\sqrt{d_k}} \right)
}{
\sum_{j'} \exp\left( \frac{q_{i,t}^\top k_{j',t}}{\sqrt{d_k}} \right)
}
$$

집계:

$$
\mathbf{h}*{i,t} = \sum_j \alpha*{i,j,t} v_{j,t}
$$

예측:

$$
\hat{y}*{i,t} = w^\top \mathbf{h}*{i,t} + b.
$$

---

## 6. 최신 시계열 아키텍처 전문가

### 6.1 TimesNet 스타일 Expert (다중 주기/주파수)

간단화 버전으로, 시계열을 주파수 도메인에서 convolution:

입력 시퀀스 $\mathbf{x}_{1:T}$에 대해 DFT:

$$
\mathbf{X}*\omega = \mathcal{F}{\mathbf{x}*{1:T}}(\omega)
$$

주파수별 필터 $G(\omega)$를 학습:

$$
\mathbf{Y}*\omega = G(\omega) \odot \mathbf{X}*\omega
$$

역변환:

$$
\tilde{\mathbf{x}}*{1:T} = \mathcal{F}^{-1}{\mathbf{Y}*\omega}
$$

마지막에 MLP 또는 Conv로 예측:

$$
\hat{y}*t = f*{\text{MLP}}(\tilde{\mathbf{x}}_{t-L+1:t}).
$$

---

### 6.2 SSM / Mamba 스타일 Expert (Neural SSM)

연속시간 선형 상태공간:

$$
\frac{d\mathbf{z}(t)}{dt} = A \mathbf{z}(t) + B \mathbf{x}(t),
\qquad
\mathbf{y}(t) = C \mathbf{z}(t) + D \mathbf{x}(t)
$$

시간 스텝 $\Delta$로 이산화:

$$
\mathbf{z}*t = \bar{A} \mathbf{z}*{t-1} + \bar{B} \mathbf{x}_t,
\qquad
\mathbf{y}_t = C \mathbf{z}_t + D \mathbf{x}_t
$$

여기서

$$
\bar{A} = e^{A \Delta}, \qquad
\bar{B} = \left(\int_0^\Delta e^{A \tau} d\tau\right) B.
$$

Neural SSM에서는 $A,B,C,D$를 신경망 파라미터로 학습.

---

### 6.3 Hyena 스타일 Expert (긴 컨볼루션)

긴 convolution을 직접 계산하지 않고, **implicit filter** $\mathbf{w}_{\theta}$를 학습:

입력 $\mathbf{x}_{1:T}$에 대해

$$
y_t = \sum_{i=0}^{T-1} w_{\theta, i}, x_{t-i}
$$

$\mathbf{w}_{\theta}$는 파라미터화된 함수 (예: MLP + positional embedding):

$$
w_{\theta, i} = g_\theta(i), \quad i = 0,\dots,T-1
$$

실제 구현은 FFT/분해를 활용해 $O(T \log T)$로 계산.

---

## 7. MoE 관점에서의 최종 형태

각 expert $E_m$이 입력 시퀀스에서 출력 $\hat{y}^{(m)}_t$를 내놓는다고 하면,

* Experts:
  $$
  \hat{y}^{(m)}*t = f_m(\mathbf{X}*{t-L+1:t}; \theta_m),
  \qquad m = 1,\dots,M
  $$

* Gating 네트워크 (예: regime-aware):

  $$
  \mathbf{g}*t = \text{softmax}\big( u(\mathbf{X}*{t-L+1:t}; \phi) \big) \in \mathbb{R}^M
  $$

* 최종 MoE 출력:

  $$
  \hat{y}*t
  = \sum*{m=1}^M g_{t}^{(m)}, \hat{y}^{(m)}_t
  $$
