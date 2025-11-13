---
title: "석사 학위 논문 연구 계획서 - Bayesian+MetaLearning"
date: 2025-11-13       
description: "졸업 논문 주제 구체화 - Bayesian+MetaLearning"
categories: [Bayesian, MetaLearning, Idea]
author: "김한울"
---


# 석사 학위 논문 연구 계획서

## 1. 연구 주제

**국문 제목**  
Task 간 유사도를 반영한 계층 베이지안 메타러닝 prior의 일반적 구성과 통계적 성질

**영문 제목**  
A General Prior Design Incorporating Task Similarity in Hierarchical Bayesian Meta-Learning and Its Statistical Properties

---

## 2. 연구 배경 및 필요성

딥러닝 기반 모델은 대규모 데이터와 연산 자원을 요구하며, 새로운 task가 등장할 때마다 학습을 처음부터 반복해야 한다는 한계를 가진다. 이를 극복하기 위해 등장한 **meta-learning(learning to learn)** 은 여러 task로부터 축적된 경험을 이용하여, 새로운 task에 대한 빠른 적응과 데이터 효율적 학습을 목표로 한다. 

최근 meta-learning 연구는 few-shot 이미지 분류, 강화학습, 베이지안 신경망 등 다양한 응용에서 활발히 진행되고 있으며, 특히 여러 task 간의 공통 구조를 활용하는 **계층 베이지안(hierarchical Bayes)** 및 **Gaussian process(GP) 기반 meta-learning** 이 주목받고 있다. 

그러나 기존 Bayesian/meta-learning 연구들은 다음과 같은 한계를 가진다.

1. **Task 유사도 구조의 모형화 부족**  
   - 많은 meta-learning 알고리즘은 암묵적으로 “task들이 유사하다”는 가정을 갖고 있으나,  
     유사도를 **명시적인 prior 공분산 구조**로 표현하고 그 통계적 효과를 분석한 연구는 제한적이다.
2. **선형–가우시안 계층 모형에서의 이론적 분석 부족**  
   - GP 기반 meta-learning은 task 간 커널을 제안하고 실험적으로 성능 향상을 보이지만,   
     단순한 선형 회귀/가우시안 노이즈 환경에서  
     **task similarity를 반영한 prior와 독립 prior의 Bayes risk를 비교·정량화하는 통계적 연구**는 상대적으로 부족하다. 
3. **Meta-learning 이론(예: PAC-Bayes bound)과 구체적 prior 구조의 연결 부족**  
   - PAC-Bayesian meta-learning은 hyper-posterior의 최적 구조(PACOH)를 제시하지만,   
     구체적인 task similarity 기반 prior가 이러한 이론적 틀 안에서 어떤 효과를 가지는지에 대한 정량적 논의는 제한적이다.

본 연구는 이러한 한계를 해결하고자, **task 간 유사도를 반영한 계층 베이지안 meta-learning prior의 일반적 구조를 제안**하고,  
**선형–가우시안 계층 모형에서의 Bayes risk 및 학습 곡선(learning curve) 관점에서 그 통계적 성질을 분석하는 것**을 목표로 한다.

---

## 3. 선행 연구 및 이론적 배경

### 3.1 Meta-learning 개요 및 taxonomy

Meta-learning은 여러 task로부터 “학습 알고리즘 자체” 또는 “초기 파라미터/표현”을 학습하여, 새로운 task에 빠르게 적응하는 것을 목표로 한다. Hospedales et al.은 meta-learning을 정리하면서, meta-train / meta-test 분할, N-way K-shot 설정, task 분포 $\mathcal{T}$ 등의 표준 수학적 세팅을 제시하고, 다양한 방법론을 포괄하는 taxonomy를 제안하였다. 

일반적으로 meta-learning은 다음과 같이 정식화된다.

- Task 분포 $\mathcal{T}$ 에서 task $t$를 샘플:
  $$
  t \sim \mathcal{T}, \quad D_t = \{(x_{ti}, y_{ti})\}_{i=1}^{n_t}
  $$
- Meta-train 단계에서 여러 $t=1,\dots,T$ 에 대해 데이터를 관측하고,
- 새로운 task $t^\*$ 에 대한 적은 양의 데이터로 빠르게 적응하는 meta-learner를 학습한다.

Hospedales et al.의 taxonomy에 따르면, meta-learning 방법은 크게  
(1) optimization-based, (2) metric-based, (3) model-based, (4) Bayesian/probabilistic 기반 방법으로 나눌 수 있다.   
본 연구는 이 중 **Bayesian/probabilistic meta-learning** 축에 속한다.

---

### 3.2 Gradient-based meta-learning

Optimization-based meta-learning의 대표적 예로 **MAML(Model-Agnostic Meta-Learning)** 계열이 있다. 이들은 모델 파라미터의 초기값 $\phi$ 를 meta-level에서 학습하고, 각 task별로 서버럴 스텝의 gradient descent를 통해 적응한다. 이러한 방법들은 다양한 신경망 구조에 적용이 가능하고, 구현이 상대적으로 간단하다는 장점이 있어 few-shot 학습에서 널리 사용된다. 

Grant et al.는 **“Recasting Gradient-Based Meta-Learning as Hierarchical Bayes”** 에서 MAML과 같은 gradient-based meta-learning이, 적당한 근사 하에서 **계층 베이지안 추론의 한 형태**로 해석될 수 있음을 보였다.   
즉, meta-parameter는 상위 계층의 hyperparameter, inner-loop 업데이트는 task-specific posterior mode 추정에 해당한다.

또한, Zou & Lu는 **Gradient-EM Bayesian Meta-Learning** 을 통해 계층 베이지안 모형에서 empirical Bayes 추정을 수행하는 gradient-EM 기반 meta-learning 알고리즘을 제안하고, 기존 gradient-based meta-learning 알고리즘을 하나의 Bayesian 틀 안에서 통합하여 해석하였다. 

이러한 연구들은 **gradient-based meta-learning과 계층 베이지안 추론 간의 연결**을 보여주지만,  
task 유사도 구조를 공분산으로 명시적으로 모델링하고 그 통계적 성질을 분석하는 데에는 초점을 두지 않는다.

---

### 3.3 Bayesian meta-learning

Bayesian meta-learning은 여러 task의 데이터를 이용하여 **prior 또는 hyperparameter를 empirical Bayes/fully Bayes 방식으로 추정**하고, 새로운 task에 대해 불확실성 추정을 포함한 적응을 수행한다. 

일반적인 계층 베이지안 meta-learning 모형은 다음과 같이 표현할 수 있다.

$$
\begin{aligned}
\eta &\sim p(\eta), \\
\theta_t \mid \eta &\sim p(\theta_t \mid \eta), \quad t = 1,\dots,T, \\
D_t \mid \theta_t &\sim p(D_t \mid \theta_t),
\end{aligned}
$$

여기서 $\eta$ 는 상위 계층의 hyperparameter, $\theta_t$ 는 task-specific 파라미터이다.  
Gradient-EM Bayesian meta-learning과 관련 연구들은 이러한 구조에서  
$\eta$ 를 empirical Bayes 방식으로 추정하는 다양한 알고리즘과 이론적 성질을 제시하였다. 

그러나 Bayesian meta-learning 문헌의 상당수는 **hyperparameter 추정 알고리즘과 실험적 성능**에 집중하며,  
task 간 유사도 구조가 prior 공분산에 어떻게 반영되며, 이로 인해 **Bayes risk와 pooling 정도가 어떻게 변하는지에 대한 체계적 분석**은 상대적으로 부족하다.

---

### 3.4 Gaussian process 기반 meta-learning 및 PAC-Bayesian meta-learning 이론

#### 3.4.1 GP 기반 meta-learning

Gaussian process(GP)는 함수 공간의 베이지안 prior로서, 불확실성을 자연스럽게 표현할 수 있다는 장점이 있다. Nguyen et al.은 **“Learning to Learn with Gaussian Processes”**에서 few-shot 회귀 문제를 위해 **Gaussian Process Meta-Learning(GPML)** 을 제안하였으며, task 간 거리를 이용한 **novel task kernel** 을 도입하여 meta-learning 환경에서 task 간 유사도를 활용하였다.   

이와 유사한 GP 기반 meta-learning 연구들은, multi-task GP, deep kernel GP, variational GP 등의 구조를 활용하여 task 간 공유 정보를 모델링하고 few-shot 상황에서 성능 향상을 보였다.   

또한 Ashton & Sollich은 **“Learning curves for multi-task Gaussian process regression”**에서  
**multi-task GP 회귀의 평균 Bayes error(learning curve)** 를 분석하여, task 간 공분산 구조가 학습 곡선에 미치는 영향을 정량적으로 연구하였다.   
이는 본 연구에서 계획하는 **task similarity 기반 prior의 Bayes risk 분석**과 직접적인 수학적 연관이 있다.

#### 3.4.2 PAC-Bayesian meta-learning

Rothfuss et al.은 **“Scalable PAC-Bayesian Meta-Learning via the PAC-Optimal Hyper-Posterior (PACOH)”**에서  
meta-learning의 generalization error에 대한 PAC-Bayesian upper bound를 유도하고, 이를 최소화하는 **PAC-optimal hyper-posterior (PACOH)** 를 도출하였다.   
PACOH는 GP, Bayesian neural network 등 다양한 base learner에 적용 가능하며, meta-level regularization을 이론적으로 정당화한다.

PAC-Bayesian meta-learning 이론은 meta-level에서의 최적 prior/hyper-posterior 구조에 대한 중요한 통찰을 제공하지만,  
구체적인 **task similarity 기반 공분산 구조**가 이러한 bound에 어떤 영향을 주는지에 대한 분석은 제한적이다.

---

### 3.5 Task similarity 기반 Bayesian/meta-learning

Multi-task learning 및 GP 기반 모델에서는 오래전부터 **task 간 유사도**를 공분산 구조로 표현해 왔다.  
예를 들어, multi-task GP에서는 입력 커널 $K_x$와 task 간 공분산 $\Sigma_{\text{task}}$의 곱으로 전체 공분산을 구성한다.

$$
K((x,s), (x',t)) = K_x(x, x') \cdot \Sigma_{\text{task}}(s,t).
$$

여기서 $\Sigma_{\text{task}}$는 task 간 유사도/상관을 반영하는 행렬이다.   

Nguyen et al.의 GPML은 task 간 거리를 활용한 **task kernel** 을 제안하여, meta-learning 환경에서 task similarity를 명시적으로 모델링한다.   
또한 다양한 multi-task GP, hierarchical GP 연구에서는 task feature, 그래프 구조, 군집 등을 이용한 공분산 설계를 시도하고 있다.   

하지만 이들 연구는 주로 **복잡한 GP 구조 및 대규모 실험에 기반한 모델 제안**에 집중하며,  
단순한 선형–가우시안 계층 모형에서

- (1) task similarity를 반영한 prior 공분산 구조가 어떤 조건 하에서 유효한지,  
- (2) 독립 prior 대비 Bayes risk 및 learning curve가 어떻게 달라지는지

를 **이론적으로 분석하는 통계적 연구는 상대적으로 부족**하다.

따라서 본 연구는, **선형–가우시안 계층 베이지안 meta-learning 모형**을 기반으로  
**task similarity 기반 prior 구조를 일반적으로 정의하고, Bayes risk 및 pooling 구조를 수학적으로 분석**함으로써,  
기존 문헌의 공백을 메우고자 한다.

---

## 4. 연구 목적 및 연구 질문

### 4.1 연구 목적

1. **Task 간 유사도를 반영하는 일반적인 계층 베이지안 meta-learning prior 구조 제안**  
2. **선형–가우시안 계층 모형에서 similarity-aware prior와 독립 prior의 Bayes risk 및 학습 곡선 비교 분석**  
3. **제안 prior 구조의 이론적 성질(유효성, risk 개선 조건 등)을 정리하고, 시뮬레이션 및 실증으로 검증**

### 4.2 구체적 연구 질문

- **RQ1.** Task feature 또는 task 간 거리/그래프 정보를 이용하여,  
  계층 베이지안 meta-learning에서 일반적으로 사용할 수 있는 **task similarity 기반 prior 공분산 구조**를 어떻게 정의할 수 있는가?

- **RQ2.** 선형 회귀 + 가우시안 노이즈 환경에서,  
  similarity-aware prior와 독립 prior에 기반한 meta-learning의 **Bayes risk**는 어떻게 비교되는가?  
  특히 어떤 조건(유사도 구조가 실제 task 관계를 잘 반영할 때 등) 하에서 risk 개선이 발생하는가?

- **RQ3.** Multi-task GP 회귀의 학습 곡선 분석 결과를 활용하여,  
  similarity-aware prior의 **평균 Bayes error(learning curve)** 에 대한 해석적 표현 또는 근사/상하한을 제시할 수 있는가?

- **RQ4.** 제안 prior 구조와 분석 결과는  
  실제 meta-learning 환경(예: few-shot 회귀/분류 데이터셋)에서 성능 향상 및 불확실성 측정 개선으로 이어지는가?

---

## 5. 연구 내용 및 방법

### 5.1 기본 모형 설정

본 연구는 다음과 같은 **선형–가우시안 계층 베이지안 meta-learning 모형**을 기본으로 한다.

- Task $t$의 회귀 모형:
  $$
  y_{ti} = x_{ti}^\top \beta_t + \epsilon_{ti}, \quad
  \epsilon_{ti} \sim \mathcal{N}(0, \sigma^2),
  $$
  여기서 $x_{ti} \in \mathbb{R}^d$, $\beta_t \in \mathbb{R}^d$.
- 각 task의 파라미터 벡터를 쌓아
  $$
  \beta = (\beta_1^\top, \dots, \beta_T^\top)^\top.
  $$

### 5.2 제안 prior 구조: task similarity 기반 공분산

#### (1) 독립 prior (baseline)

기존 계층 모형에서 자주 사용하는 baseline prior는 다음과 같다.

$$
\beta_t \sim \mathcal{N}(0, \tau^2 I_d), \quad t = 1,\dots,T,
$$

또는 전체 벡터에 대해

$$
\beta \sim \mathcal{N}(0, I_T \otimes \tau^2 I_d).
$$

이는 task 간 독립성을 가정하며, task 간 유사도 구조를 반영하지 않는다.

#### (2) Task similarity 기반 prior

본 연구에서는 task feature $\phi(t) \in \mathbb{R}^q$ 또는 task 간 거리/그래프 정보를 이용하여  
다음과 같은 **task covariance 행렬**을 정의한다.

- 커널 기반 구조:
  $$
  \Sigma_{\text{task}}(s,t) = k(\phi(s), \phi(t)),
  $$
  여기서 $k$는 positive definite kernel (예: RBF, Matérn 등)이다.
- 그래프 라플라시안 기반 구조:
  $$
  \Sigma_{\text{task}} = (L + \lambda I)^{-1},
  $$
  여기서 $L$은 task 그래프의 라플라시안, $\lambda>0$는 regularization 파라미터이다.

이를 이용하여 전체 prior 공분산을

$$
\operatorname{cov}(\beta) = \Sigma_{\text{task}} \otimes \tau^2 I_d
$$

로 정의하는 **similarity-aware prior**를 제안한다.

이때 $k$의 positive definiteness, $L$의 성질 등을 이용하여  
$\Sigma_{\text{task}}$ 및 $\Sigma_{\text{task}} \otimes \tau^2 I_d$ 가 양정치 행렬이 됨을 보이고,  
이에 따라 prior가 well-defined multivariate Gaussian이 됨을 정리 형태로 제시한다.

### 5.3 이론적 분석 계획

#### (1) Posterior 및 예측 분포 도출

선형–가우시안 모형에서 similarity-aware prior를 사용하면,  
posterior 및 posterior predictive distribution은 닫힌형으로 표현 가능하다.

- Posterior:
  $$
  p(\beta \mid D_{1:T}) = \mathcal{N}(\mu_{\beta\mid D}, \Sigma_{\beta\mid D}),
  $$
  여기서 $\mu_{\beta\mid D}$, $\Sigma_{\beta\mid D}$는 prior 공분산과 데이터 행렬 $X_{1:T}$, 노이즈 분산 $\sigma^2$에 의해 결정된다.

- 새로운 task $t^\*$ 에 대한 예측 분포:
  $$
  p(y^\* \mid x^\*, D_{1:T}, D_{t^\*}) = \mathcal{N}(m(x^\*), v(x^\*)),
  $$

이를 독립 prior와 similarity-aware prior 두 경우에 대해 명시적으로 도출한다.

#### (2) Bayes risk 비교

새로운 task에서의 예측 MSE를 Bayes risk로 정의한다.

$$
R = \mathbb{E}\left[(y^\* - \hat{y}^\*)^2\right],
$$

여기서 기대는 데이터 및 prior/likelihood에 대한 joint 분포에 대해 취한다.

- 독립 prior: $R_{\text{ind}}$
- similarity-aware prior: $R_{\text{sim}}$

를 각각 계산하거나 상·하한을 도출하고,  
특히 task covariance 행렬 $\Sigma_{\text{task}}$와 참 covariance $\Sigma_{\text{true}}$의 정렬 정도(예: eigen 구조, 코사인 유사도 등)에 따라

$$
R_{\text{sim}} \le R_{\text{ind}}
$$

가 성립하는 조건을 정리 형태(정리/레마)로 제시한다.

이 과정에서 multi-task GP learning curve 분석에서 사용된 테크닉  을 참고하여,  
평균 Bayes error를 task 수 $T$, 각 task의 샘플 수 $n_t$의 함수로 표현하는 근사식을 도출하는 것을 목표로 한다.

#### (3) 학습 곡선(learning curve) 관점 해석

Ashton & Sollich의 multi-task GP learning curve 결과를 차용하여,   
본 연구에서 정의한 선형–가우시안 모형이 multi-task GP의 특수한 경우에 해당함을 보이고,  
similarity-aware prior의 학습 곡선을

$$
\epsilon(n) = \mathbb{E}\left[ (f_{t^\*}(x) - \hat{f}_{t^\*}(x))^2 \right]
$$

형태로 표현하거나 근사함으로써,

- task similarity 구조가 클수록,  
- 다른 task의 데이터가 많을수록,

새로운 task의 Bayes error가 더 빠르게 감소한다는 결과를 이론적으로 설명한다.

### 5.4 시뮬레이션 및 실증 연구 계획

1. **시뮬레이션 환경 구성**
   - Task feature 및 참 task covariance $\Sigma_{\text{true}}$ 를 설계하여,
     - (i) similarity-aware prior가 참 구조와 잘 맞는 경우,
     - (ii) 구조가 mismatch된 경우,
     - (iii) 실제로 task들이 독립인 경우,
     를 비교.
   - 각 설정에서 $T$, $n_t$를 변화시키며 독립 prior vs similarity-aware prior의  
     Bayes risk 및 학습 곡선을 비교.

2. **실제 데이터 기반 meta-learning 실험**
   - 공개된 few-shot 회귀/분류 데이터셋(예: UCI 회귀 데이터셋을 여러 task로 나눈 환경 등)에 대해,
   - task feature(예: 입력 분포 통계량, domain index 등)를 구성하고  
     제안 prior 구조를 적용.
   - 예측 정확도, 불확실성 calibration, 샘플 효율성 등의 지표 비교를 통해  
     이론 결과와의 일관성을 확인.

---

## 6. 기대 효과 및 학문적 기여

1. **이론적 기여**
   - Task 유사도를 반영한 계층 베이지안 meta-learning prior의 **일반적 구성 틀**을 제시하고,  
     그 유효성(positive definiteness)과 Bayes risk 측면의 이점을 **정리 형태로 제시**한다.
   - 선형–가우시안 계층 모형에서 similarity-aware prior와 독립 prior의 **risk/learning curve 비교 분석**을 통해,  
     기존 GP/meta-learning 문헌의 공백을 메운다.

2. **범용성 있는 방법론 제안**
   - 제안 prior 구조는 task feature, 그래프, 클러스터 등 다양한 유사도 정보를 커널/공분산 형태로 통합할 수 있어,  
     회귀, 분류, GP, BNN 등 다양한 meta-learning 환경에 적용 가능하다.

3. **Meta-learning 이론과 실용 알고리즘 간의 연결 강화**
   - Multi-task GP와 PAC-Bayesian meta-learning의 이론적 결과를  
     구체적인 prior 설계 문제와 연결함으로써,  
     meta-learning 알고리즘 설계에 대한 통계적·이론적 가이드를 제공한다.

---

## 7. 연구 일정 (예시: 석사 3학기 기준)

| 기간               | 내용                                                                 |
|--------------------|----------------------------------------------------------------------|
| 1학기 전반 (3–4월) | Meta-learning 및 Bayesian/meta-learning, GP, multi-task GP 문헌 조사 |
| 1학기 후반 (5–7월) | 모형 설정 구체화, prior 구조 정의, 기본 정리(유효성) 도출           |
| 여름 방학 (7–8월)  | Bayes risk/learning curve 이론적 분석, 초벌 증명 정리               |
| 2학기 전반 (9–10월)| 시뮬레이션 코드 구현, synthetic 실험 및 결과 분석                   |
| 2학기 후반 (11–1월)| 실증 데이터 실험, 결과 해석 및 이론과의 연결                         |
| 3학기 전반 (3–4월) | 논문 초고(1–4장) 작성, 정리/보완                                    |
| 3학기 후반 (5–7월) | 논문 최종 수정, 심사 준비 및 발표                                   |

(실제 일정은 지도교수와의 논의를 거쳐 조정 예정)

---

## 8. 참고 문헌 (예시)

- Hospedales, T., Antoniou, A., Micaelli, P., & Storkey, A. (2021). *Meta-Learning in Neural Networks: A Survey*.   
- Grant, E., Finn, C., Levine, S., Darrell, T., & Griffiths, T. (2018). *Recasting Gradient-Based Meta-Learning as Hierarchical Bayes*. ICLR.   
- Zou, Y., & Lu, X. (2020). *Gradient-EM Bayesian Meta-Learning*. NeurIPS.   
- Nguyen, Q. P., Low, B. K. H., & Jaillet, P. (2021). *Learning to Learn with Gaussian Processes*. UAI.   
- Ashton, S. R. F., & Sollich, P. (2012). *Learning Curves for Multi-task Gaussian Process Regression*. NeurIPS.   
- Rothfuss, J., Josifoski, M., Fortuin, V., & Krause, A. (2021). *Scalable PAC-Bayesian Meta-Learning via the PAC-Optimal Hyper-Posterior*.   
- Chai, K. M. A. (2010). *Multi-task Learning with Gaussian Processes*.   

(최종 참고 문헌 목록은 실제 논문 작성 시 추가·수정 예정)

  
