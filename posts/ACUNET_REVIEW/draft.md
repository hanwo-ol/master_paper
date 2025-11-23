---
title: "paper upgrade plan -draft"
date: 2025-11-23       
description: "AC U-Net upgrade"
categories: [Plan, U-Net, Idea]
author: "김한울"
---



# 연구 제안서: 시공간 태양광 일사량 예측을 위한 상황 인지형 통합 메타러닝 프레임워크
**(Context-Aware Unified Meta-Learning Framework for Spatiotemporal Solar Irradiance Forecasting)**

## 1. 서론 및 배경 (Introduction)

기존의 AC U-Net 모델은 결정론적(deterministic) 접근과 고정된 손실 함수를 사용하여, 급변하는 기상 상황이나 계절적 변동성에 유연하게 대처하는 데 한계가 있었다. 특히, 동일한 구름 형태라도 계절과 기상 상황에 따라 일사량에 미치는 영향이 다르다는 점은 모델이 단순한 패턴 인식을 넘어 **'상황에 따른 적응(Adaptation)'** 능력을 갖춰야 함을 시사한다.

본 연구에서는 이를 해결하기 위해 **Diffusion Probabilistic Model**을 기반으로 하여 다중 시점을 동시에 예측하는 **Multi2Multi** 구조로 전환하고, 서베이 논문[1]의 분류 체계에 기반한 4가지 메타러닝 전략을 하나의 프레임워크로 통합한 **'상황 인지형 적응 학습 시스템'**을 제안한다.

## 2. 문제 정의 및 노테이션 (Problem Definition & Notation)

### 2.1 기본 설정

*   **입력 시퀀스 ($\mathbf{X}$)**: 과거 $T_{in}$ 시점의 위성 영상 및 보조 데이터. $\mathbf{X} \in \mathbb{R}^{T_{in} \times C \times H \times W}$.
*   **출력 시퀀스 ($\mathbf{Y}$)**: 미래 $T_{out}$ 시점의 일사량 맵. $\mathbf{Y} \in \mathbb{R}^{T_{out} \times 1 \times H \times W}$.
*   **기반 모델 ($f_\theta$)**: 파라미터 $\theta$를 갖는 Diffusion 기반의 조건부 생성 모델. $p_\theta(\mathbf{Y}|\mathbf{X})$.

### 2.2 태스크(Task) 정의
메타러닝을 위해 전체 데이터셋을 개별적인 '기상 이벤트' 단위의 태스크로 정의한다.

*   **태스크 $\mathcal{T}_i$**: 특정 시점 $t$를 기준으로 샘플링된 기상 이벤트.
*   **데이터셋 구성**: 각 태스크 $\mathcal{T}_i$는 적응을 위한 서포트 셋(Support Set)과 평가를 위한 쿼리 셋(Query Set)으로 구성된다.
    *   $\mathcal{D}_{\text{sup}}^{(i)} = \{(\mathbf{X}_j, \mathbf{Y}_j)\}_{j=1}^{K}$: 해당 기상 이벤트의 초기 관측 데이터 (적응용).
    *   $\mathcal{D}_{\text{qry}}^{(i)} = \{(\mathbf{X}_k, \mathbf{Y}_k)\}_{k=1}^{Q}$: 해당 기상 이벤트의 미래 예측 데이터 (평가용).
*   **태스크 분포 $p(\mathcal{T})$**: 가능한 모든 기상 조건(맑음, 흐림, 태풍, 장마 등)의 분포.

## 3. 제안 방법론: 통합 메타러닝 프레임워크 (Proposed Methodology)

우리는 다음 4가지 메타 지식(Meta-Knowledge) 집합 $\Omega = \{\omega_{\text{init}}, \omega_{\text{opt}}, \omega_{\text{loss}}, \omega_{\text{curr}}\}$을 동시에 최적화하는 이중 루프(Bi-level) 최적화 문제를 정의한다.

### 3.1 구성 요소 (Components)

#### **Strategy 1: 상황 적응형 초기 조건 (Meta-Initialization)**

*   **Taxonomy**: Meta-Representation (Parameter Initialization)
*   **정의**: 모든 태스크에 대해 빠르게 적응할 수 있는 최적의 초기 파라미터 $\omega_{\text{init}}$.
*   **수식**: 내부 루프의 시작점 $\theta_0 = \omega_{\text{init}}$.

#### **Strategy 5: 적응형 메타-최적화기 (Meta-Optimizer)**

*   **Taxonomy**: Meta-Representation (Optimizer)
*   **정의**: 고정된 SGD나 Adam 대신, 현재의 파라미터 상태와 경사도(Gradient)를 입력받아 최적의 업데이트 벡터를 출력하는 신경망 $M_{\omega_{\text{opt}}}$.
*   **수식**: 내부 루프 업데이트 규칙.
    $$ \theta_{k+1} = \theta_k - M_{\omega_{\text{opt}}}(\nabla_{\theta_k} \mathcal{L}_{\text{inner}}, \theta_k) $$
    이는 날씨 변화율(Gradient의 크기 및 방향)에 따라 학습률을 동적으로 조절하는 효과를 갖는다.

#### **Strategy 2: 동적 손실 함수 (Dynamic Loss Function)**

*   **Taxonomy**: Meta-Representation (Losses)
*   **정의**: 기상 상황(Context)에 따라 픽셀 정확도($\mathcal{L}_{L1}$)와 구조적 유사성($\mathcal{L}_{SSIM}$)의 가중치 $\lambda$를 조절하는 네트워크 $h_{\omega_{\text{loss}}}$.
*   **수식**: 내부 루프 손실 함수.
    $$ \mathcal{L}_{\text{inner}}(\theta; \mathcal{D}_{\text{sup}}^{(i)}) = \lambda_i \cdot \mathcal{L}_{SSIM} + (1-\lambda_i) \cdot \mathcal{L}_{L1} $$
    $$ \text{where } \lambda_i = h_{\omega_{\text{loss}}}(\text{Encoder}(\mathbf{X} \in \mathcal{D}_{\text{sup}}^{(i)})) $$

#### **Strategy 4: 불확실성 기반 커리큘럼 (Meta-Curriculum)**

*   **Taxonomy**: Meta-Objective (Episode Design)
*   **정의**: 모델의 예측 불확실성(Uncertainty)이 높은 '어려운 태스크'를 우선적으로 샘플링하는 확률 분포 $q_{\omega_{\text{curr}}}(\mathcal{T})$.
*   **수식**: 외부 루프에서의 태스크 샘플링 가중치. 난이도 $H(\mathcal{T}_i)$에 비례하여 샘플링 확률을 조정.

### 3.2 통합 최적화 과정 (Unified Optimization)

전체 프레임워크는 내부 루프(Inner Loop)와 외부 루프(Outer Loop)로 구성된다.

**[Step 1: Inner Loop - Task Adaptation]**
태스크 $\mathcal{T}_i \sim q_{\omega_{\text{curr}}}(\mathcal{T})$가 주어졌을 때, 초기 파라미터 $\omega_{\text{init}}$에서 시작하여 $K$번의 적응 단계를 거친다. 이때 메타-최적화기($M$)와 동적 손실 함수($h$)를 사용한다.

$$ \theta_0^{(i)} = \omega_{\text{init}} $$
$$ \theta_{k+1}^{(i)} = \theta_k^{(i)} - M_{\omega_{\text{opt}}} \left( \nabla_{\theta} \mathcal{L}_{\text{inner}}(\theta_k^{(i)}; \mathcal{D}_{\text{sup}}^{(i)}, \omega_{\text{loss}}), \theta_k^{(i)} \right) $$
$$ \text{Result: Adapted Parameter } \theta_{K}^{(i)} $$

**[Step 2: Outer Loop - Meta Update]**
적응된 파라미터 $\theta_{K}^{(i)}$를 사용하여 쿼리 셋 $\mathcal{D}_{\text{qry}}^{(i)}$에 대한 일반화 성능을 평가하고, 메타 파라미터 $\Omega$를 업데이트한다. 외부 루프의 손실 함수 $\mathcal{L}_{\text{outer}}$는 객관적인 평가를 위해 고정된 지표(예: MSE)를 사용한다.

$$ \min_{\Omega} \mathbb{E}_{\mathcal{T}_i \sim q_{\omega_{\text{curr}}}(\mathcal{T})} \left[ \mathcal{L}_{\text{outer}}(\theta_{K}^{(i)}; \mathcal{D}_{\text{qry}}^{(i)}) \right] $$

## 4. 기대 효과 및 결론 (Expected Contribution)

본 제안서는 서베이 논문의 Taxonomy를 기반으로 4가지 핵심 전략을 통합하였다.

1.  **Multi2Multi Diffusion**: 시점 간 일관성을 유지하며 불확실성을 확률적으로 모델링한다.
2.  **Meta-Initialization & Optimizer**: 급변하는 기상 상황(Gradient가 큰 상황)과 안정적인 상황을 구분하여 학습 속도를 조절함으로써 **적응성(Adaptability)**을 극대화한다.
3.  **Dynamic Loss**: 구름의 형태가 중요한 상황과 픽셀 값이 중요한 상황을 스스로 판단하여 **목적 함수를 유연하게 변경**한다.
4.  **Curriculum**: 예측이 어려운 태스크를 집중 학습하여 모델의 **강건성(Robustness)**을 확보한다.

이 통합 프레임워크는 기존의 정적인 딥러닝 모델이 갖는 한계를 극복하고, 기상학적 특성을 데이터로부터 스스로 학습하는 차세대 태양광 예측 모델이 될 것이다.

---
**참고 문헌**
[1] T. Hospedales, A. Antoniou, P. Micaelli, and A. Storkey, "Meta-Learning in Neural Networks: A Survey," IEEE Transactions on Pattern Analysis and Machine Intelligence, 2021.