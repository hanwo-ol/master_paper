---
title: "paper upgrade plan"
date: 2025-11-23       
description: "AC U-Net upgrade"
categories: [Plan, U-Net, Idea]
author: "김한울"
---

"같은 구름이라도 계절마다 일사량에 미치는 영향력이 다르다". 

**'상황(Context)'에 따라 구름의 물리적 의미를 다르게 해석하고 적용하는 능력**, 즉 **적응 능력(Adaptability)**을 배워야 함을 의미합니다.

---

### 메타러닝 기반 개선 전략?

#### **전략 1: 상황 적응형 초기 조건 학습 (MAML 스타일)**

*   **아이디어**: 어떤 날씨 상황이든 빠르게 적응할 수 있는 '만능 초기 U-Net'을 학습합니다.
*   **Taxonomy**: **Meta-Representation (Parameter Initialization)**
*   **Task 정의**: 하나의 '날씨 이벤트'를 하나의 태스크 $\mathcal{T}_i$로 정의합니다.
    *   **표기법**:
        *   태스크 $\mathcal{T}_i$: 특정 시점 $t$를 중심으로 한 짧은 영상 시퀀스.
        *   $D_{\text{train}}^{(i)}$ (Support Set): 이벤트의 초기 관측 영상. 예: $[I_{t-4}, I_{t-3}, I_{t-2}]$
        *   $D_{\text{val}}^{(i)}$ (Query Set): 해당 이벤트의 예측 대상 영상. 예: $[I_{t+1}, I_{t+2}, I_{t+3}, I_{t+4}]$
*   **설명**:
    메타러닝의 목표는 모든 태스크에 대해 다음 손실을 최소화하는 단일 초기 가중치 $\omega$ (U-Net의 초기 파라미터)를 찾는 것입니다.
    $$ \omega^* = \arg\min_{\omega} \sum_i \mathcal{L}(D_{\text{val}}^{(i)}; \theta'_i) \quad \text{where} \quad \theta'_i = \text{Update}(\omega, D_{\text{train}}^{(i)}) $$
    즉, "어떤 날씨(태풍, 맑은 날, 장마)의 초기 몇 프레임만 보고 잠깐 학습(Update)해도, 그 날씨의 미래를 가장 잘 예측할 수 있는 **최적의 사전 지식($\omega$)**"을 배우는 것입니다. 이는 모델이 새로운 기상 패턴에 매우 빠르게 적응하는 능력을 갖추게 합니다.

#### **전략 2: 날씨 패턴에 따른 동적 손실 함수 메타러닝**

*   **아이디어**: 날씨 상황에 따라 '정확도'와 '구조적 유사성'의 중요도를 동적으로 조절하는 손실 함수를 학습합니다.
*   **Taxonomy**: **Meta-Representation (Loss Function)**
*   **Task 정의**: 전략 1과 동일하게 '날씨 이벤트'를 태스크 $\mathcal{T}_i$로 정의합니다.
*   **설명**:
    현재 손실 함수 $\mathcal{L}_{\text{total}} = \alpha \cdot \mathcal{L}_{\text{MS-SSIM}} + (1 - \alpha) \cdot \mathcal{L}_{\text{L1}}$에서 가중치 $\alpha$는 고정되어 있습니다. 이 전략에서는 $\alpha$를 메타러닝의 대상($\omega$)으로 봅니다. 작은 신경망 $f_\omega$가 입력 영상의 특징(예: 인코더의 병목 특징)을 보고 최적의 $\alpha_i$ 값을 출력합니다.
    $$ \omega^* = \arg\min_{\omega} \sum_i \mathcal{L}_{\text{total}}(D_{\text{val}}^{(i)}; \theta_i, \alpha_i=f_\omega(D_{\text{train}}^{(i)})) $$
    이를 통해 모델은 다음과 같이 학습할 수 있습니다.
    *   **맑은 날**: 구름이 거의 없으므로 구조보다 픽셀 정확도가 중요. 모델이 $\alpha \approx 0$을 출력하여 $\mathcal{L}_{\text{L1}}$에 집중.
    *   **복잡한 구름**: 구름의 형태와 움직임이 중요. 모델이 $\alpha$ 값을 높여 $\mathcal{L}_{\text{MS-SSIM}}$에 집중.

#### **전략 3: 기상 현상 전문가 모듈 조합 학습**

*   **아이디어**: '안개 전문가', '태풍 전문가', '뭉게구름 전문가' 등 특정 기상 현상에 특화된 여러 디코더(모듈)를 만들어두고, 상황에 맞게 이들을 조합하는 방법을 메타러닝합니다.
*   **Taxonomy**: **Meta-Representation (Modules)**
*   **Task 정의**: 전략 1과 동일하게 '날씨 이벤트'를 태스크 $\mathcal{T}_i$로 정의합니다.
*   **설명**:
    U-Net의 인코더는 공유하고, 디코더는 $K$개의 전문가 디코더 $\{Dec_1, \dots, Dec_K\}$로 구성합니다. 메타러너 $f_\omega$는 입력 영상의 특징을 보고 각 전문가의 결과물을 어떻게 섞을지에 대한 가중치 $w_1, \dots, w_K$를 출력합니다.
    $$ \text{Final Prediction} = \sum_{k=1}^K w_k \cdot Dec_k(\text{Encoder}(X)) \quad \text{where} \quad [w_1, \dots, w_K] = f_\omega(\text{Encoder}(X)) $$
    메타러닝은 최종 예측의 성능을 최대화하는 조합기($f_\omega$)를 학습합니다. 이는 단일 모델이 모든 상황을 처리하려는 부담을 줄이고, '구성적 학습(Compositional Learning)'을 통해 더 복잡한 현상에 대응할 수 있게 합니다.

#### **전략 4: 예측 불확실성 기반의 커리큘럼 메타러닝**

*   **아이디어**: 모델이 예측하기 어려워하는 '어려운 문제'를 집중적으로 학습시켜 강인함(Robustness)을 기릅니다.
*   **Taxonomy**: **Meta-Objective (Episode Design)**
*   **Task 정의**:
    *   태스크 $\mathcal{T}_i$: 전략 1과 동일한 '날씨 이벤트'.
    *   **난이도 $H(\mathcal{T}_i)$**: 베이스라인 모델의 예측 오차 또는 Diffusion 모델의 경우 여러 번 샘플링했을 때 결과의 분산(Uncertainty)으로 정의.
*   **설명**:
    메타-훈련 시, 모든 태스크를 무작위로 샘플링하지 않습니다. 주기적으로 현재 모델이 가장 어려워하는 태스크(난이도 $H(\mathcal{T}_i)$가 높은 태스크)를 더 높은 확률로 샘플링하여 훈련 데이터 배치에 포함시킵니다. 이는 서베이에서 언급된 'Hard Task Meta-Batch'와 같은 커리큘럼 학습 방식입니다. 이 전략의 메타-목표는 단순히 평균 성능을 높이는 것이 아니라, **'최악의 경우(worst-case) 성능'을 개선**하고 예측이 어려운 기상 현상에 대한 모델의 강건성을 극대화하는 것입니다.

#### **전략 5: 날씨 변화율에 적응하는 메타-최적화기 학습**

*   **아이디어**: 날씨가 급변할 때와 안정적일 때, 모델의 가중치를 업데이트하는 '학습률'이나 '방향'을 다르게 적용하는 맞춤형 최적화기를 학습합니다.
*   **Taxonomy**: **Meta-Representation (Optimizer)**
*   **Task 정의**: 전략 1과 동일하게 '날씨 이벤트'를 태스크 $\mathcal{T}_i$로 정의합니다.
*   **설명**:
    기존의 AdamW 같은 고정된 최적화기 대신, 작은 RNN이나 LSTM 기반의 메타-최적화기 $M_\omega$를 학습합니다. 이 최적화기는 내부 루프의 각 스텝에서 현재 모델의 파라미터 $\theta_t$와 손실의 경사도 $\nabla \mathcal{L}(\theta_t)$를 입력받아, 다음 파라미터 업데이트 값 $\Delta \theta_t$를 출력합니다.
    $$ \theta_{t+1} = \theta_t - M_\omega(\theta_t, \nabla \mathcal{L}(\theta_t)) $$
    외부 루프는 이 최적화기($M_\omega$)를 사용하여 최종 예측 성능이 가장 좋아지도록 $\omega$를 업데이트합니다. 이를 통해 모델은 다음과 같은 동적인 학습 전략을 배울 수 있습니다.
    *   **구름이 빠르게 이동/생성될 때**: 경사도 정보를 더 신뢰하고 큰 폭으로 파라미터를 업데이트.
    *   **날씨가 안정적일 때**: 미세한 변화에 과민 반응하지 않도록 작은 폭으로 파라미터를 업데이트.