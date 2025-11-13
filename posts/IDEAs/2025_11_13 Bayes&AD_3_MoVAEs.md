---
title: "Mixture of Experts But VAE - Bayesian+AnomalyDetection"
date: 2025-11-13       
description: "졸업 논문 주제 구체화 - Bayesian+AnomalyDetection"
categories: [Bayesian, AnomalyDetection, Idea, MoE, VAE]
author: "김한울"
---



## 1. Mixture-of-Experts(MoE)의 기본 모양

가장 교과서적인 MoE는 보통 **지도학습** 문맥에서 소개됨:

* 입력: (x)
* 출력: (y)
* 여러 개의 expert (f_k(x))와 gating network (g(x))

모양은 대충 이렇게:

$$
y \approx \sum_{k=1}^K \pi_k(x), f_k(x), \qquad
\pi_k(x) = \text{softmax}_k(g(x))
$$

* **expert**: (f_k(x))

  * “모드 k일 때의 예측기” (회귀, 분류, 등등)
* **gating network**: (g(x))

  * 입력 (x)를 보고 “어느 expert를 얼마나 쓸지” 가중치 (\pi_k(x))를 냄
* **주 용도**:

  * 복잡한 함수 (x \mapsto y)를 여러 지역/모드로 나눠서 각자 다른 네트워크가 담당하게 하는 구조
  * 예: piecewise function, 여러 작업을 나눠 맡는 모델 등

→ 여기서 포인트는 보통 **“조건부 모델링 (p(y\mid x))”**를 한다는 것.

---

## 2. Mixture-of-VAEs 기본 모양

Mixture-of-VAEs는 말 그대로 **expert가 “VAE”인 mixture 모델**
즉, 각 expert가 **생성모델** (p_{\theta_k}(x))을 담당한다고 보면 됨.

구조는 대략:

1. 모드/클러스터 (k)를 latent로 도입:

$$
p(k) = \pi_k \quad (\text{또는 } \pi_k(x) = p(k\mid x) \text{로 gating})
$$

2. 모드별 VAE:

$$
z_k \sim p(z_k) = \mathcal{N}(0,I)
$$

$$
p_{\theta_k}(x \mid z_k), \quad q_{\phi_k}(z_k \mid x)
$$

3. 전체 분포:

$$
p(x) = \sum_{k=1}^K \pi_k, p_{\theta_k}(x)
\quad \text{(또는 } \sum_k \pi_k(x), p_{\theta_k}(x) \text{)}
$$

즉,

* **expert = “x를 생성하는 VAE 하나”**
* mixture는 **데이터 분포 (p(x))** 자체를 여러 모드로 나눠서 설명하는 구조.

이건 **Mixture-of-Experts의 unsupervised / generative 버전**이라고 볼 수 있음.

---

## 3. 공통점과 차이점 정리

### 3.1 공통점 (구조적인 측면)

1. **여러 개의 expert 존재**

   * MoE: (f_k(x)) (회귀/분류 등)
   * Mixture-of-VAEs: (p_{\theta_k}(x\mid z_k)) (생성모델)

2. **gating 개념**

   * MoE: (g(x))로부터 (\pi_k(x)) (입력 의존적인 expert 가중치)
   * Mixture-of-VAEs:

     * 간단 버전: (\pi_k) 고정 mixture weight
     * 조금 더 MoE 느낌: (\pi_k(x) = p(k\mid x))를 네트워크로 모델링 가능

3. **전체 출력은 “expert들의 weighted combination”**

   * MoE: (\sum_k \pi_k(x) f_k(x))
   * MoVAE: (\sum_k \pi_k(x) p_{\theta_k}(x)) (density 혹은 likelihood)

→ 그래서 개념적으로는 **“Mixture-of-VAEs도 MoE의 한 종류”**라고 보는 게 자연스러움

---

### 3.2 차이점 (보통 쓰이는 맥락 / modeling target)

1. **목표로 하는 확률분포가 다름**

   * 일반적인 MoE:

     * (p(y\mid x)) (조건부 분포, supervised)
     * 예측 문제: 입력-출력 mapping
   * Mixture-of-VAEs:

     * (p(x)) (joint / marginal 분포, unsupervised)
     * 생성/이상치 탐지 문제: 데이터 분포 자체를 모델링

2. **expert의 내부 구조**

   * MoE:

     * expert는 그냥 “네트워크 함수” (f_k(x))일 때가 많음
       (일반 MLP, CNN, RNN 등등)
   * Mixture-of-VAEs:

     * expert는 **VAE 전체** (encoder + decoder + latent z)
     * 내부에 또 하나의 latent 구조 (z_k)가 있어서 **두 단계 latent (k, z)** 가 됨
       → 일종의 hierarchical latent model

3. **학습 방식**

   * MoE (supervised):

     * 보통 backprop으로 end-to-end 학습
     * 혹은 EM-like 알고리즘 (예: hard gating 등)
   * Mixture-of-VAEs:

     * mixture model + VAE라서

       * mixture responsibility(E-step 비슷한 역할)와
       * 각 VAE parameter 업데이트(M-step 비슷한 역할)를
         번갈아 하거나 joint training
     * 논문/구현에 따라 EM 스타일 / ELBO 최적화 스타일 등이 변주됨

---

## 4. 네가 쓰려는 논문에서는 어떻게 부르면 좋을까?

지금 우리가 이야기한 구조는:

* 각 모드별로 **VAE 하나**가 있고
* 모드 index (k)에 대한 mixture가 있어서

$$
p(x) = \sum_k \pi_k, p_{\theta_k}(x)
$$

이거라서,

* **이론/수학 쪽에서 설명할 땐**

  * “Mixture-of-VAEs”, “mixture of latent variable models”
* **딥러닝/엔지니어링 쪽에서 설명할 땐**

  * “각 expert가 VAE인 Mixture-of-Experts 구조”
    라고 같이 언급해도 좋아.

> We adopt a mixture-of-VAEs architecture,
> which can be interpreted as a Mixture-of-Experts model
> where each expert is a VAE that models a specific mode of the data distribution.



---

## 5. 석사 논문 스토리에서 활용 포인트

1. **3장 – VAE 기반 이상치 + UQ**

   * 단일 모드(normal 상태가 하나라고 보는) 데이터 가정
   * $p(x)$를 하나의 VAE로 모델링

2. **4장 – Mixture-of-VAEs 확장**

   * “실제 현장 데이터는 여러 정상 모드(운영 상태)를 가진다”
   * 이를 위해 **Mixture-of-Experts 관점**에서,
     각 모드를 담당하는 VAE expert를 두고 gating/mixture 구조를 도입
   * 즉, “VAE를 expert로 하는 Mixture-of-Experts = Mixture-of-VAEs”

3. **기여 포인트 강조**

   * 기존 anomaly detection은 “하나의 정상 모드”만 가정하는 경우가 많다.
   * 우리는 **모드(k) + 모드 내부 latent(z)** 두 단계로 분해해

     * 모드별 이상치,
     * 모드 간 불확실성까지 구분해서 다룸.

---

## 6. 짧게 요약하면

* **Mixture-of-Experts**

  * 일반적으로 (p(y\mid x))를 여러 expert + gating으로 나눠 모델링하는 **조건부 모델**
  * expert는 보통 “함수 네트워크”
* **Mixture-of-VAEs**

  * 각 expert가 **VAE 같은 확률 생성모델**
  * (p(x)) (데이터 분포) 자체를 mixture로 모델링하는 **생성/unsupervised 모델**
  * 구조적으로는 “expert를 VAE로 둔 generative MoE”
