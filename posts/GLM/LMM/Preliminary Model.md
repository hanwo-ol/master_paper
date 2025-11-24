---
title: "Linear Mixed Model - Preliminary Model"
date: 2025-11-24       
description: "20251124 GLM class notes"
categories: [lecture, glm, lmm, LinearMixedModel, PreliminaryModel]
author: "김한울"
---

## Linear Mixed Model의 Preliminary Model 

### 개요

Preliminary Model은 선형혼합모형(Linear Mixed Model, LMM)의 기초가 되는 **주변 모형(Marginal Model)** 관점을 다룹니다. 이 섹션은 고정효과 추정과 분산 성분 추정의 이론적 기초를 제공하며, 후속 내용인 최대우도추정(MLE)과 제한최대우도추정(REML)으로 연결됩니다.

***

### 1. 주변 모형(Marginal Model) 개념

#### 기본 아이디어

선형혼합모형은 다음과 같이 표현됩니다:

**Stage 1 (개체 내 변이)**
$$
\mathbf{Y}_i = \mathbf{X}_i \boldsymbol{\beta} + \mathbf{Z}_i \mathbf{b}_i + \boldsymbol{\epsilon}_i
$$

여기서:
- $$\mathbf{Y}_i$$: $$i$$번째 개체의 반응벡터 ($$n_i \times 1$$)
- $$\mathbf{X}_i$$: $$i$$번째 개체의 고정효과 설계행렬 ($$n_i \times p$$)
- $$\boldsymbol{\beta}$$: 고정효과 계수벡터 ($$p \times 1$$)
- $$\mathbf{Z}_i$$: $$i$$번째 개체의 랜덤효과 설계행렬 ($$n_i \times q$$)
- $$\mathbf{b}_i$$: 개체 특이적 랜덤효과 ($$q \times 1$$), $$\mathbf{b}_i \sim N(\mathbf{0}, \mathbf{G})$$
- $$\boldsymbol{\epsilon}_i$$: 측정오차 ($$n_i \times 1$$), $$\boldsymbol{\epsilon}_i \sim N(\mathbf{0}, \mathbf{R}_i)$$

#### 주변 모형 유도

랜덤효과 $$\mathbf{b}_i$$를 적분으로 제거하면(마지널화), 다음의 **주변 모형**을 얻습니다:

$$
E[\mathbf{Y}_i] = \mathbf{X}_i \boldsymbol{\beta}
$$

$$
\text{Var}[\mathbf{Y}_i] = \mathbf{V}_i = \mathbf{Z}_i \mathbf{G} \mathbf{Z}_i^T + \mathbf{R}_i
$$

따라서 $$\mathbf{Y}_i \sim N(\mathbf{X}_i \boldsymbol{\beta}, \mathbf{V}_i)$$

**중요한 특징**: 주변 모형은 랜덤효과의 존재를 **명시적으로 가정하지 않습니다**. 대신 공분산 구조 $$\mathbf{V}_i$$로 개체 간 관측값의 상관성과 개체 내 이질성을 모델링합니다.

***

### 2. 기호법(Notation)

#### 매개변수 정의

| 기호 | 의미 | 차원 |
|------|------|------|
| $$\boldsymbol{\beta}$$ | 고정효과 계수 벡터 | $$p \times 1$$ |
| $$\boldsymbol{\alpha}$$ | 분산 성분 벡터 (모든 $$\mathbf{G}$$와 $$\mathbf{R}$$의 원소) | 변수 |
| $$\boldsymbol{\theta}$$ | 주변 모형의 전체 매개변수 벡터 = $$[\boldsymbol{\beta}^T, \boldsymbol{\alpha}^T]^T$$ | - |
| $$\mathbf{V}_i$$ | $$i$$번째 개체의 공분산 행렬 = $$\mathbf{Z}_i \mathbf{G} \mathbf{Z}_i^T + \mathbf{R}_i$$ | $$n_i \times n_i$$ |
| $$\mathbf{W}_i$$ | 가중 행렬 = $$\mathbf{V}_i^{-1}$$ | $$n_i \times n_i$$ |

***

### 3. 주변 우도함수(Marginal Likelihood Function)

#### 전체 우도 함수

$$N$$명의 개체로부터 수집된 데이터에 대한 주변 우도함수는:

$$
L(\boldsymbol{\theta}) = \prod_{i=1}^{N} (2\pi)^{-n_i/2} |\mathbf{V}_i|^{-1/2} \exp\left\{-\frac{1}{2}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i^{-1} (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})\right\}
$$

로그 우도함수(Log-likelihood):

$$
\log L(\boldsymbol{\theta}) = -\frac{1}{2}\sum_{i=1}^{N}\left[n_i \log(2\pi) + \log|\mathbf{V}_i| + (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i^{-1} (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})\right]
$$

***

### 4. 고정효과 추정

#### $$\boldsymbol{\alpha}$$가 알려진 경우

분산 성분 $$\boldsymbol{\alpha}$$를 알고 있다고 가정하면, 로그 우도함수를 $$\boldsymbol{\beta}$$에 대해 최대화할 때:

$$
\frac{\partial \log L}{\partial \boldsymbol{\beta}} = \sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta}) = \mathbf{0}
$$

이를 풀면, **일반화 최소제곱(Generalized Least Squares, GLS) 추정량**을 얻습니다:

$$
\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}) = \left(\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{W}_i \mathbf{X}_i\right)^{-1} \sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{W}_i \mathbf{Y}_i
$$

여기서 $$\mathbf{W}_i = \mathbf{V}_i^{-1}$$는 **가중행렬**입니다.

**의미**: 각 개체별 공분산 구조를 반영한 가중치 $$\mathbf{W}_i$$를 사용하여 고정효과를 추정합니다. 측정오차가 큰 관측값일수록 가중치가 작아집니다.

#### $$\boldsymbol{\alpha}$$가 미지수인 경우

실제로 분산 성분은 미지수이므로, $$\boldsymbol{\alpha}$$의 추정량 $$\hat{\boldsymbol{\alpha}}$$로 대체합니다:

$$
\hat{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}}(\hat{\boldsymbol{\alpha}})
$$

이것이 **일반화 최소제곱의 시행적 적용(Iterative Generalized Least Squares)**입니다.

***

### 5. 분산 성분 추정 방법

#### 5.1 최대우도추정(Maximum Likelihood Estimation, MLE)

**원리**: 분산 성분 $$\boldsymbol{\alpha}$$는 프로파일 우도함수(profile likelihood)를 최대화하여 추정합니다.

**절차**:
1. $$\boldsymbol{\alpha}$$의 초기값을 설정
2. 각 $$\boldsymbol{\alpha}$$ 값에 대해 $$\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})$$를 계산
3. 프로파일 우도 $$L_{ML}(\boldsymbol{\alpha}, \hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}))$$를 최대화하는 $$\boldsymbol{\alpha}$$를 찾음
4. 최종 추정량: $$\hat{\boldsymbol{\alpha}}_{ML}$$, $$\hat{\boldsymbol{\beta}}_{ML} = \hat{\boldsymbol{\beta}}(\hat{\boldsymbol{\alpha}}_{ML})$$

**문제점**: MLE는 고정효과 매개변수의 수를 추정할 때 자유도 손실을 고려하지 않아, 분산 성분을 **과소 추정(biased downward)**하는 경향이 있습니다.

#### 5.2 제한최대우도추정(Restricted Maximum Likelihood Estimation, REML)

**동기**: MLE의 편향 문제를 해결하기 위해 개발되었습니다.

**핵심 아이디어**: 고정효과 $$\boldsymbol{\beta}$$에 대한 정보를 제거한 **오차 대비(error contrasts)**를 사용합니다.

**수학적 구성**:

데이터 $$\mathbf{Y}$$를 다음과 같이 변환합니다:

$$
\mathbf{U} = \mathbf{A}^T \mathbf{Y}
$$

여기서 $$\mathbf{A}$$는 $$N \times (N-p)$$ 행렬로서 다음을 만족합니다:
- $$\mathbf{A}^T \mathbf{A} = \mathbf{I}_{N-p}$$ (직교성)
- $$\mathbf{A}^T \mathbf{X} = \mathbf{0}$$ ($$\mathbf{X}$$와 직교)

변환 후:

$$
E[\mathbf{U}] = \mathbf{A}^T \mathbf{Z} \mathbf{b} + \mathbf{A}^T \boldsymbol{\epsilon}
$$

$$\mathbf{X}$$가 제거되어 $$\boldsymbol{\beta}$$에 의존하지 않습니다.

**REML 우도함수**:

$$
L_{REML}(\boldsymbol{\alpha}) = (2\pi)^{-(N-p)/2} |\mathbf{A}^T \mathbf{V} \mathbf{A}|^{-1/2} \exp\left\{-\frac{1}{2} \mathbf{U}^T (\mathbf{A}^T \mathbf{V} \mathbf{A})^{-1} \mathbf{U}\right\}
$$

또는 동등하게:

$$
L_{REML}(\boldsymbol{\alpha}) \propto |\mathbf{V}|^{-1/2} |\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}|^{1/2} \exp\left\{-\frac{1}{2} (\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}})^T \mathbf{V}^{-1} (\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}})\right\}
$$

**장점**:
- 분산 성분의 **불편추정량(unbiased estimate)** 제공
- 고정효과의 추정에 필요한 자유도를 자동으로 반영
- 반복측정 데이터 분석에서 표준 방법

**비고**: REML은 분산 성분 추정에만 사용되며, 추정된 분산 성분은 고정효과 추정에 다시 사용됩니다($$\hat{\boldsymbol{\beta}}_{REML} = \hat{\boldsymbol{\beta}}(\hat{\boldsymbol{\alpha}}_{REML})$$).

***

### 6. MLE와 REML의 비교: 간단한 예시

#### 예시: 정규분포에서 분산 추정

$$N$$개 독립 관측치 $$Y_1, \ldots, Y_N \sim N(\mu, \sigma^2)$$에서:

**경우 1: $$\mu$$가 알려진 경우**

$$
\hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_{i=1}^{N}(Y_i - \mu)^2
$$

$$E[\hat{\sigma}^2_{MLE}] = \sigma^2$$ ✓ 불편

**경우 2: $$\mu$$가 미지수인 경우**

$$
\hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_{i=1}^{N}(Y_i - \bar{Y})^2
$$

$$E[\hat{\sigma}^2_{MLE}] = \frac{N-1}{N}\sigma^2$$ ✗ 편향 (과소추정)

**REML 추정량**:

$$
\hat{\sigma}^2_{REML} = \frac{1}{N-1}\sum_{i=1}^{N}(Y_i - \bar{Y})^2
$$

$$E[\hat{\sigma}^2_{REML}] = \sigma^2$$ ✓ 불편

이 간단한 예시는 Preliminary Model 섹션에서 제시되는 핵심 논리를 보여줍니다.

***

### 7. 실제 적용의 의미

#### 왜 주변 모형인가?

주변 모형 관점의 장점:

1. **모델 불확정성 해소**: 랜덤효과 구조가 명확하지 않은 경우에도, 공분산 구조 $$\mathbf{V}_i$$를 직접 모델링하여 분석 가능

2. **인구 평균 해석(Population Averaging)**: 고정효과 $$\boldsymbol{\beta}$$는 모집단 평균 효과를 나타내므로 임상 해석이 직관적

3. **불균형 데이터 처리**: 각 개체별로 다른 수의 측정값을 가져도 $$\mathbf{V}_i$$로 자동 처리

#### 고정효과 vs 분산 성분 추정의 구분

- **고정효과** $$\boldsymbol{\beta}$$: 분산 성분이 알려졌을 때 GLS로 추정 → 효율적이고 편향 없음
- **분산 성분** $$\boldsymbol{\alpha}$$: REML로 추정 → 자유도 손실 반영, 불편 추정량 제공

이 이원적 추정 구조(two-stage estimation)가 LMM의 핵심입니다.

***

### 8. Preliminary Model과 후속 내용의 연결

| 단계 | 내용 | 목적 |
|------|------|------|
| **Preliminary Model** | 주변 모형 정의, MLE와 REML 소개 | 이론적 기초 구축 |
| **Parameter Estimation** | MLE와 REML의 상세 유도 | 실제 추정 알고리즘 |
| **Analysis with LMM** | 주변 vs 랜덤효과 모형 선택 | 실무 적용 기준 |
| **Marginal & Random Effect Model** | 두 모형의 관계와 차이 | 해석 방법 |

***

### 요약

**Preliminary Model 섹션의 핵심 메시지**:

1. 선형혼합모형은 **주변 모형** 관점에서 이해되며, 공분산 구조 $$\mathbf{V}_i = \mathbf{Z}_i \mathbf{G} \mathbf{Z}_i^T + \mathbf{R}_i$$로 개체 간 상관성을 모델링합니다.

2. 고정효과는 분산 성분을 알 때 **GLS**로, 모를 때 **반복적 GLS**로 추정합니다.

3. 분산 성분은 **MLE**(편향)와 **REML**(불편)로 추정할 수 있으며, 실무에서는 REML을 선호합니다.

4. 이러한 두 단계 추정 구조가 불균형 데이터와 결측치가 있는 반복측정 설계에서 유연성을 제공합니다.

---

## 고정효과 추정: GLS 공식의 상세 전개

이 섹션은 **정규 방정식(normal equation)에서 GLS 추정량까지의 완전한 수학적 전개**를 단계별로 설명합니다.[1]

***

### 1. 우도함수 설정 및 미분

#### 주변 모형의 로그 우도함수 (복습)

선형혼합모형에서 $$i$$번째 개체의 데이터는 다음을 따릅니다:

$$
\mathbf{Y}_i \sim N(\mathbf{X}_i \boldsymbol{\beta}, \mathbf{V}_i)
$$

여기서:
- $$\mathbf{Y}_i$$: $$n_i \times 1$$ 반응벡터
- $$\mathbf{X}_i$$: $$n_i \times p$$ 설계행렬
- $$\boldsymbol{\beta}$$: $$p \times 1$$ 고정효과 계수 (모든 개체에 동일)
- $$\mathbf{V}_i = \mathbf{Z}_i \mathbf{G} \mathbf{Z}_i^T + \mathbf{R}_i$$: $$n_i \times n_i$$ 공분산행렬

**전체 데이터에 대한 로그 우도함수** (분산 성분 $$\boldsymbol{\alpha}$$는 알려졌다고 가정):

$$
\ell(\boldsymbol{\beta}) = \sum_{i=1}^{N} \ell_i(\boldsymbol{\beta})
$$

여기서 각 개체별 로그 우도는:

$$
\ell_i(\boldsymbol{\beta}) = -\frac{1}{2}\left[n_i \log(2\pi) + \log|\mathbf{V}_i| + (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})\right]
$$

$$\boldsymbol{\beta}$$를 포함하지 않는 항들을 제거하면:

$$
\ell(\boldsymbol{\beta}) \propto -\frac{1}{2}\sum_{i=1}^{N} (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})
$$

***

#### 정규 방정식 유도

$$\boldsymbol{\beta}$$에 대한 로그 우도함수의 **1차 미분** (정규 방정식):

$$
\frac{\partial \ell(\boldsymbol{\beta})}{\partial \boldsymbol{\beta}} = -\frac{1}{2}\sum_{i=1}^{N} \frac{\partial}{\partial \boldsymbol{\beta}} (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})
$$

**연쇄 법칙 적용**:

임의의 벡터 $$\mathbf{u}(\boldsymbol{\beta})$$와 대칭행렬 $$\mathbf{A}$$에 대해:

$$
\frac{\partial}{\partial \boldsymbol{\beta}} [\mathbf{u}(\boldsymbol{\beta})]^T \mathbf{A} \mathbf{u}(\boldsymbol{\beta}) = 2 \left[\frac{\partial \mathbf{u}^T}{\partial \boldsymbol{\beta}}\right] \mathbf{A} \mathbf{u}(\boldsymbol{\beta})
$$

여기서 $$\mathbf{u} = \mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta}$$이고, $$\frac{\partial \mathbf{u}}{\partial \boldsymbol{\beta}} = -\mathbf{X}_i$$입니다.

따라서:

$$
\frac{\partial}{\partial \boldsymbol{\beta}} (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta}) = 2(-\mathbf{X}_i)^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})
$$

$$
= -2\mathbf{X}_i^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})
$$

**정규 방정식** (1차 미분을 0으로 설정):

$$
\frac{\partial \ell}{\partial \boldsymbol{\beta}} = -\frac{1}{2}\sum_{i=1}^{N} \left[-2\mathbf{X}_i^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})\right] = \mathbf{0}
$$

정리하면:

$$
\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta}) = \mathbf{0}
$$

또는 다르게 표현하면:

$$
\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}\mathbf{Y}_i = \sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}\mathbf{X}_i \boldsymbol{\beta}
$$

***

### 2. 정규 방정식에서 GLS 추정량까지

#### 정규 방정식의 행렬 형태로 정리

$$
\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}\mathbf{X}_i \boldsymbol{\beta} = \sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}\mathbf{Y}_i
$$

**가중 행렬** $$\mathbf{W}_i = \mathbf{V}_i^{-1}$$를 정의하면:

$$
\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{W}_i \mathbf{X}_i \boldsymbol{\beta} = \sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{W}_i \mathbf{Y}_i
$$

#### GLS 추정량 도출

양변에서 $$\left[\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{W}_i \mathbf{X}_i\right]$$의 역행렬을 곱합니다:

$$
\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}) = \left(\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{W}_i \mathbf{X}_i\right)^{-1} \sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{W}_i \mathbf{Y}_i
$$

**행렬 표현**:

전체 데이터를 $$\mathbf{Y} = (\mathbf{Y}_1^T, \ldots, \mathbf{Y}_N^T)^T$$, $$\mathbf{X} = (\mathbf{X}_1^T, \ldots, \mathbf{X}_N^T)^T$$로 블록 행렬로 나타내면:

$$
\mathbf{V} = \begin{pmatrix} \mathbf{V}_1 & \mathbf{0} & \cdots & \mathbf{0} \\ \mathbf{0} & \mathbf{V}_2 & \cdots & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{0} & \mathbf{0} & \cdots & \mathbf{V}_N \end{pmatrix}
$$

$$\mathbf{W} = \mathbf{V}^{-1}$$는 블록 대각 행렬이므로:

$$
\mathbf{W} = \begin{pmatrix} \mathbf{W}_1 & \mathbf{0} & \cdots & \mathbf{0} \\ \mathbf{0} & \mathbf{W}_2 & \cdots & \mathbf{0} \\ \vdots & \vdots & \ddots & \vdots \\ \mathbf{0} & \mathbf{0} & \cdots & \mathbf{W}_N \end{pmatrix}
$$

따라서:

$$
\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}) = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{Y}
$$

이것이 **일반화 최소제곱(Generalized Least Squares, GLS)** 추정량의 표준 형태입니다.

***

### 3. GLS 추정량의 성질

#### 3.1 불편성(Unbiasedness)

$$\boldsymbol{\alpha}$$가 알려져 있을 때, $$\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})$$는 $$\boldsymbol{\beta}$$의 **불편추정량**입니다.

**증명**:

$$
E[\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})] = E\left[(\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{Y}\right]
$$

$$\mathbf{X}$$는 확정적(deterministic)이므로:

$$
= (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} E[\mathbf{Y}]
$$

$$E[\mathbf{Y}] = \mathbf{X} \boldsymbol{\beta}$$를 이용하면:

$$
= (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{X} \boldsymbol{\beta} = \boldsymbol{\beta}
$$

따라서 $$\hat{\boldsymbol{\beta}}$$는 불편추정량입니다. ✓

#### 3.2 공분산 행렬(Covariance Matrix)

$$
\text{Var}[\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})] = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \text{Var}[\mathbf{Y}] \mathbf{W} \mathbf{X} (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1}
$$

$$\text{Var}[\mathbf{Y}] = \mathbf{V}$$이고 $$\mathbf{W} = \mathbf{V}^{-1}$$이므로:

$$
\text{Var}[\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})] = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{W} \mathbf{V} \mathbf{W} \mathbf{X} (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1}
$$

$$\mathbf{W}\mathbf{V}\mathbf{W} = \mathbf{W}$$이므로 (왜냐하면 $$\mathbf{W} = \mathbf{V}^{-1}$$):

$$
\text{Var}[\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})] = (\mathbf{X}^T \mathbf{W} \mathbf{X})^{-1}
$$

더 명시적으로:

$$
\text{Var}[\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})] = \left(\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1} \mathbf{X}_i\right)^{-1}
$$

#### 3.3 최적성(Optimality): Gauss-Markov 정리

**정리**: $$\mathbf{Y}$$가 선형 모형 $$\mathbf{Y} \sim N(\mathbf{X}\boldsymbol{\beta}, \mathbf{V})$$를 따를 때, GLS 추정량 $$\hat{\boldsymbol{\beta}}$$는 모든 선형 불편추정량 중에서 **가장 작은 분산**을 가집니다(BLUE: Best Linear Unbiased Estimator).

***

### 4. GLS와 OLS(최소제곱법)의 비교

#### OLS (일반적인 선형 회귀)

OLS는 공분산 구조가 $$\mathbf{V}_i = \sigma^2 \mathbf{I}_{n_i}$$ (관측값들이 독립이고 동일 분산)인 특수한 경우입니다.

$$
\hat{\boldsymbol{\beta}}_{OLS} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{Y}
$$

#### GLS와의 관계

GLS는 OLS를 **일반화**합니다:

$$
\hat{\boldsymbol{\beta}}_{GLS} = (\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{V}^{-1} \mathbf{Y}
$$

**변환 해석**: GLS는 다음의 변환 선형 모형에 OLS를 적용하는 것과 동등합니다.

$$\mathbf{V}_i^{-1/2}$$를 사용하여 데이터를 화이트닝(whitening)하면:

$$
\mathbf{V}_i^{-1/2}\mathbf{Y}_i = \mathbf{V}_i^{-1/2}\mathbf{X}_i \boldsymbol{\beta} + \mathbf{V}_i^{-1/2}\boldsymbol{\epsilon}_i
$$

변환된 데이터는 $$\text{Var}[\mathbf{V}_i^{-1/2}\boldsymbol{\epsilon}_i] = \mathbf{I}_{n_i}$$를 만족하므로 (독립), 변환 데이터에 OLS를 적용하면 GLS와 동일한 결과를 얻습니다.

***

### 5. 수치 예제: 간단한 2개체 사례

#### 데이터 설정

각 개체에서 2개 시점에서 측정:

- 개체 1: $$\mathbf{Y}_1 = (y_{11}, y_{12})^T$$, $$\mathbf{X}_1 = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$$
  
- 개체 2: $$\mathbf{Y}_2 = (y_{21}, y_{22})^T$$, $$\mathbf{X}_2 = \begin{pmatrix} 1 & 0 \\ 1 & 1 \end{pmatrix}$$

**고정효과**: $$\boldsymbol{\beta} = (\beta_0, \beta_1)^T$$ (절편과 시간 효과)

**공분산 구조** (교환 가능, exchangeable):

$$
\mathbf{V}_i = \begin{pmatrix} \sigma^2 & \rho\sigma^2 \\ \rho\sigma^2 & \sigma^2 \end{pmatrix}, \quad i=1,2
$$

#### GLS 계산

**역행렬**:

$$
\mathbf{V}_i^{-1} = \frac{1}{\sigma^2(1-\rho^2)} \begin{pmatrix} 1 & -\rho \\ -\rho & 1 \end{pmatrix}
$$

**고정 공분산** $$\sigma^2 = 1$$, $$\rho = 0.5$$라 가정:

$$
\mathbf{V}_i^{-1} = \frac{1}{0.75} \begin{pmatrix} 1 & -0.5 \\ -0.5 & 1 \end{pmatrix} = \begin{pmatrix} 1.33 & -0.67 \\ -0.67 & 1.33 \end{pmatrix}
$$

**GLS 행렬 연산** (계산 생략):

$$
\sum_{i=1}^{2} \mathbf{X}_i^T \mathbf{V}_i^{-1} \mathbf{X}_i = \begin{pmatrix} 2.67 & 0.67 \\ 0.67 & 2.67 \end{pmatrix}
$$

$$
\hat{\boldsymbol{\beta}}_{GLS} = \begin{pmatrix} 2.67 & 0.67 \\ 0.67 & 2.67 \end{pmatrix}^{-1} \left(\sum_{i=1}^{2} \mathbf{X}_i^T \mathbf{V}_i^{-1} \mathbf{Y}_i\right)
$$

이는 측정 간 상관성을 고려한 최적 추정값을 제공합니다.

***

### 6. $$\boldsymbol{\alpha}$$가 미지수인 경우: 반복 알고리즘

실무에서 분산 성분 $$\boldsymbol{\alpha}$$는 알려지지 않으므로, **다음의 반복 절차**를 거칩니다:[1]

#### 알고리즘: 시행적 GLS (Iterative GLS)

**초기화**: 초기 추정량 $$\hat{\boldsymbol{\alpha}}^{(0)}$$ 설정 (예: 표본 분산으로부터)

**반복 $$k = 0, 1, 2, \ldots$$**:

1. **Step 1**: 현재 분산 추정량 $$\hat{\boldsymbol{\alpha}}^{(k)}$$를 이용해 $$\hat{\mathbf{V}}_i^{(k)}$$ 구성

2. **Step 2**: GLS를 적용하여 고정효과 추정:

$$
\hat{\boldsymbol{\beta}}^{(k+1)} = \left(\sum_{i=1}^{N} \mathbf{X}_i^T (\hat{\mathbf{V}}_i^{(k)})^{-1} \mathbf{X}_i\right)^{-1} \sum_{i=1}^{N} \mathbf{X}_i^T (\hat{\mathbf{V}}_i^{(k)})^{-1} \mathbf{Y}_i
$$

3. **Step 3**: 업데이트된 고정효과를 이용해 잔차 계산:

$$
\hat{\boldsymbol{\epsilon}}_i^{(k+1)} = \mathbf{Y}_i - \mathbf{X}_i \hat{\boldsymbol{\beta}}^{(k+1)}
$$

4. **Step 4**: 분산 성분 재추정:

$$
\hat{\boldsymbol{\alpha}}^{(k+1)} = \text{update}(\hat{\boldsymbol{\alpha}}^{(k)}, \hat{\boldsymbol{\epsilon}}^{(k+1)})
$$

(구체적인 업데이트 규칙은 MLE, REML 등에 따라 다름)

5. **수렴 판정**: $$\|\hat{\boldsymbol{\beta}}^{(k+1)} - \hat{\boldsymbol{\beta}}^{(k)}\| < \epsilon$$ 또는 $$\|\hat{\boldsymbol{\alpha}}^{(k+1)} - \hat{\boldsymbol{\alpha}}^{(k)}\| < \epsilon$$이면 수렴으로 판정, 중단. 아니면 Step 1로 돌아감.

**최종 추정량**:

$$
\hat{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}}^{(K+1)}, \quad \hat{\boldsymbol{\alpha}} = \hat{\boldsymbol{\alpha}}^{(K+1)}
$$

(여기서 $$K$$는 수렴 반복 횟수)

***

### 7. 의미 해석

#### 가중 최소제곱의 관점

GLS는 **가중 최소제곱 (Weighted Least Squares)**으로 해석할 수 있습니다:

- **측정 정밀도가 높은** 관측값 ($$\mathbf{V}_i$$의 분산이 작음): 가중치 $$w_{ij}$$ 크다 → 추정에 더 많은 영향
- **측정 정밀도가 낮은** 관측값 ($$\mathbf{V}_i$$의 분산이 큼): 가중치 $$w_{ij}$$ 작다 → 추정에 적은 영향

#### 개체 내 상관성 반영

$$\mathbf{V}_i$$가 대각 행렬이 아니면 (즉, $$\rho \neq 0$$이면):

- 같은 개체 내 반복측정들 간의 상관성이 자동으로 모델에 반영됨
- OLS처럼 독립성을 가정하지 않으므로 더 효율적인 추정 가능

***

### 요약

| 단계 | 식 | 의미 |
|------|-----|------|
| **정규방정식** | $$\sum_i \mathbf{X}_i^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i\boldsymbol{\beta}) = \mathbf{0}$$ | 우도함수의 1차 미분 = 0 |
| **GLS 추정량** | $$\hat{\boldsymbol{\beta}} = (\sum_i \mathbf{X}_i^T \mathbf{V}_i^{-1}\mathbf{X}_i)^{-1} \sum_i \mathbf{X}_i^T \mathbf{V}_i^{-1}\mathbf{Y}_i$$ | 고정효과 MLE |
| **분산** | $$\text{Var}[\hat{\boldsymbol{\beta}}] = (\sum_i \mathbf{X}_i^T \mathbf{V}_i^{-1}\mathbf{X}_i)^{-1}$$ | 신뢰도 평가 |
| **반복 알고리즘** | $$\hat{\boldsymbol{\beta}}^{(k+1)} = f(\hat{\boldsymbol{\alpha}}^{(k)})$$ | $$\boldsymbol{\alpha}$$ 미지수시 사용 |
