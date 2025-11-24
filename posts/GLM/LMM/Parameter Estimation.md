---
title: "Linear Mixed Model - Parameter Estimation"
date: 2025-11-24       
description: "20251124 GLM class notes"
categories: [lecture, glm, lmm, LinearMixedModel, ParameterEstimation]
author: "김한울"
---

## Parameter Estimation 섹션 상세 설명

이 섹션은 선형혼합모형에서 **고정효과**와 **분산 성분**을 추정하는 두 가지 주요 방법—**최대우도추정(MLE)**과 **제한최대우도추정(REML)**—을 상세히 다룹니다. 핵심은 왜 REML이 필요한가에 있습니다.

***

## 1. 최대우도추정(Maximum Likelihood Estimation, MLE)

### 1.1 기본 개념

고정효과 $$\boldsymbol{\beta}$$와 분산 성분 $$\boldsymbol{\alpha}$$를 동시에 추정하는 방법입니다.

#### 전체 우도함수 (전개)

선형혼합모형의 전체 우도함수는:

$$
L_{ML}(\boldsymbol{\theta}) = L_{ML}(\boldsymbol{\beta}, \boldsymbol{\alpha}) = \prod_{i=1}^{N} (2\pi)^{-n_i/2} |\mathbf{V}_i(\boldsymbol{\alpha})|^{-1/2} \exp\left\{-\frac{1}{2}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i(\boldsymbol{\alpha})^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})\right\}
$$

로그 우도:

$$
\ell_{ML}(\boldsymbol{\beta}, \boldsymbol{\alpha}) = -\frac{1}{2}\sum_{i=1}^{N}\left[n_i \log(2\pi) + \log|\mathbf{V}_i| + (\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \boldsymbol{\beta})\right]
$$

### 1.2 추정 절차

#### 방법 1: 프로파일 우도 (Profile Likelihood)

**Step 1**: 먼저 $$\boldsymbol{\beta}$$를 고정하고, 주어진 $$\boldsymbol{\alpha}$$ 값에서 GLS로 $$\boldsymbol{\beta}$$ 추정:

$$
\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}) = \left(\sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}(\boldsymbol{\alpha}) \mathbf{X}_i\right)^{-1} \sum_{i=1}^{N} \mathbf{X}_i^T \mathbf{V}_i^{-1}(\boldsymbol{\alpha}) \mathbf{Y}_i
$$

**Step 2**: 프로파일 우도함수 구성 (estimated $$\boldsymbol{\beta}$$를 원래 우도에 대입):

$$
\ell_{ML}^{\text{prof}}(\boldsymbol{\alpha}) = \ell_{ML}(\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}), \boldsymbol{\alpha})
$$

자세히:

$$
\ell_{ML}^{\text{prof}}(\boldsymbol{\alpha}) = -\frac{1}{2}\sum_{i=1}^{N}\left[\log|\mathbf{V}_i| + (\mathbf{Y}_i - \mathbf{X}_i \hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}))^T \mathbf{V}_i^{-1}(\mathbf{Y}_i - \mathbf{X}_i \hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}))\right] + \text{const}
$$

**Step 3**: 프로파일 우도를 $$\boldsymbol{\alpha}$$에 대해 최대화:

$$
\frac{\partial \ell_{ML}^{\text{prof}}(\boldsymbol{\alpha})}{\partial \boldsymbol{\alpha}} = \mathbf{0}
$$

이를 풀어서 $$\hat{\boldsymbol{\alpha}}_{ML}$$ 얻음.

**Step 4**: 최종 추정량:

$$
\hat{\boldsymbol{\beta}}_{ML} = \hat{\boldsymbol{\beta}}(\hat{\boldsymbol{\alpha}}_{ML})
$$

#### 방법 2: 동시 최대화

$$\boldsymbol{\beta}$$와 $$\boldsymbol{\alpha}$$를 **동시에** 최대화할 수도 있습니다:

$$
\frac{\partial \ell_{ML}(\boldsymbol{\beta}, \boldsymbol{\alpha})}{\partial \boldsymbol{\beta}} = \mathbf{0}, \quad \frac{\partial \ell_{ML}(\boldsymbol{\beta}, \boldsymbol{\alpha})}{\partial \boldsymbol{\alpha}} = \mathbf{0}
$$

방법 1과 방법 2는 동등한 결과를 줍니다 (확인: Newton-Raphson 알고리즘에서).

### 1.3 MLE의 문제: 편향성

MLE는 **분산 성분을 과소 추정(biased downward)**하는 경향이 있습니다.

#### 간단한 예: 정규분포에서 분산 추정

$$N$$개 독립 관측치 $$Y_1, \ldots, Y_N \sim N(\mu, \sigma^2)$$를 생각합시다.

**경우 1**: $$\mu$$가 알려진 경우

$$
\hat{\sigma}^2_{MLE,\mu\text{known}} = \frac{1}{N}\sum_{i=1}^{N}(Y_i - \mu)^2
$$

$$E[\hat{\sigma}^2_{MLE,\mu\text{known}}] = \sigma^2$$ ✓ 불편

**경우 2**: $$\mu$$가 미지수인 경우

$$
\hat{\sigma}^2_{MLE,\mu\text{unknown}} = \frac{1}{N}\sum_{i=1}^{N}(Y_i - \bar{Y})^2
$$

기댓값 계산:

$$
E[\hat{\sigma}^2_{MLE,\mu\text{unknown}}] = E\left[\frac{1}{N}\sum_{i=1}^{N}(Y_i - \bar{Y})^2\right]
$$

$$\sum_{i=1}^{N}(Y_i - \bar{Y})^2 \sim \sigma^2 \chi^2_{N-1}$$ (자유도 $$N-1$$) 이므로:

$$
E[\hat{\sigma}^2_{MLE,\mu\text{unknown}}] = \frac{1}{N} \cdot (N-1)\sigma^2 = \frac{N-1}{N}\sigma^2
$$

따라서:

$$
E[\hat{\sigma}^2_{MLE,\mu\text{unknown}}] = \left(1 - \frac{1}{N}\right)\sigma^2 < \sigma^2
$$

**결론**: 평균 $$\mu$$를 추정해야 하기 때문에 분산 추정량이 편향됩니다. 이것이 MLE의 근본적인 문제입니다.

***

## 2. 제한최대우도추정(Restricted Maximum Likelihood Estimation, REML)

REML은 **고정효과 추정으로 인한 자유도 손실을 보정**하여 분산 성분의 불편 추정량을 제공합니다.

### 2.1 기본 아이디어: 오차 대비(Error Contrasts)

#### 핵심 질문

> "고정효과 $$\boldsymbol{\beta}$$를 추정하지 않으면서 분산 성분 $$\boldsymbol{\alpha}$$를 추정할 수 있을까?"

**답**: 데이터를 변환하여 고정효과를 제거합니다.

#### 변환 행렬 $$\mathbf{A}$$ 구성

$$\mathbf{A}$$를 다음을 만족하는 $$n \times (n-p)$$ 행렬로 정의합니다:

$$
\mathbf{A}^T \mathbf{A} = \mathbf{I}_{n-p}, \quad \mathbf{A}^T \mathbf{X} = \mathbf{0}
$$

여기서:
- $$\mathbf{I}_{n-p}$$: $$(n-p) \times (n-p)$$ 단위행렬
- $$\mathbf{X}$$: 고정효과 설계행렬 ($$n \times p$$)

**기하학적 의미**: $$\mathbf{A}$$는 $$\mathbf{X}$$의 열 공간(column space)에 **직교**하는 $$n-p$$개의 정규직교 벡터로 이루어진 행렬입니다.

#### 투영(Projection) 해석

투영 행렬을 다음과 같이 정의:

$$
\mathbf{P} = \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T
$$

($$\mathbf{X}$$의 열 공간으로의 정사영)

보정 행렬:

$$
\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T = \mathbf{I} - \mathbf{P}
$$

($$\mathbf{X}$$의 직교 여공간으로의 정사영)

**성질**: $$\mathbf{M}$$은 대칭이고 멱등(idempotent)입니다:
- $$\mathbf{M}^T = \mathbf{M}$$ (대칭)
- $$\mathbf{M}^2 = \mathbf{M}$$ (멱등)

$$\mathbf{M}$$의 고유값(eigenvalue)은 $$0$$ (중복도 $$p$$)과 $$1$$ (중복도 $$n-p$$)입니다.

#### 데이터 변환

변환된 데이터:

$$
\mathbf{U} = \mathbf{A}^T \mathbf{Y}
$$

차원: $$\mathbf{U}$$는 $$(n-p) \times 1$$ 벡터입니다.

**중요 성질**:

$$
E[\mathbf{U}] = \mathbf{A}^T E[\mathbf{Y}] = \mathbf{A}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{0}
$$

($$\mathbf{A}^T \mathbf{X} = \mathbf{0}$$이므로)

따라서 고정효과 $$\boldsymbol{\beta}$$가 **제거**됩니다!

### 2.2 REML 우도함수

#### 변환 데이터의 분포

$$
\mathbf{U} = \mathbf{A}^T \mathbf{Y} \sim N(\mathbf{0}, \mathbf{A}^T \mathbf{V} \mathbf{A})
$$

**REML 우도함수**:

$$
L_{REML}(\boldsymbol{\alpha}) = (2\pi)^{-(n-p)/2} |\mathbf{A}^T \mathbf{V} \mathbf{A}|^{-1/2} \exp\left\{-\frac{1}{2}\mathbf{U}^T (\mathbf{A}^T \mathbf{V} \mathbf{A})^{-1} \mathbf{U}\right\}
$$

로그 REML:

$$
\ell_{REML}(\boldsymbol{\alpha}) = -\frac{1}{2}\left[(n-p)\log(2\pi) + \log|\mathbf{A}^T \mathbf{V} \mathbf{A}| + \mathbf{U}^T (\mathbf{A}^T \mathbf{V} \mathbf{A})^{-1} \mathbf{U}\right]
$$

#### 대안적 표현 (더 사용하기 편함)

다음과 같이도 표현됩니다:

$$
\ell_{REML}(\boldsymbol{\alpha}) = -\frac{1}{2}\left[\log|\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}| + \log|\mathbf{V}| + (\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}})^T \mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}})\right] + \text{const}
$$

여기서 $$\hat{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})$$는 GLS 추정량입니다.

**해석**:
- $$\log|\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}|$$: 정보 행렬의 행렬식 (고정효과 정보)
- $$\log|\mathbf{V}|$$: 전체 공분산의 행렬식
- 마지막 항: 잔차 제곱합

### 2.3 REML 추정 절차

#### 알고리즘

**Step 1**: 초기값 $$\hat{\boldsymbol{\alpha}}^{(0)}$$ 설정

**Step 2**: 반복 $$k = 0, 1, 2, \ldots$$에서:

1. 현재 추정량 $$\hat{\boldsymbol{\alpha}}^{(k)}$$로 GLS 수행:

$$
\hat{\boldsymbol{\beta}}^{(k)} = \left(\sum_i \mathbf{X}_i^T (\hat{\mathbf{V}}_i^{(k)})^{-1} \mathbf{X}_i\right)^{-1} \sum_i \mathbf{X}_i^T (\hat{\mathbf{V}}_i^{(k)})^{-1} \mathbf{Y}_i
$$

2. 잔차 계산:

$$
\hat{\boldsymbol{\epsilon}}_i^{(k)} = \mathbf{Y}_i - \mathbf{X}_i \hat{\boldsymbol{\beta}}^{(k)}
$$

3. REML 점수(score) 계산:

$$
\mathbf{s}_{REML}^{(k)} = \frac{\partial \ell_{REML}(\boldsymbol{\alpha})}{\partial \boldsymbol{\alpha}}\Big|_{\boldsymbol{\alpha}=\hat{\boldsymbol{\alpha}}^{(k)}}
$$

4. 뉴턴-랩슨(Newton-Raphson) 업데이트:

$$
\hat{\boldsymbol{\alpha}}^{(k+1)} = \hat{\boldsymbol{\alpha}}^{(k)} + (\mathbf{H}^{(k)})^{-1} \mathbf{s}_{REML}^{(k)}
$$

여기서 $$\mathbf{H}^{(k)}$$는 Hessian 행렬 (2차 미분).

**Step 3**: 수렴 판정: $$\|\hat{\boldsymbol{\alpha}}^{(k+1)} - \hat{\boldsymbol{\alpha}}^{(k)}\| < \epsilon$$이면 종료.

#### 최종 추정량

$$
\hat{\boldsymbol{\alpha}}_{REML}, \quad \hat{\boldsymbol{\beta}}_{REML} = \hat{\boldsymbol{\beta}}(\hat{\boldsymbol{\alpha}}_{REML})
$$

### 2.4 간단한 예: REML이 편향을 어떻게 해결하는가

#### 예제 설정

$$N$$개 관측치 $$Y_1, \ldots, Y_N \sim N(\mu, \sigma^2)$$, 회귀 모형:

$$
Y_i = \mu + \epsilon_i, \quad \epsilon_i \sim N(0, \sigma^2)
$$

설계행렬: $$\mathbf{X} = \mathbf{1}_N$$ ($$N \times 1$$ ones 벡터)

#### MLE 추정량 (복습)

$$
\hat{\mu}_{MLE} = \bar{Y}, \quad \hat{\sigma}^2_{MLE} = \frac{1}{N}\sum_{i=1}^{N}(Y_i - \bar{Y})^2
$$

$$
E[\hat{\sigma}^2_{MLE}] = \frac{N-1}{N}\sigma^2 < \sigma^2 \quad \text{(편향)}
$$

#### REML 추정량

변환 행렬 $$\mathbf{A}$$는 다음을 만족:

$$
\mathbf{A}^T \mathbf{1}_N = \mathbf{0} \quad \text{(직교 조건)}
$$

예를 들어, $$\mathbf{A}^T$$의 행들은 "대비(contrast)":

$$
\mathbf{A}^T \mathbf{Y} = \begin{pmatrix} Y_1 - \bar{Y} \\ Y_2 - \bar{Y} \\ \vdots \\ Y_{N-1} - \bar{Y} \end{pmatrix}
$$

(독립적인 $$N-1$$개 대비)

REML 우도로부터:

$$
\hat{\sigma}^2_{REML} = \frac{1}{N-1}\sum_{i=1}^{N}(Y_i - \bar{Y})^2
$$

기댓값:

$$
E[\hat{\sigma}^2_{REML}] = E\left[\frac{1}{N-1}\sum_{i=1}^{N}(Y_i - \bar{Y})^2\right] = \frac{1}{N-1} \cdot (N-1)\sigma^2 = \sigma^2
$$

**결론**: REML은 $$N$$의 자리에 $$N-1$$을 사용함으로써 **불편 추정량** 제공. 이는 고정효과 추정으로 소비된 $$1$$개 자유도를 반영합니다!

### 2.5 회귀 모형에서의 일반화

#### 일반 선형 회귀

$$
\mathbf{Y} \sim N(\mathbf{X}\boldsymbol{\beta}, \sigma^2\mathbf{I})
$$

**MLE의 분산 추정량** (점근적 불편성만 만족):

$$
\hat{\sigma}^2_{MLE} = \frac{1}{n}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2 = \frac{1}{n}\text{RSS}
$$

$$\text{RSS} \sim \sigma^2 \chi^2_{n-p}$$이므로:

$$
E[\hat{\sigma}^2_{MLE}] = \frac{n-p}{n}\sigma^2 < \sigma^2 \quad \text{(편향)}
$$

**표준 불편 추정량** (교과서의 MSE):

$$
\hat{\sigma}^2_{unbiased} = \frac{1}{n-p}\text{RSS} = \frac{1}{n-p}\sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2
$$

이는 **REML에서 나오는 추정량과 동일합니다**!

### 2.6 REML의 중요한 성질들

#### 성질 1: 변수 변환 불변성(Invariance to Linear Reparameterization)

고정효과를 $$\boldsymbol{\beta} \to \mathbf{C}\boldsymbol{\beta}$$로 재매개변수화해도 REML 추정량은 **불변**입니다.

**증명 스케치**:

변환 행렬 $$\mathbf{A}$$는 $$\mathbf{X}$$의 직교 여공간에 있으므로, $$\mathbf{X}$$의 선형 결합은 $$\mathbf{A}^T$$ 변환에 의해 제거됩니다.

#### 성질 2: MLE와의 관계

$$
L_{REML}(\boldsymbol{\alpha}) \propto \frac{L_{ML}(\boldsymbol{\alpha}, \hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}))}{|\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}|^{1/2}}
$$

분자: 프로파일 MLE  
분모: 고정효과 정보의 행렬식의 제곱근 (Fisher 정보)

**의미**: REML은 프로파일 MLE에서 고정효과 정보를 제거합니다.

***

## 3. 제한(Restricted) 우도함수

### 정의

REML 추정량 $$\hat{\boldsymbol{\alpha}}_{REML}$$, $$\hat{\boldsymbol{\beta}}_{REML}$$는 다음을 동시에 최대화하여 얻을 수 있습니다:

$$
L_{REL}(\boldsymbol{\theta}) = L_{REL}(\boldsymbol{\beta}, \boldsymbol{\alpha})
$$

이를 **제한 우도함수(Restricted Likelihood Function)** 또는 **합동 우도함수(Joint Likelihood)**라 합니다.

**명시적 형태**:

$$
\ell_{REL}(\boldsymbol{\beta}, \boldsymbol{\alpha}) \propto -\frac{1}{2}\left[\log|\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}| + \log|\mathbf{V}| + (\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T \mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})\right]
$$

정규 방정식:

$$
\frac{\partial \ell_{REL}}{\partial \boldsymbol{\beta}} = -\mathbf{X}^T \mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) = \mathbf{0} \quad \Rightarrow \quad \hat{\boldsymbol{\beta}} = \hat{\boldsymbol{\beta}}(\boldsymbol{\alpha})
$$

$$
\frac{\partial \ell_{REL}}{\partial \boldsymbol{\alpha}} = \mathbf{0} \quad \Rightarrow \quad \hat{\boldsymbol{\alpha}}
$$

### 명칭 주의

엄밀히 말하면 $$L_{REL}$$은 확률 우도함수(likelihood)가 **아닙니다** (왜냐하면 $$\boldsymbol{\beta}$$의 완전한 확률 모형이 아니기 때문). 그러나 관례상 "**제한 우도함수**"라 부릅니다.

***

## 4. MLE vs REML: 비교 요약

| 측면 | MLE | REML |
|------|-----|------|
| **고정효과 추정** | GLS (프로파일에서) | GLS (동등) |
| **분산 성분 추정** | $$\frac{1}{n}\text{RSS}$$ (편향) | $$\frac{1}{n-p}\text{RSS}$$ (불편) |
| **편향** | $$E[\hat{\sigma}^2_{MLE}] = \frac{n-p}{n}\sigma^2$$ | $$E[\hat{\sigma}^2_{REML}] = \sigma^2$$ |
| **정보 사용** | 전체 우도 사용 | 고정효과 정보 제거 |
| **모형 비교** | $$L_{ML}$$ (계층 모형 비교 가능) | $$L_{REML}$$ (같은 고정효과만) |
| **권장 사항** | 모형 비교 (고정효과 다를 때) | 분산 성분 추정, 신뢰구간 |

***

## 5. 실제 적용

### 5.1 언제 어느 것을 사용할 것인가?

**REML 사용**:
- 분산 성분의 신뢰도 있는 추정이 주된 목표
- 표준 오차, 신뢰구간 계산
- 모수적 검정 (고정효과는 동일하고 분산만 비교)

**MLE 사용**:
- 중첩 모형 비교 (가능도비 검정, likelihood ratio test)
- 고정효과가 다른 모형 비교 (예: $$H_0: \boldsymbol{\beta} = \mathbf{0}$$ 검정)

### 5.2 소프트웨어 구현

대부분의 통계 소프트웨어(R의 `lme4`, SAS의 `PROC MIXED`, SPSS 등)는:

- **기본값**: REML 사용
- **옵션**: `method="ML"`로 지정하여 MLE 사용 가능

***

## 요약

**Parameter Estimation 섹션의 핵심 메시지**:

1. **MLE**는 고정효과와 분산 성분을 동시에 추정하지만, 분산 성분을 **편향되게** 추정합니다. 이는 고정효과 추정으로 인한 자유도 손실 때문입니다.

2. **REML**은 데이터를 변환하여 고정효과를 제거한 후, 변환 데이터에서 분산 성분을 추정함으로써 **불편 추정량**을 제공합니다.

3. REML의 수학적 기초는 **오차 대비(error contrasts)**: 고정효과 설계행렬 $$\mathbf{X}$$에 직교하는 방향으로 데이터를 투영합니다.

4. 실무에서는 **REML이 기본**이며, MLE는 모형 비교(모수 선택) 때 사용됩니다.

5. 간단한 정규 샘플의 예에서: $$\frac{1}{N}(MLE)$$ vs $$\frac{1}{N-1}(REML)$$ 공식이 정확히 나타나며, 이는 일반적인 결과의 특수 경우입니다.
