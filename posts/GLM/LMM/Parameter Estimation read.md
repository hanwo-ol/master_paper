---
title: "Linear Mixed Model - Preliminary Model Further Read"
date: 2025-11-24       
description: "20251124 GLM class notes 3"
categories: [lecture, glm, lmm, LinearMixedModel, PreliminaryModel]
author: "김한울"
---

## REML에 대한 심층 고찰 및 수식 전개

### 목차

1. REML의 여러 유도 경로
2. Bayesian 관점에서의 REML
3. Fisher 정보 행렬과 점근 성질
4. 변환 불변성(Transformation Invariance)
5. REML vs ML: 편향 보정의 수학적 구조
6. 일반화된 REML (GLMMs로의 확장)

***

## 1. REML의 여러 유도 경로

REML은 **여러 관점에서 유도**할 수 있으며, 각 관점은 동일한 추정량을 제공합니다.[1][2][3]

### 1.1 경로 1: 선형 변환과 오차 대비 (Error Contrasts)

이미 앞에서 다룬 **기본 경로**입니다.

#### 핵심 아이디어

고정효과 $$\boldsymbol{\beta}$$에 직교하는 선형 변환 $$\mathbf{K}$$를 구성:

$$
\mathbf{K}^T \mathbf{X} = \mathbf{0}, \quad \mathbf{K}^T \mathbf{K} = \mathbf{I}_{n-p}
$$

변환 데이터:

$$
\mathbf{U} = \mathbf{K}^T \mathbf{Y}
$$

$$
E[\mathbf{U}] = \mathbf{K}^T \mathbf{X} \boldsymbol{\beta} = \mathbf{0}
$$

$$
\text{Var}[\mathbf{U}] = \mathbf{K}^T \mathbf{V} \mathbf{K}
$$

REML 우도:

$$
L_{REML}(\boldsymbol{\alpha}) = (2\pi)^{-(n-p)/2} |\mathbf{K}^T \mathbf{V} \mathbf{K}|^{-1/2} \exp\left\{-\frac{1}{2}\mathbf{U}^T (\mathbf{K}^T \mathbf{V} \mathbf{K})^{-1} \mathbf{U}\right\}
$$

### 1.2 경로 2: 마진화(Marginalization) - 베이지안 해석

#### 고정효과에 대한 사전분포 설정

고정효과 $$\boldsymbol{\beta}$$에 대해 **균일 비정보적 사전분포(uniform improper prior)**를 부여:

$$
p(\boldsymbol{\beta}) \propto 1
$$

#### 주변화(Integrating out $$\boldsymbol{\beta}$$)

전체 우도:

$$
p(\mathbf{Y} | \boldsymbol{\beta}, \boldsymbol{\alpha}) = (2\pi)^{-n/2} |\mathbf{V}|^{-1/2} \exp\left\{-\frac{1}{2}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T \mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})\right\}
$$

$$\boldsymbol{\beta}$$에 대해 적분:

$$
p(\mathbf{Y} | \boldsymbol{\alpha}) = \int p(\mathbf{Y} | \boldsymbol{\beta}, \boldsymbol{\alpha}) p(\boldsymbol{\beta}) \, d\boldsymbol{\beta}
$$

#### 적분 계산 (Gaussian 적분 공식)

일반 Gaussian 적분 공식:

$$
\int \exp\left\{-\frac{1}{2}(\mathbf{x} - \boldsymbol{\mu})^T \mathbf{A} (\mathbf{x} - \boldsymbol{\mu})\right\} d\mathbf{x} = (2\pi)^{k/2} |\mathbf{A}|^{-1/2}
$$

우도 함수를 $$\boldsymbol{\beta}$$에 대한 이차형식으로 정리:

$$
(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T \mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta}) = (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^T (\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}) (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}) + \mathbf{r}^T \mathbf{V}^{-1} \mathbf{r}
$$

여기서:
- $$\hat{\boldsymbol{\beta}} = (\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X})^{-1} \mathbf{X}^T \mathbf{V}^{-1} \mathbf{Y}$$: GLS 추정량
- $$\mathbf{r} = \mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}$$: 잔차

**증명** (이차형식 완성):

$$
(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})^T \mathbf{V}^{-1}(\mathbf{Y} - \mathbf{X}\boldsymbol{\beta})
$$

$$
= \mathbf{Y}^T \mathbf{V}^{-1} \mathbf{Y} - 2\boldsymbol{\beta}^T \mathbf{X}^T \mathbf{V}^{-1} \mathbf{Y} + \boldsymbol{\beta}^T \mathbf{X}^T \mathbf{V}^{-1} \mathbf{X} \boldsymbol{\beta}
$$

$$\mathbf{H} = \mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}$$로 표기하고, 이차형식 완성:

$$
= (\boldsymbol{\beta} - \mathbf{H}^{-1}\mathbf{X}^T \mathbf{V}^{-1}\mathbf{Y})^T \mathbf{H} (\boldsymbol{\beta} - \mathbf{H}^{-1}\mathbf{X}^T \mathbf{V}^{-1}\mathbf{Y}) + \mathbf{Y}^T \mathbf{V}^{-1}\mathbf{Y} - \mathbf{Y}^T \mathbf{V}^{-1}\mathbf{X} \mathbf{H}^{-1} \mathbf{X}^T \mathbf{V}^{-1}\mathbf{Y}
$$

$$
= (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^T \mathbf{H} (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}}) + \mathbf{Y}^T (\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}\mathbf{H}^{-1}\mathbf{X}^T\mathbf{V}^{-1})\mathbf{Y}
$$

마지막 항은 투영 행렬을 이용하여 다음과 같이 쓸 수 있습니다:

$$
\mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1} = \mathbf{V}^{-1}\mathbf{M}
$$

여기서 $$\mathbf{M} = \mathbf{I} - \mathbf{X}(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}$$는 잔차 형성 행렬(residual-forming matrix)입니다.

따라서:

$$
\mathbf{r}^T \mathbf{V}^{-1} \mathbf{r} = \mathbf{Y}^T \mathbf{V}^{-1}\mathbf{M}\mathbf{Y}
$$

#### 적분 실행

$$
p(\mathbf{Y} | \boldsymbol{\alpha}) = (2\pi)^{-n/2} |\mathbf{V}|^{-1/2} \exp\left\{-\frac{1}{2}\mathbf{r}^T \mathbf{V}^{-1} \mathbf{r}\right\} \int \exp\left\{-\frac{1}{2}(\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^T \mathbf{H} (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})\right\} d\boldsymbol{\beta}
$$

적분 부분:

$$
\int \exp\left\{-\frac{1}{2}(\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})^T \mathbf{H} (\boldsymbol{\beta} - \hat{\boldsymbol{\beta}})\right\} d\boldsymbol{\beta} = (2\pi)^{p/2} |\mathbf{H}|^{-1/2}
$$

최종 REML 우도:

$$
p(\mathbf{Y} | \boldsymbol{\alpha}) = (2\pi)^{-(n-p)/2} |\mathbf{V}|^{-1/2} |\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}|^{-1/2} \exp\left\{-\frac{1}{2}\mathbf{r}^T \mathbf{V}^{-1} \mathbf{r}\right\}
$$

로그 REML:

$$
\ell_{REML}(\boldsymbol{\alpha}) = -\frac{1}{2}\left[(n-p)\log(2\pi) + \log|\mathbf{V}| + \log|\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}| + \mathbf{r}^T \mathbf{V}^{-1} \mathbf{r}\right]
$$

**결론**: REML은 고정효과에 대한 균일 사전분포를 가정한 **주변 우도(marginal likelihood)**입니다.[4][5]

### 1.3 경로 3: 조건부 우도 (Conditional Likelihood)

#### Sufficient Statistic 관점

$$\mathbf{t} = \mathbf{X}^T \mathbf{V}^{-1} \mathbf{Y}$$는 고정효과 $$\boldsymbol{\beta}$$에 대한 **충분통계량(sufficient statistic)**입니다.[6]

REML은 충분통계량 $$\mathbf{t}$$가 주어졌을 때 $$\mathbf{Y}$$의 **조건부 우도(conditional likelihood)**로 해석됩니다:

$$
L_{REML}(\boldsymbol{\alpha}) = p(\mathbf{Y} | \mathbf{t}, \boldsymbol{\alpha})
$$

#### 조건부 분포 유도

$$
p(\mathbf{Y} | \mathbf{t}, \boldsymbol{\alpha}) = \frac{p(\mathbf{Y}, \mathbf{t} | \boldsymbol{\alpha})}{p(\mathbf{t} | \boldsymbol{\alpha})} = \frac{p(\mathbf{Y} | \boldsymbol{\alpha})}{p(\mathbf{t} | \boldsymbol{\alpha})}
$$

($$\mathbf{t}$$는 $$\mathbf{Y}$$의 함수이므로 $$p(\mathbf{Y}, \mathbf{t}) = p(\mathbf{Y})$$)

충분통계량 $$\mathbf{t}$$의 분포:

$$
\mathbf{t} = \mathbf{X}^T \mathbf{V}^{-1} \mathbf{Y} \sim N(\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X} \boldsymbol{\beta}, \mathbf{X}^T \mathbf{V}^{-1} \mathbf{X})
$$

조건부 우도는 경로 2의 주변 우도와 **동일**합니다.[6]

***

## 2. Bayesian 관점에서의 REML

### 2.1 Harville (1974)의 연결

Harville (1974)은 REML이 **특정 사전분포를 가정한 Bayesian 추론**과 동등함을 보였습니다.[5][4]

#### 사전분포 설정

- **고정효과**: $$p(\boldsymbol{\beta}) \propto 1$$ (균일 비정보적)
- **분산 성분**: $$p(\boldsymbol{\alpha}) \propto 1$$ (균일 비정보적)

#### 사후분포 모드(Posterior Mode)

사후분포:

$$
p(\boldsymbol{\beta}, \boldsymbol{\alpha} | \mathbf{Y}) \propto p(\mathbf{Y} | \boldsymbol{\beta}, \boldsymbol{\alpha}) p(\boldsymbol{\beta}) p(\boldsymbol{\alpha})
$$

$$\boldsymbol{\beta}$$를 적분으로 제거:

$$
p(\boldsymbol{\alpha} | \mathbf{Y}) = \int p(\boldsymbol{\beta}, \boldsymbol{\alpha} | \mathbf{Y}) d\boldsymbol{\beta} \propto p(\mathbf{Y} | \boldsymbol{\alpha})
$$

이는 **REML 우도**와 동일합니다.

따라서:

$$
\hat{\boldsymbol{\alpha}}_{REML} = \arg\max_{\boldsymbol{\alpha}} p(\boldsymbol{\alpha} | \mathbf{Y})
$$

REML 추정량은 분산 성분의 **사후 모드(posterior mode) 또는 MAP (Maximum A Posteriori) 추정량**입니다.[5]

### 2.2 Laplace 근사와의 연결

REML 우도는 전체 우도에 대한 **Laplace 근사(Laplace approximation)**로도 해석됩니다.[5]

Laplace 근사는 적분을 최빈값(mode) 주변의 이차 근사로 계산하는 방법입니다:

$$
\int e^{f(x)} dx \approx e^{f(x^*)} \sqrt{\frac{2\pi}{-f''(x^*)}}
$$

여기서 $$x^*$$는 $$f(x)$$의 최빈값입니다.

REML 우도의 $$\log|\mathbf{X}^T \mathbf{V}^{-1} \mathbf{X}|^{-1/2}$$ 항이 정확히 Laplace 근사의 **곡률 보정 항(curvature correction)**에 해당합니다.

***

## 3. Fisher 정보 행렬과 점근 성질

### 3.1 Fisher 정보 행렬 (Fisher Information Matrix, FIM)

#### 정의

Fisher 정보 행렬은 로그 우도의 **2차 미분의 기댓값의 음수**입니다:

$$
\mathcal{I}(\boldsymbol{\theta}) = -E\left[\frac{\partial^2 \ell(\boldsymbol{\theta})}{\partial \boldsymbol{\theta} \partial \boldsymbol{\theta}^T}\right]
$$

또는 점수 함수(score function)의 외적:

$$
\mathcal{I}(\boldsymbol{\theta}) = E\left[\frac{\partial \ell(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}} \frac{\partial \ell(\boldsymbol{\theta})}{\partial \boldsymbol{\theta}}^T\right]
$$

#### REML의 Fisher 정보 행렬

REML 점수 함수:

$$
\mathbf{s}_{REML}(\boldsymbol{\alpha}) = \frac{\partial \ell_{REML}(\boldsymbol{\alpha})}{\partial \boldsymbol{\alpha}}
$$

각 분산 성분 $$\alpha_k$$에 대해:

$$
\frac{\partial \ell_{REML}}{\partial \alpha_k} = -\frac{1}{2}\text{tr}\left[\mathbf{P} \frac{\partial \mathbf{V}}{\partial \alpha_k}\right] + \frac{1}{2}\mathbf{r}^T \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \alpha_k} \mathbf{V}^{-1} \mathbf{r}
$$

여기서 $$\mathbf{P}$$는 **조정 투영 행렬(adjusted projection matrix)**:

$$
\mathbf{P} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}
$$

**Fisher 정보 행렬** (REML):

$$
\mathcal{I}_{REML}(\boldsymbol{\alpha})_{k,\ell} = \frac{1}{2}\text{tr}\left[\mathbf{P} \frac{\partial \mathbf{V}}{\partial \alpha_k} \mathbf{P} \frac{\partial \mathbf{V}}{\partial \alpha_\ell}\right]
$$

**ML의 Fisher 정보 행렬** (비교):

$$
\mathcal{I}_{ML}(\boldsymbol{\alpha})_{k,\ell} = \frac{1}{2}\text{tr}\left[\mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \alpha_k} \mathbf{V}^{-1} \frac{\partial \mathbf{V}}{\partial \alpha_\ell}\right]
$$

**차이**: REML은 $$\mathbf{V}^{-1}$$ 대신 $$\mathbf{P}$$를 사용하여 **고정효과 추정의 불확실성**을 반영합니다.[7][8]

### 3.2 점근 성질 (Asymptotic Properties)

#### 일관성(Consistency)

모형이 다음을 만족하면 REML 추정량은 **일치 추정량(consistent estimator)**입니다:[9][10]

1. **점근적 식별가능성(Asymptotically Identifiable)**: 서로 다른 $$\boldsymbol{\alpha}$$ 값이 서로 다른 공분산 구조를 생성
2. **무한 정보량(Infinitely Informative)**: Fisher 정보 행렬이 $$n \to \infty$$일 때 무한대로 발산

$$
\hat{\boldsymbol{\alpha}}_{REML} \xrightarrow{p} \boldsymbol{\alpha}_0 \quad \text{as } n \to \infty
$$

#### 점근 정규성(Asymptotic Normality)

모형이 점근적으로 **비퇴화(non-degenerate)**이면 REML 추정량은 점근적으로 정규분포를 따릅니다:[11][10][9]

$$
\sqrt{n}(\hat{\boldsymbol{\alpha}}_{REML} - \boldsymbol{\alpha}_0) \xrightarrow{d} N(\mathbf{0}, \mathcal{I}_{REML}^{-1}(\boldsymbol{\alpha}_0))
$$

**공분산 행렬**:

$$
\text{Var}[\hat{\boldsymbol{\alpha}}_{REML}] \approx \mathcal{I}_{REML}^{-1}(\hat{\boldsymbol{\alpha}}_{REML})
$$

이는 **Cramér-Rao 하한(Cramér-Rao Lower Bound)**에 점근적으로 도달합니다.[12][11]

#### MLE와의 비교

- **MLE**: 점근적으로 불편이지만, 유한 표본에서 편향
- **REML**: 유한 표본에서 불편 (분산 성분), 점근적으로 MLE와 동등[10][9]

$$
\sqrt{n}(\hat{\boldsymbol{\alpha}}_{REML} - \hat{\boldsymbol{\alpha}}_{ML}) \xrightarrow{p} \mathbf{0}
$$

***

## 4. 변환 불변성 (Transformation Invariance)

### 4.1 성질

REML은 고정효과의 **선형 재매개변수화(linear reparameterization)**에 대해 불변입니다.[3]

#### 수학적 표현

고정효과를 다음과 같이 변환:

$$
\boldsymbol{\gamma} = \mathbf{C}\boldsymbol{\beta}, \quad \mathbf{C}: r \times p \text{ 풀랭크}
$$

새로운 설계 행렬:

$$
\mathbf{X}^* = \mathbf{X}\mathbf{C}^{-1}
$$

**주장**: REML 추정량 $$\hat{\boldsymbol{\alpha}}_{REML}$$는 $$\mathbf{C}$$의 선택에 무관합니다.

#### 증명 스케치

REML 우도는 잔차 $$\mathbf{r} = \mathbf{Y} - \mathbf{X}\hat{\boldsymbol{\beta}}$$에만 의존합니다.

$$\mathbf{r}$$는 $$\mathbf{X}$$의 열 공간으로의 투영의 **직교 여공간(orthogonal complement)**이므로:

$$
\mathbf{r} = (\mathbf{I} - \mathbf{P}_\mathbf{X})\mathbf{Y}
$$

여기서 $$\mathbf{P}_\mathbf{X} = \mathbf{X}(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}$$.

$$\mathbf{X}^*$$도 동일한 열 공간을 span하므로:

$$
\mathbf{P}_{\mathbf{X}^*} = \mathbf{P}_\mathbf{X}
$$

따라서 잔차와 REML 우도는 불변입니다. ✓

**실용적 의미**: 고정효과 모형의 매개변수화 방식(예: 제약 조건, 더미 코딩)이 분산 성분 추정에 영향을 주지 않습니다.

***

## 5. REML vs ML: 편향 보정의 수학적 구조

### 5.1 편향의 근원

#### 자유도 손실

ML은 고정효과 추정에 사용된 $$p$$개 자유도를 고려하지 않고, 전체 $$n$$개 관측값을 기준으로 분산을 추정합니다.

단순 선형 회귀 예제:

$$
\text{RSS} = \sum_{i=1}^{n}(Y_i - \hat{Y}_i)^2 \sim \sigma^2 \chi^2_{n-p}
$$

$$
\hat{\sigma}^2_{ML} = \frac{\text{RSS}}{n} \quad \Rightarrow \quad E[\hat{\sigma}^2_{ML}] = \frac{n-p}{n}\sigma^2
$$

**편향**:

$$
\text{Bias}[\hat{\sigma}^2_{ML}] = -\frac{p}{n}\sigma^2
$$

### 5.2 REML의 보정 메커니즘

REML은 $$n-p$$개의 독립적인 오차 대비를 사용하여 분산을 추정:

$$
\hat{\sigma}^2_{REML} = \frac{\text{RSS}}{n-p}
$$

$$
E[\hat{\sigma}^2_{REML}] = \sigma^2 \quad \text{(불편)}
$$

#### 일반화: 다변량 경우

분산 성분 $$\boldsymbol{\alpha}$$에 대해:

**MLE 점수 함수**:

$$
\mathbf{s}_{ML}(\boldsymbol{\alpha}) = -\frac{1}{2}\text{tr}\left[\mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \boldsymbol{\alpha}}\right] + \frac{1}{2}\mathbf{e}^T \mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \boldsymbol{\alpha}}\mathbf{V}^{-1}\mathbf{e}
$$

여기서 $$\mathbf{e} = \mathbf{Y} - \mathbf{X}\boldsymbol{\beta}$$.

**REML 점수 함수** (보정):

$$
\mathbf{s}_{REML}(\boldsymbol{\alpha}) = -\frac{1}{2}\text{tr}\left[\mathbf{P}\frac{\partial \mathbf{V}}{\partial \boldsymbol{\alpha}}\right] + \frac{1}{2}\mathbf{r}^T \mathbf{V}^{-1}\frac{\partial \mathbf{V}}{\partial \boldsymbol{\alpha}}\mathbf{V}^{-1}\mathbf{r}
$$

**차이**: 첫 번째 항에서 $$\mathbf{V}^{-1}$$ 대신 $$\mathbf{P} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}$$를 사용.

이는 **고정효과에 의한 차원 축소**를 정확히 반영합니다.[2][13]

### 5.3 자유도 보정의 행렬 표현

#### 유효 자유도 (Effective Degrees of Freedom)

REML의 로그 행렬식 항:

$$
\log|\mathbf{V}| + \log|\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X}|
$$

이는 다음과 같이 해석됩니다:

$$
\log|\mathbf{V}| + \log|\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X}| = \log|\mathbf{V}| + \log\left|\frac{\partial^2 \ell_{ML}}{\partial \boldsymbol{\beta} \partial \boldsymbol{\beta}^T}\right|
$$

두 번째 항은 고정효과에 대한 **Fisher 정보의 행렬식**으로, 고정효과 추정의 불확실성을 나타냅니다.

***

## 6. 일반화된 REML: GLMMs로의 확장

### 6.1 문제점

선형혼합모형(LMM)에서 REML의 성공에도 불구하고, **일반화 선형혼합모형(GLMM)**으로의 확장은 자명하지 않습니다.[14][11]

#### 왜 어려운가?

- **비정규 반응변수**: 이항, 포아송 등
- **링크 함수**: 비선형 변환
- **적분 불가능**: 주변 우도가 닫힌 형태(closed form)로 표현되지 않음

### 6.2 GLMM에서의 REML 확장 방법들

문헌에서 제안된 네 가지 주요 접근법:[14][11]

#### 방법 1: 근사 선형화 (Approximate Linearization)

**아이디어**: 반응변수를 **작업 변수(working variate)**로 선형화한 후 REML 적용.

Schall (1991)의 방법:

$$
\mathbf{z} = \boldsymbol{\eta} + \mathbf{D}^{-1}(\mathbf{y} - \boldsymbol{\mu})
$$

여기서:
- $$\boldsymbol{\eta} = g(\boldsymbol{\mu})$$: 링크 함수
- $$\mathbf{D} = \text{diag}(g'(\mu_i))$$: Jacobian

근사 선형 모형:

$$
\mathbf{z} \approx \mathbf{X}\boldsymbol{\beta} + \mathbf{Z}\mathbf{b} + \boldsymbol{\epsilon}
$$

이에 대해 REML 적용.

#### 방법 2: 적분 우도 (Integrated Likelihood)

Laplace 근사나 적응적 Gauss-Hermite 구적법을 사용하여 랜덤효과를 적분:

$$
L(\boldsymbol{\beta}, \boldsymbol{\alpha}) = \int p(\mathbf{y} | \mathbf{b}, \boldsymbol{\beta}) p(\mathbf{b} | \boldsymbol{\alpha}) d\mathbf{b}
$$

이후 고정효과를 적분 제거하여 REML 우도 구성.

#### 방법 3: 수정 프로파일 우도 (Modified Profile Likelihood)

프로파일 우도에 **편향 보정 항**을 추가:

$$
\ell_{MPL}(\boldsymbol{\alpha}) = \ell_{ML}(\hat{\boldsymbol{\beta}}(\boldsymbol{\alpha}), \boldsymbol{\alpha}) - \frac{1}{2}\log|\mathcal{I}_{\boldsymbol{\beta}\boldsymbol{\beta}}(\hat{\boldsymbol{\beta}}, \boldsymbol{\alpha})|
$$

이는 LMM의 REML과 유사한 구조입니다.

#### 방법 4: 점수 함수의 직접 편향 보정 (Direct Bias Correction)

ML 점수 함수에 **편향 보정 항**을 직접 추가:

$$
\mathbf{s}_{REML}(\boldsymbol{\alpha}) = \mathbf{s}_{ML}(\boldsymbol{\alpha}) + \text{correction term}
$$

### 6.3 통일된 결과

놀랍게도, 이 네 가지 접근법은 종종 **동일하거나 매우 유사한 추정량**을 제공합니다.[11][14]

***

## 7. REML의 한계와 주의사항

### 7.1 모형 비교의 제약

REML은 **고정효과가 동일한 모형**끼리만 비교 가능합니다.[15][16]

#### 왜?

REML 우도는 **잔차**에 기반하므로:

$$
\mathbf{r}_1 = (\mathbf{I} - \mathbf{P}_{\mathbf{X}_1})\mathbf{Y}, \quad \mathbf{r}_2 = (\mathbf{I} - \mathbf{P}_{\mathbf{X}_2})\mathbf{Y}
$$

$$\mathbf{X}_1 \neq \mathbf{X}_2$$이면 $$\mathbf{r}_1$$과 $$\mathbf{r}_2$$는 **서로 다른 데이터**입니다.

**해결책**: 고정효과가 다른 모형을 비교할 때는 **ML 사용**.[16][15]

### 7.2 편향-분산 트레이드오프

REML은 분산 성분에 대해 불편이지만, 고정효과 추정량은 **약간 편향**될 수 있습니다 (왜냐하면 분산 성분 추정의 불확실성 때문).[17]

***

## 요약

| 관점 | 핵심 아이디어 | 수학적 표현 |
|------|--------------|------------|
| **오차 대비** | 고정효과에 직교하는 선형 변환 | $$\mathbf{K}^T\mathbf{X} = \mathbf{0}$$, $$\mathbf{U} = \mathbf{K}^T\mathbf{Y}$$ |
| **주변화** | 고정효과를 적분으로 제거 | $$p(\mathbf{Y}\|\boldsymbol{\alpha}) = \int p(\mathbf{Y}\|\boldsymbol{\beta},\boldsymbol{\alpha})d\boldsymbol{\beta}$$ |
| **조건부 우도** | 충분통계량 주어진 조건부 분포 | $$L_{REML} = p(\mathbf{Y}\|\mathbf{t})$$ |
| **Bayesian** | 균일 사전분포의 사후 모드 | $$\hat{\boldsymbol{\alpha}} = \arg\max p(\boldsymbol{\alpha}\|\mathbf{Y})$$ |
| **편향 보정** | 자유도 손실 반영 | $$\frac{1}{n-p}$$ vs $$\frac{1}{n}$$ |
| **Fisher 정보** | 고정효과 불확실성 반영 | $$\mathbf{P} = \mathbf{V}^{-1} - \mathbf{V}^{-1}\mathbf{X}(\mathbf{X}^T\mathbf{V}^{-1}\mathbf{X})^{-1}\mathbf{X}^T\mathbf{V}^{-1}$$ |

**REML의 본질**: 고정효과 추정의 불확실성을 고려하여 분산 성분을 불편 추정하는 체계적인 방법론이며, 여러 수학적 경로를 통해 동일한 결과에 도달합니다.

[1](https://www.sciencedirect.com/science/article/pii/S0022030293775694)
[2](https://psoerensen.github.io/qgnotes/REML.pdf)
[3](https://en.wikipedia.org/wiki/Restricted_maximum_likelihood)
[4](https://www.mathworks.com/help/stats/estimating-parameters-in-linear-mixed-effects-models.html)
[5](https://pubmed.ncbi.nlm.nih.gov/19598183/)
[6](https://gksmyth.github.io/pubs/CondREML-Interface11Jul1994.pdf)
[7](https://pmc.ncbi.nlm.nih.gov/articles/PMC8550113/)
[8](https://computo-journal.org/published-202311-delattre-fim/)
[9](https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-1/REML-estimation-asymptotic-behavior-and-related-topics/10.1214/aos/1033066209.full)
[10](https://www.sciencedirect.com/science/article/pii/S0047259X83710341)
[11](https://arxiv.org/html/2402.12719v1)
[12](https://www3.stat.sinica.edu.tw/statistica/j8n3/j8n314/j8n314.htm)
[13](http://users.stat.umn.edu/~gary/classes/5303/handouts/REML.pdf)
[14](https://arxiv.org/abs/2402.12719)
[15](http://blackwell.math.yorku.ca/MATH4939/files/Mixed_Models/REML_vs_ML.pdf)
[16](https://community.jmp.com/t5/Discussions/How-to-switch-from-REML-to-ML-for-Model-Comparison/td-p/752963?code=ko-KR)
[17](https://blog.naver.com/j_nia22/220884213936)
[18](https://blog.naver.com/jaebum8888/220747546227)
[19](https://www.reddit.com/r/AskStatistics/comments/fm06z6/restricted_vs_full_maximum_likelihood_estimation/)
[20](https://www.stat.cmu.edu/technometrics/70-79/VOL-18-01/v1801031.pdf)
[21](https://docs.tibco.com/data-science/GUID-EB3C85FF-96E4-49EB-BB07-DF8CA12CAD84.html)
[22](https://xiuming.info/docs/tutorials/reml.pdf)
[23](https://www.sciencedirect.com/topics/agricultural-and-biological-sciences/restricted-maximum-likelihood)
[24](https://onlinelibrary.wiley.com/doi/pdf/10.1002/9781118762387.app3)
[25](https://www.tandfonline.com/doi/abs/10.1080/00401706.1984.10487921)
[26](https://projecteuclid.org/journals/annals-of-statistics/volume-24/issue-1/REML-estimation-asymptotic-behavior-and-related-topics/10.1214/aos/1033066209.pdf)
[27](https://www.sciencedirect.com/science/article/abs/pii/S1871141311003945)
[28](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=JAKO200416049040311&dbt=NART)
[29](https://lirias.kuleuven.be/retrieve/606854)
[30](https://www.stats.ox.ac.uk/~ripley/LME4.pdf)
[31](https://newprairiepress.org/cgi/viewcontent.cgi?article=1243&context=agstatconference)
[32](https://openresearch-repository.anu.edu.au/items/eabf5f74-2c73-4cce-a0f0-73bac79384f5)
[33](https://www.stats.net.au/Maths_REML_manual.pdf)
[34](https://onlinelibrary.wiley.com/doi/abs/10.1111/j.1467-842X.1994.tb00636.x)
[35](https://www.jstatsoft.org/article/view/v087c01/1267)
[36](https://scienceon.kisti.re.kr/srch/selectPORSrchArticle.do?cn=NART113951120)
[37](https://www.jstor.org/stable/pdf/2242618.pdf)
