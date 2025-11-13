## 1. 연구 주제(가제)

> **다변량 딥러닝 기반 이상치 탐지에서의 불확실성 추정과 위험 민감 의사결정 프레임워크**

영문 예시:

> **Risk-Aware Decision Framework with Uncertainty-Aware Deep Multivariate Anomaly Detection**

---

## 2. 연구 배경 및 필요성

최근 제조 공정, 금융 거래, 네트워크 트래픽, 의료 모니터링 등 다양한 분야에서
**다변량(multivariate) 시계열·표형 데이터**를 이용한 이상치 탐지가 중요해지고 있다.
기존 딥러닝 기반 이상치 탐지 모델(Autoencoder, VAE, LSTM-AE 등)은 높은 표현력을 가지지만,

* 이상치 점수만 제공하고,
* **모델의 신뢰도/불확실성(uncertainty)을 정량적으로 제공하지 못하며,**
* 실제 운영환경에서 요구되는 **비용 구조(오경보·미탐지·추가 점검 비용 등)를 반영한 risk-aware 의사결정**까지는 연결되지 않는 한계가 있다.

이에 본 연구는 **딥러닝 기반 베이지안(latent factor/autoencoder) 모델**을 이용해
다변량 데이터에서 **이상치 점수와 불확실성을 동시에 추정하고**,
이를 활용해 **STOP / CHECK / IGNORE**와 같은 위험 민감 의사결정을 수행하는
프레임워크를 제안하고자 한다.

---

## 3. 연구 목표 및 연구 질문

### 3.1 연구 목표

1. **다변량 데이터를 위한 딥러닝 기반 Bayesian Latent Factor/Autoencoder 모델**을 설계하여

   * 전역(global) 이상치 점수와 불확실성,
   * 변수별(feature-wise) 이상치 점수와 불확실성을 동시에 제공한다.
2. 추정된 이상치·불확실성 정보를 활용하여
   **비용 구조를 반영한 risk-aware 의사결정 규칙(STOP/CHECK/IGNORE)을 설계**한다.
3. 제안 방법이 **여러 도메인(예: 공정·트래픽·일반 multivariate benchmark)** 에서
   범용적으로 동작함을 실험적으로 검증한다.

### 3.2 연구 질문 (Research Questions)

* **RQ1.** 다변량 딥러닝 모델(VAE/AE 계열)을 Bayesian/Ensemble 방식으로 확장하여,
  전역 및 변수별 이상치 점수와 불확실성을 동시에 추정할 수 있는가?
* **RQ2.** 추정된 이상치 점수 (A(x))와 불확실성 (U(x))를 이용해,
  도메인별 비용 구조를 반영한 **risk-aware 의사결정 규칙**을 설계할 수 있으며,
  이는 기존 “점수 기준 단일 threshold” 방식보다 더 낮은 위험도(risk)를 달성하는가?
* **RQ3.** 제안 프레임워크는 공정/트래픽/일반 multivariate 데이터 등
  **서로 다른 도메인에서도 일관된 성능·의사결정 이득**을 제공하는가?

---

## 4. 문헌 조사 계획 및 키워드 세트

### 4.1 문헌 조사 기간

* **1–2개월차 (집중 문헌 조사 및 정리)**

  * 이후에도 필요 시 지속적으로 업데이트하되,
    초기 2개월 동안 “큰 그림 + 대표 논문”을 정리하는 것을 목표로 함.

### 4.2 키워드 세트

(영문 위주 + 필요시 한글 병기)

* **Anomaly / Outlier Detection 관련**

  * multivariate anomaly detection
  * multivariate time series anomaly detection
  * deep learning anomaly detection, deep autoencoder, VAE anomaly detection
  * reconstruction-based anomaly detection, density-based anomaly detection

* **딥러닝 & 베이지안 / 불확실성**

  * deep learning, representation learning
  * Bayesian deep learning, variational inference
  * Monte Carlo Dropout, deep ensemble
  * uncertainty quantification, aleatoric/epistemic uncertainty
  * Bayesian VAE, probabilistic autoencoder, latent factor model

* **의사결정 & 위험**

  * selective classification, selective prediction, abstention
  * risk-aware decision making, cost-sensitive learning
  * risk–coverage trade-off, calibration, decision-theoretic anomaly detection

### 4.3 문헌 정리 방식

* 스프레드시트/Notion에 다음 항목으로 구조화:

  * 제목 / 연도 / venue
  * 데이터 타입 (시계열, 이미지, 트래픽, 센서 등)
  * 딥러닝 여부 / 베이지안·UQ 사용 여부
  * 이상치 점수 정의 방식
  * risk-aware / selective decision 관점 고려 여부
  * “본 논문과의 차이/갭” 한 줄 요약

---

## 5. 연구 방법

### 5.1 모델링 개요

1. **Backbone: 딥러닝 기반 Bayesian Latent Autoencoder (PyTorch/fastai 사용)**

   * 입력: $x \in \mathbb{R}^d$ (다변량 벡터 또는 윈도우된 시계열)
   * 인코더 $q_\phi(z \vert x)$, 디코더 $p_\theta(x \vert z)$를

     * MLP / 1D-CNN / LSTM / Transformer 등으로 구성 (도메인에 맞게 선택)
   * 베이지안/불확실성:

     * Variational VAE 구조
     * * MC Dropout 또는 Deep Ensemble을 활용해 **모델 불확실성** 추정

2. **이상치 점수 및 불확실성 정의**

   * 변수별 reconstruction error: (r_j(x))
   * 변수별 예측 분산(uncertainty): (u_j(x))
   * 전역 이상치 점수: (A(x) = \sum_j w_j r_j(x))
   * 전역 불확실성: (U(x) = \sum_j w_j u_j(x)) (합/평균/최댓값 등)

3. **위험 민감 의사결정 레이어**

   * 액션: (\delta(x) \in {\text{IGNORE}, \text{CHECK}, \text{STOP}})
   * 실제 상태 (y \in {\text{normal}, \text{anomaly}}), 비용 함수 (C(\delta(x), y)) 정의
   * 단순하지만 해석 가능한 rule 예:

     * (A(x) \ge \tau_A, U(x) \le \tau_U \Rightarrow STOP)
     * (A(x) \ge \tau_A, U(x) > \tau_U \Rightarrow CHECK)
     * (A(x) < \tau_A \Rightarrow IGNORE)
   * validation set에서 경험적 risk (\hat R(\tau_A,\tau_U)) 최소화하는 ((\tau_A,\tau_U)) 탐색

### 5.2 이론 분석 (초기 계획)

* selective risk / coverage 개념 도입
* 이상치·불확실성 score가 만족해야 할 성질(단조성, calibration 등)을 가정하고,
  제안 decision rule의 **risk–coverage trade-off**에 대한 이론적 성질(간단한 bound 또는 lemma 수준)을 도출하는 것을 목표로 함.

### 5.3 데이터셋 및 실험 디자인 (예시)

* **Synthetic multivariate 데이터**

  * 다변량 Gaussian, mixture, non-linear manifold 등에서 inlier/outlier 생성
  * score & UQ 성질 및 decision rule 기본 검증

* **실제 또는 공개 다변량 데이터**

  * 예: multivariate time series anomaly benchmark(SMD, MSL, SWaT 등)
  * 필요 시 트래픽/센서/탭형 anomaly dataset 병행

* **비교 기준**

  * 딥러닝 기반 이상치 탐지 baseline (AE, VAE, LSTM-AE 등)
  * UQ 없는 score-only decision vs. 제안 UQ-aware risk decision

* **평가지표**

  * anomaly detection 성능: AUROC, AUPR, F1, FPR@95TPR 등
  * decision 관점:

    * risk–coverage curve
    * false alarm 수, missed anomaly 수
    * CHECK(사람 검토) 발생 비율 등

### 5.4 구현 환경

* **프레임워크**: PyTorch (필수), fastai (학습 loop·callback 활용 시)
* **언어/환경**: Python, CUDA GPU (학교 서버 or 클라우드)
* 코드 구조:

  * `models/` (encoder/decoder, Bayesian/ensemble 모듈)
  * `experiments/` (데이터셋별 스크립트, config)
  * `analysis/` (결과 시각화, risk–coverage 분석)

---

## 6. 연구 일정(예시, 12개월 기준)

> 실제 시작 시점에 맞춰 “월” 번호만 조정하면 됨.

1. **1–2개월차: 문헌 조사 및 문제 정의**

   * 키워드 기반 논문 수집 및 표 정리
   * 관련 연구 요약 문서 작성 (2–3p)
   * 연구 질문(RQ) 및 기여 포인트 확정

2. **3–4개월차: 수식 정식화 및 모델 설계**

   * Bayesian 딥러닝 기반 latent 모델 구조 수식 정리
   * 이상치·불확실성 스코어 정의
   * risk-aware decision rule 및 비용 함수 정의
   * 정리/lemma 초안 작성

3. **5–7개월차: PyTorch/fastai 구현 및 초기 실험**

   * backbone 모델 및 UQ 모듈 구현
   * synthetic + 1개 공개 데이터셋으로 1차 검증
   * 코드 안정화 및 hyperparameter 기본 설정

4. **8–9개월차: 본 실험 및 ablation**

   * 1–2개 도메인 데이터셋에서 본 실험 수행
   * baseline과의 비교, ablation study
   * risk–coverage, decision-level 분석 및 시각화

5. **10–11개월차: 논문 본문 집필**

   * 3–5장(방법·이론·실험) 우선 작성
   * 1–2장(서론·관련연구), 6장(결론) 작성 및 통합
   * 지도교수 피드백 반영, 1–2회 수정

6. **12개월차: 최종 정리 및 제출**

   * 오탈자·형식·참고문헌 정리
   * 그림/표 정리 및 통일
   * 심사 발표 자료 준비 및 리허설

---

이 정도면 **지도교수님께 그대로 가져가서 논의 가능한 수준의 연구계획서 뼈대**가 될 거예요.
원하면 이걸 **진짜 LaTeX / 한글 보고서 형식(표지, 목차, 요약 포함)**으로도 한 번 잡아줄 수 있어요.
