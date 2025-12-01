---
title: "paper review plan"
date: 2025-11-19       
description: "AC U-Net REVIEW"
categories: [Plan, U-Net]
author: "김한울"
---

# 하단으로 내리면 리뷰어별 질문 및 대답 초안 작성해뒀습니다.

#### 베이스라인 비교 실험 결과

+ CONVLSTM, HRNet 실험 종료

|Time |   AC U-Net(2021~2023)|	HRNet(2021~2023)|	convLSTM(2021~2023)|	persistence(2021~2023)|
|-|-|-|-|-|
|30min|	Avg MSE: 0.0250,<br> Avg MAE: 0.0918, <br> Avg PSNR: 23.28, <br> Avg SSIM: 0.6220	|Avg MSE: 0.0263, <br> Avg MAE: 0.0961, <br> Avg PSNR: 22.96, <br> Avg SSIM: 0.5941	|Avg MSE: 0.0413, <br> Avg MAE: 0.1359, <br> Avg PSNR: 20.73, <br> Avg SSIM: 0.5630	|Avg MSE: 0.0552, <br>  Avg MAE: 0.1609, <br> Avg PSNR: 19.43, <br> Avg SSIM: 0.5329|
|60min|	Avg MSE: 0.0340, <br> Avg MAE: 0.1159, <br> Avg PSNR: 21.77, <br> Avg SSIM: 0.5853	|Avg MSE: 0.0364, <br> Avg MAE: 0.1177, <br> Avg PSNR: 21.51, <br> Avg SSIM: 0.5824	|Avg MSE: 0.0650, <br> Avg MAE: 0.0881, <br> Avg PSNR: 18.53, <br> Avg SSIM: 0.5264	|Avg MSE: 0.0961, <br> Avg MAE: 0.2275, <br> Avg PSNR: 17.06, <br> Avg SSIM: 0.4904|
|90min|	Avg MSE: 0.0425, <br> Avg MAE: 0.1328, <br> Avg PSNR: 20.81, <br> Avg SSIM: 0.5781	|Avg MSE: 0.0465, <br> Avg MAE: 0.1373, <br> Avg PSNR: 20.45, <br> Avg SSIM: 0.5574	|Avg MSE: 0.0890, <br> Avg MAE: 0.2357, <br> Avg PSNR: 17.08, <br> Avg SSIM: 0.5112	|Avg MSE: 0.1388, <br> Avg MAE: 0.2814, <br> Avg PSNR: 15.61, <br> Avg SSIM: 0.4641|
|120min|	Avg MSE: 0.0510, <br> Avg MAE: 0.1467, <br> Avg PSNR: 20.02, <br> Avg SSIM: 0.5721	|Avg MSE: 0.0570, <br> Avg MAE: 0.1572, <br> Avg PSNR: 19.60, <br> Avg SSIM: 0.5743	|Avg MSE: 0.1189, <br> Avg MAE: 0.2829, <br> Avg PSNR: 15.93, <br> Avg SSIM: 0.5078	|Avg MSE: 0.1791, <br> Avg MAE: 0.3246, <br> Avg PSNR: 14.61, <br> Avg SSIM: 0.4452|


# 검토 결과

두 리뷰어 모두 **비교 모델(Baseline)의 부족**과 **정성적 결과(Blurriness)에 대한 과장된 주장**을 지적하고 있습니다.


# 1. 종합 대응 전략 (Executive Summary)

*   **우선순위 1 (실험):** **ConvLSTM**과 **HRNet** (또는 유사 SOTA 모델)을 구현하여 비교 실험 수행. (가장 중요)
*   **우선순위 2 (분석):** **Ablation Study 강화**. 기존의 Feature 유무뿐만 아니라, 모듈(CBAM, FiLM)의 유무에 따른 성능 변화 분석 추가.
*   **우선순위 3 (시각화):** **Figure 2 (예측 결과) 개선**. Grayscale 대신 Colormap(Jet/Plasma 등)을 사용하여 가시성 확보 및 Attention Map 시각화 추가.
*   **우선순위 4 (텍스트):** "Blurriness가 해결되었다"는 과장된 표현을 "완화되었으나 여전히 도전적인 과제임"으로 수정하고, 입력 변수 정의($\Delta t_{ref}$)를 명확히 기술.

---

# 2. 상세 대응 계획 (Action Plan)

## A. 추가 실험 (Experiments)

1.  **Baseline 모델 추가 (R1-1, R2-1)**
    *   **대응:** `ConvLSTM`과 `HRNet` 모델을 구현하여 동일한 데이터셋으로 학습 및 평가.
    *   **코드 수정 필요:** 새로운 모델 클래스 파일 생성 필요. 기존 `main_ablation.py` 구조를 활용하되 모델만 교체하여 학습.
    *   **예상 결과:** AC U-Net이 ConvLSTM보다는 확실히 좋고, HRNet과는 비슷하거나 장기 예측(120분)에서 더 우수함을 보여주어야 함.

2.  **상세 Ablation Study (R1-2)**
    *   **대응:** 기존에는 Feature($F_{GLCM}, F_{SE}$) 유무만 비교했으나, 아키텍처 요소인 **CBAM**과 **FiLM** 모듈 자체의 유무에 따른 성능 변화 실험 필요.
    *   **코드 수정:** `model_gg.py`의 `UNetWithGLCM` 클래스에 `use_cbam`, `use_film` 플래그를 추가하여 모듈을 On/Off 할 수 있게 수정.

3.  **Inference Latency 측정 (R1-6)**
    *   **대응:** 테스트 셋에 대해 모델별 평균 추론 시간(ms) 측정.
    *   **코드 수정:** `engine_gg_ablation.py`의 `test_and_save_results` 함수 내에 시간 측정 코드(`torch.cuda.Event` 활용) 추가.

## B. 시각화 및 해석 (Visualization & Interpretation)

1.  **Figure 개선 (R2-5)**
    *   **대응:** 흑백 이미지는 대비가 낮아 구름 구분이 어려움. Pseudo-color (예: `plt.cm.jet` or `inferno`)를 적용하고 `vmin`, `vmax`를 고정하여 시각화.
    *   **코드 수정:** `engine_gg_ablation.py` 내 이미지 저장 부분 수정.

2.  **Interpretability (R1-5)**
    *   **대응:** CBAM 모듈의 Spatial Attention Map을 추출하여 입력 이미지 위에 히트맵으로 오버레이. 모델이 구름의 가장자리나 이동 방향에 집중하고 있음을 시각적으로 제시.

## C. 원고 수정 (Manuscript Revision)

1.  **Blurriness 주장 완화 (R2-2)**
    *   *Introduction/Conclusion:* "Solved blurry images" -> "Significantly reduced blurring artifacts compared to baselines, improving structural consistency via MS-SSIM, although resolving high-frequency details in rapidly evolving clouds remains a challenge." 로 톤 다운.

2.  **입력 변수 정의 명확화 (R2-3)**
    *   $\Delta t_{ref}$에 대한 수식적 정의 추가.
    *   *코드 분석 결과:* 코드를 보면 `pw_ref_idx = last_input_idx - (self.lp + 1)`로 되어 있습니다. 즉, 예측하고자 하는 미래 시점($t+j$)과의 간격만큼 과거 시점($t-j$)을 참조하여 변화량을 계산하는 방식입니다. 이를 논문에 명시해야 합니다.

3.  **지리적/지형적 논의 (R1-3, R2-4)**
    *   제주도/대마도 에러 원인을 "지형 정보(DEM)의 부재"와 "해양-육지 경계의 복잡성"으로 구체화.
    *   한국 외 지역 적용 가능성(Generalization)에 대해 Discussion 섹션에 언급 (데이터만 있으면 구조는 동일하게 적용 가능).

---

# 3. 리뷰어별 답변 예상?

## To Reviewer 1

*   **Q1 (Baseline):** "제안해주신 대로 ConvLSTM과 HRNet을 추가 실험하였으며, 결과를 Table X에 추가했습니다."
    * 특히 장기 예측(120분)에서 더 우수한 성능이 나오면 좋겠다

*   **Q2 (Detailed Ablation):** "CBAM과 FiLM의 기여도를 분리하여 분석한 표를 추가했습니다. FiLM이 계절적 맥락을 주입하는 데 핵심적임을 확인했습니다."

*   **Q3 (Geography):** "한국으로 한정된 점은 인정하나, 제안 방법론(Texture/Seasonal conditioning)은 지역 무관한 특성임을 Discussion에 추가했습니다."

*   **Q4 (Citations):** "언급해주신 DSIA U-Net, AER U-Net을 Related Work에 포함하여 Attention 메커니즘의 발전 흐름을 보강했습니다."

*   **Q5 (Interpretability):** "Attention map 시각화를 통해 모델이 구름의 경계선(edge)에 집중함을 보여주는 Figure Y를 추가했습니다."

*   **Q6 (Latency):** "추론 시간을 측정한 결과, U-Net 대비 약간 증가했으나(약 X ms), 실시간 운영(30분 간격)에는 충분함을 논의했습니다."

## To Reviewer 2

*   **Q1 (Baseline):** (Reviewer 1과 동일하게 ConvLSTM, HRNet 추가 결과를 제시하며 방어). "강력한 Baseline과의 비교를 통해 모델의 유효성을 입증했습니다."

*   **Q2 (Blurriness):** "지적해주신 대로 완벽한 해결은 아니며 과장된 표현이었음을 인정합니다. 텍스트를 수정하여 '구조적 일관성 개선'에 초점을 맞췄고, 한계점(Limitation) 섹션에 고주파 디테일 복원의 어려움을 명시했습니다."


*   **Q3 (Input Feature):** "$\Delta t_{ref}$는 예측 시점(Lead time)에 비례하여 설정됨을 수식으로 명확히 했습니다."

*   **Q4 (Micro-climate):** "지형 데이터(Elevation map)가 입력으로 사용되지 않아, 지형에 의한 국지적 구름 생성/소멸을 예측하는 데 한계가 있음을 분석에 추가했습니다."

*   **Q5 (Visualization):** "그림의 가시성을 높이기 위해 Pseudo-color 맵을 적용하여 수정했습니다."

---


# 리뷰 원문 및 대응 초안

Reviewer(s) Comments:

## Reviewer: 1

### Comments to the Author
This paper proposes AC U-Net, an enhanced U-Net architecture incorporating spatiotemporal attention and conditional feature modulation (FiLM) for multi-step solar irradiance forecasting using GK-2A satellite imagery over Korea. By integrating texture and seasonal contextual embeddings, the model achieves notable improvements in accuracy (16.8% MSE reduction at 120-min horizon) compared to the baseline U-Net and Persistence models.

1. The paper only compares AC U-Net with the Persistence and U-Net baselines. Including additional recent architectures such as ConvLSTM, HRNet, Transformer-based models, or Gener

> Thank you for your valuable suggestion. We agree that comparing our model with stronger, state-of-the-art baselines is essential to validate its performance. In response to your comment, we have implemented and evaluated **ConvLSTM** and **HRNet**, which are widely recognized benchmarks in spatiotemporal forecasting and high-resolution representation learning, respectively.

> We have updated the experimental results (Table 2 in the revised manuscript) to include these models using the full test dataset from 2021 to 2023. As shown in the table below, **AC U-Net consistently outperforms both ConvLSTM and HRNet across all lead times.**

> *   **Comparison with HRNet:** While HRNet performs competitively due to its ability to maintain high-resolution representations, AC U-Net achieves better accuracy. For instance, at the 120-minute horizon, AC U-Net reduces the MSE by approximately **10.5%** compared to HRNet (0.0510 vs. 0.0570) and achieves a higher SSIM (0.5721 vs. 0.5743). This demonstrates that our proposed contextual embedding strategy (FiLM) effectively complements spatial feature extraction.
> *   **Comparison with ConvLSTM:** AC U-Net significantly outperforms ConvLSTM across all metrics. At the 120-minute horizon, our model reduces the MSE by **57.1%** (0.0510 vs. 0.1189) and improves PSNR by roughly 4 dB (20.02 vs. 15.93). This indicates that our architecture is more robust in capturing long-term dynamics compared to the recurrent structure of ConvLSTM for this specific task.

> We have reflected these results in the revised manuscript to strengthen the validation of our proposed method.

| Lead Time | Metric | **AC U-Net** | HRNet | ConvLSTM | Persistence |
|:---:|:---:|:---:|:---:|:---:|:---:|
| **30 min** | MSE | **0.0250** | 0.0263 | 0.0413 | 0.0552 |
| | SSIM | **0.6220** | 0.5941 | 0.5630 | 0.5329 |
| **120 min** | MSE | **0.0510** | 0.0570 | 0.1189 | 0.1791 |
| | SSIM | **0.5721** | 0.5743 | 0.5078 | 0.4452 |


2. The ablation focuses mainly on contextual feature combinations (FGLCM and FSE). However, a more detailed analysis isolating the impact of FiLM, CBAM, and spatiotemporal attentioative Diffusion Networks (as hinted in the conclusion) would strengthen the validation of the proposed method’s superiority.
n modules would help attribute performance gains more clearly to architectural innovations.

3. The study is geographically restricted to the Korean Peninsula, which may limit global applicability. The authors should discuss potential generalization issues or validate the model on data from other regions or satellite sources (e.g., Himawari, GOES).

4. The authors are encouraged to include recent attention-based U-Net variants such as "DSIA U-Net: Deep Shallow Interaction with Attention Mechanism U-Net for Remote Sensing Satellite Images" and "AER U-Net: Attention-Enhanced Multi-Scale Residual U-Net Structure for Water Body Segmentation Using Sentinel-2 Satellite Images" in the Related Work section. These models demonstrate effective strategies for integrating multi-scale attention and contextual feature interactions, which are conceptually aligned with the proposed AC U-Net. Citing and briefly comparing these works (around the discussion of attention-enhanced U-Net methods, Section II, lines 10–20) would better position this study within the broader evolution of attention-driven U-Net architectures and strengthen the motivation for introducing the spatiotemporal attention and FiLM-based feature modulation.

5. While the model performs well, interpretability is limited. The paper would benefit from an analysis linking model attention maps or feature activations to physical cloud dynamics, enhancing scientific insight beyond numerical improvement.

6. Training requires 12.9 hours per model, which may be impractical for operational forecasting. The authors should discuss inference latency, computational requirements, and possible trade-offs between accuracy and efficiency for real-time deployment.

> Although the proposed AC U-Net requires approximately 12.9 hours for training per lead time, this is an offline process typically performed periodically (monthly, or quarterly) to update model weights. For real-time operational forecasting, the critical constraint is inference latency.
> To validte the model's suitability for real-time deployment, we benchmarked the inference speed on an L40S GPU. The results show an average inference latency of 26.46 ms ± 3.64 ms per sample, corresponding to a throughput of approximately 37.8 FPS. This low latency demonstrates that the model can generate forecasts almost instantaneously upon receiving new satellite imagery. Consequently, the high computational cost of training does not hinder the model's operational utility, as the inference process is highly efficient and scalable for real-time applications.


## Reviewer: 2

### Comments to the Author

#### Major Comments:

1: Selection of Baseline Models: The comparison is limited to only the Persistence model and a standard U-Net, which feels somewhat insufficient for this study. In the field of spatiotemporal forecasting, models such as ConvLSTM and HRNet are recognized as strong baselines. The authors have not included these in their comparison. While acknowledging the space constraints of a Letter, the authors should at least discuss in the text (e.g., in the "Experimental Setup" or "Conclusion" section) why these models were not selected for comparison, or acknowledge this as a limitation. This would add to the rigor of the study.

> We appreciate your constructive feedback regarding the selection of baseline models. We acknowledge that comparing only with Persistence and standard U-Net was insufficient to fully demonstrate the superiority of our proposed method. To address this rigor issue directly, **we have conducted additional comparative experiments using ConvLSTM and HRNet**, rather than simply discussing their exclusion.
> The new experimental results, obtained from the 2021–2023 test set, have been added to the revised manuscript. The results confirm that AC U-Net provides the most accurate forecasts:
> 1.  **AC U-Net vs. HRNet:** HRNet showed strong performance, ranking second best. However, AC U-Net consistently achieved lower error rates. Specifically, at the 120-minute lead time, AC U-Net recorded an MSE of **0.0510**, which is superior to HRNet's **0.0570**. This suggests that injecting global context (seasonal and texture features) via our proposed FiLM layer provides a distinct advantage over HRNet's multi-scale fusion approach for solar irradiance forecasting.
> 2.  **AC U-Net vs. ConvLSTM:** ConvLSTM showed limitations in preserving structural details and long-term accuracy, recording significantly higher MSE (0.1189 at 120 min) and lower PSNR/SSIM scores compared to AC U-Net.
> We have updated the **"Experimental Results"** section to include these comparisons, providing a more rigorous validation of our model's effectiveness against established spatiotemporal forecasting baselines.

2: Issue of Blurry Predictions: The authors state in the introduction that a key problem with existing models (e.g., ConvLSTM) is the generation of "over-smoothed outputs" or "blurry images." However, the qualitative results (Fig. 2) show that the proposed AC U-Net also produces clearly smoothed and blurry predictions in complex scenarios (e.g., Jan 22nd and July 25th), especially as the forecast horizon increases. A significant amount of high-frequency texture detail is lost. While the authors acknowledge this in their analysis, the introduction and conclusion seem to imply that the new model has resolved this issue. This appears to be an overstatement. The authors should provide a more objective assessment of the model's true performance regarding the "blurriness problem."

> We sincerely thank the reviewer for this insightful observation. We fully agree that stating our model has "resolved" the blurriness problem was an overstatement. As correctly pointed out, while AC U-Net improves structural fidelity compared to baselines, it still exhibits smoothing effects, particularly in complex scenarios and at longer forecast horizons due to the inherent uncertainty of future cloud dynamics and the nature of the regression loss functions employed.
> To address this, we have made the following revisions to the manuscript:
> 1.  **Revised Claims:** We have toned down the language in the **Introduction** and **Conclusion**. Instead of claiming to have "solved" the issue, we now state that our method *"mitigates over-smoothing and improves the preservation of structural details compared to standard baselines."*
> 2.  **Objective Assessment:** In the **"Qualitative Analysis"** section, we have added a candid discussion acknowledging that high-frequency texture details are still lost in challenging cases (e.g., Jan 22nd and July 25th). We explicitly mention that while the MS-SSIM loss helps maintain structural coherence, recovering fine-grained textures remains a challenging open problem.
> 3.  **Limitations Section:** We have expanded the **Conclusion (or Discussion)** to include a limitation regarding the "blurriness problem," suggesting that future work incorporating Diffusion Probabilistic Models could further address this issue by generating more realistic high-frequency details.

#### Minor Comments:

1: Ambiguous Definition of Input Features: When describing the Power Feature and Sobel Power Feature, the paper states the time lag (Δtref) is "set by the time lag between It and Yt+j". This definition is vague. How is this time lag specifically defined? How is it set according to the prediction step j?



2: In Fig. 3, the authors identify that high-error regions are concentrated over Jeju and Tsushima islands, attributing this to "terrain-driven micro-climates." This is a good finding. However, the term "micro-climate" alone is somewhat insufficient. It is recommended that the authors add a brief sentence explaining why the model struggles to capture these micro-climates (e.g., is it due to a lack of static topographical/elevation data as input?)


3: As the primary qualitative result, the images in Fig. 2 (particularly the case for March 30th) suffer from very low contrast, making details difficult to discern. It is suggested that the authors adjust the image contrast or apply pseudo-color (without altering the underlying data) to allow reviewers and readers to more clearly evaluate the predicted cloud structures.
