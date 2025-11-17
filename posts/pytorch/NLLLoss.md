---
title: "NLL Loss - pytorch"
date: 2025-11-17       
description: "NLL Loss"
categories: [Pytorch, LossFunction]
author: "김한울"
---

## PyTorch `NLLLoss` 완벽 가이드

`NLLLoss`는 **음성 로그 가능도 손실(Negative Log Likelihood Loss)**의 약자로, 다중 클래스 분류(multi-class classification) 문제를 해결하기 위한 딥러닝 모델을 훈련할 때 주로 사용되는 손실 함수입니다.

이 손실 함수의 핵심은 **모델의 출력이 로그 확률(log-probabilities) 값이라고 가정**하고 손실을 계산한다는 점입니다.

### ⚠️ 가장 중요한 핵심: `CrossEntropyLoss`와의 관계

많은 사용자들이 `NLLLoss`와 `CrossEntropyLoss`의 차이를 헷갈려 합니다. 결론부터 말하면, **`CrossEntropyLoss`는 내부적으로 `LogSoftmax`와 `NLLLoss`를 합친 것과 같습니다.**

> `nn.CrossEntropyLoss(input, target)` ≈ `nn.NLLLoss(nn.LogSoftmax(dim=1)(input), target)`

따라서, 대부분의 분류 모델에서는 마지막 레이어에 활성화 함수 없이 `CrossEntropyLoss`를 사용하는 것이 더 간편하고 수치적으로 안정적입니다. `NLLLoss`는 모델의 마지막 레이어에 `LogSoftmax`를 직접 추가하여 로그 확률 값을 출력하도록 설계했을 때 사용합니다.

---

### 클래스 정의

```python
class torch.nn.NLLLoss(
    weight=None,
    size_average=None,  # Deprecated
    ignore_index=-100,
    reduce=None,        # Deprecated
    reduction='mean'
)
```

### 파라미터(Parameters) 상세 설명

*   **`weight`** (`Tensor`, optional): 각 클래스(class)에 수동으로 가중치를 부여하는 1D 텐서입니다. 클래스의 개수(`C`)와 동일한 크기여야 합니다. 데이터셋이 불균형할 때 특정 클래스에 더 높은 가중치를 주어 학습에 영향을 줄 수 있습니다. 기본값은 모든 클래스에 1을 부여하는 것과 같습니다.

*   **`ignore_index`** (`int`, optional): 손실 계산에서 무시할 특정 정답(target) 값을 지정합니다. 예를 들어, 자연어 처리에서 패딩(padding) 토큰을 무시하고 싶을 때 해당 토큰의 인덱스를 여기에 지정하면, 그 부분에서는 손실이 계산되지 않고 역전파에도 영향을 주지 않습니다. 기본값은 `-100`입니다.

*   **`reduction`** (`str`, optional): 계산된 손실 값들을 어떻게 처리할지 지정합니다.
    *   `'none'`: 아무 처리도 하지 않고, 각 데이터 샘플에 대한 손실 값을 그대로 반환합니다. 출력은 배치 크기와 같은 모양의 텐서가 됩니다.
    *   `'mean'` (기본값): 모든 손실 값의 가중 평균을 계산하여 단일 스칼라 값으로 반환합니다.
    *   `'sum'`: 모든 손실 값의 합을 계산하여 단일 스칼라 값으로 반환합니다.

> **참고:** `size_average`와 `reduce`는 더 이상 사용되지 않는(deprecated) 인자입니다. `reduction`을 사용해 주세요.

---

### 동작 원리 및 수식

`NLLLoss`의 계산 방식은 매우 직관적입니다. 모델이 출력한 **로그 확률** 중에서 **정답 클래스에 해당하는 값**을 가져온 뒤, 부호를 반전시켜 손실로 사용합니다.

예를 들어, 3개 클래스 분류 문제에서 모델의 출력이 다음과 같다고 가정해 봅시다. (이미 `LogSoftmax`를 거친 값)
*   `input` = `[-0.7, -1.2, -2.5]` (클래스 0, 1, 2에 대한 로그 확률)
*   `target` = `1` (정답은 1번 클래스)

`NLLLoss`는 `input`에서 `target` 인덱스에 해당하는 값, 즉 `-1.2`를 선택하고 부호를 바꿉니다. 따라서 이 데이터 샘플의 손실은 `1.2`가 됩니다.

수식은 다음과 같습니다:
*   `l(x, y) = L = {l_1, ..., l_N}`
*   `l_n = -w_{y_n} * x_{n, y_n}`

여기서 `x`는 입력(로그 확률), `y`는 정답(타겟), `w`는 가중치, `N`은 배치 크기입니다. `reduction`이 `'mean'`이면 이 `l_n`들의 평균을 구하고, `'sum'`이면 합을 구합니다.

---

### 입출력 형태 (Shape)

*   **입력 (Input)**: `(N, C)` 또는 `(N, C, d1, d2, ...)` 형태의 텐서.
    *   `N`: 배치 크기 (Batch Size)
    *   `C`: 클래스의 개수 (Number of Classes)
    *   `d1, d2, ...`: 이미지 데이터의 높이, 너비 등 추가적인 차원. 이미지 분할(Image Segmentation) 같은 픽셀 단위 분류에서 사용됩니다.

*   **타겟 (Target)**: `(N)` 또는 `(N, d1, d2, ...)` 형태의 텐서.
    *   **매우 중요**: 타겟은 각 데이터의 정답 클래스 **인덱스**를 담고 있어야 하며, 데이터 타입은 `torch.long`이어야 합니다.
    *   값의 범위는 `[0, C-1]` 이어야 합니다.

*   **출력 (Output)**:
    *   `reduction='mean'` 또는 `'sum'`: 스칼라(Scalar) 값 (하나의 숫자).
    *   `reduction='none'`: 타겟과 동일한 형태의 텐서.

---

### 사용 예제 (Code Examples)

#### 1. 기본 사용법 (1D 데이터)

```python
import torch
import torch.nn as nn

# 1. 모델의 출력을 로그 확률로 변환하는 LogSoftmax 레이어
log_softmax = nn.LogSoftmax(dim=1)

# 2. NLLLoss 함수 정의
loss_fn = nn.NLLLoss()

# 3. 가상의 모델 출력 (LogSoftmax를 거치기 전의 raw output)
# 배치 크기(N)=3, 클래스 수(C)=5
raw_output = torch.randn(3, 5, requires_grad=True)

# 4. 모델의 출력을 로그 확률로 변환
log_probs = log_softmax(raw_output)

# 5. 정답 데이터 (클래스 인덱스, torch.long 타입)
# 각 값은 0에서 4 사이여야 함 (0 <= value < C)
target = torch.tensor([1, 0, 4])

# 6. 손실 계산
loss = loss_fn(log_probs, target)

print("모델의 Raw Output:\n", raw_output)
print("\nLog Softmax 결과 (NLLLoss의 입력):\n", log_probs)
print("\n정답 (Target):\n", target)
print(f"\n계산된 손실 (Loss): {loss.item():.4f}")

# 7. 역전파 수행
loss.backward()
print("\nInput의 Gradient:\n", raw_output.grad)
```

#### 2. 이미지 데이터 예제 (2D 데이터)

이미지 분할(Image Segmentation)과 같이 각 픽셀을 분류해야 하는 경우에 사용됩니다.

```python
import torch
import torch.nn as nn

# 배치 크기=5, 클래스 수=4
N, C = 5, 4

# NLLLoss 함수 정의
loss_fn = nn.NLLLoss()

# 1. 가상의 이미지 데이터 및 모델
# 입력 데이터: (N, 16, 10, 10)
data = torch.randn(N, 16, 10, 10)
# Conv2d를 거쳐 (N, C, 8, 8) 형태의 출력을 만듦
conv = nn.Conv2d(16, C, kernel_size=(3, 3))
log_softmax = nn.LogSoftmax(dim=1) # 채널 차원에 대해 소프트맥스 적용

# 2. 모델의 최종 출력 (로그 확률)
# (N, C, H, W) -> (5, 4, 8, 8)
output = log_softmax(conv(data))

# 3. 정답 데이터 (각 픽셀에 대한 클래스 인덱스)
# (N, H, W) -> (5, 8, 8) 형태이며, 각 값은 [0, C-1] 범위
target = torch.empty(N, 8, 8, dtype=torch.long).random_(0, C)

# 4. 손실 계산
loss = loss_fn(output, target)

print(f"입력(Output) 형태: {output.shape}")
print(f"타겟(Target) 형태: {target.shape}")
print(f"\n계산된 손실 (Loss): {loss.item():.4f}")

# 5. 역전파
loss.backward()
```

### 핵심 요약 및 주의사항

1.  **입력은 반드시 로그 확률(log-probabilities)이어야 합니다.** 모델 마지막 단에 `nn.LogSoftmax(dim=1)`을 적용한 결과를 입력으로 넣어주세요.
2.  **`CrossEntropyLoss` 사용을 먼저 고려하세요.** `LogSoftmax`와 `NLLLoss`를 합친 `CrossEntropyLoss`가 더 간편하고 안정적입니다.
3.  **타겟(Target)은 클래스 인덱스여야 합니다.** One-hot 인코딩된 벡터가 아닌, `[0, 1, 4, ...]` 와 같은 형태의 `torch.long` 타입 텐서를 사용해야 합니다.
4.  `ignore_index`를 사용하면 특정 레이블을 손실 계산에서 제외할 수 있어 편리합니다. (예: 패딩 토큰)
5.  클래스 불균형 문제가 심각하다면 `weight` 인자를 사용하여 소수 클래스의 중요도를 높여보세요.