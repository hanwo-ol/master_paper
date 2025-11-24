---
title: "What is Meta Learning?"
date: 2025-11-24       
description: "Paper Review Twice! : Meta Learning in Neural Networks: A Survey"
categories: [Metalearning, survey, review, Idea]
author: "김한울"
---

# Paper Review Twice: Meta Learning in Neural Networks: A Survey

``` 
@article{hospedales2021meta,
  title={Meta-learning in neural networks: A survey},
  author={Hospedales, Timothy and Antoniou, Antreas and Micaelli, Paul and Storkey, Amos},
  journal={IEEE transactions on pattern analysis and machine intelligence},
  volume={44},
  number={9},
  pages={5149--5169},
  year={2021},
  publisher={IEEE}
}
```


# 그래서 메타러닝이 뭔데

+ 누가? 머신러닝 모델이 
+ 무엇을? 공부하는 법을

배우게 하는 paradigm.

즉, 주어진 문제나 문제 그룹에 가장 적합한 알고리즘(귀납적 편향)을 탐색함으로써 일반화 성능을 향상시키는 도구라는 것.

## 기준이 애매하다.

“문제에 맞는 해결책을 찾는다”는 정의가 너무 광범위함.

+ 관련은 있지만 엄연히 다른 여러 기법들까지 전부 `메타러닝`의 범주에 포함시켜 버린다는 문제가 생김. 
  + transfer, multi-task, feature-selection, and model-ensemble learning등이 메타러닝에 포함되는데, 엄연히 메타러닝이 아닌 다른 범주의 무언가다.
+ 오늘날의 메타러닝은 단순히 알고리즘을 `선택`하거나 `재사용`하는 것을 넘어, `학습하는 과정 자체`를 `최적화`하는 더 구체적인 의미로 사용되는 단어기 때문에 해당 서베이 논문에서는 아래와 같이 말하고 있음. 

# 오늘 다룰 메타러닝은 (찐)

현대 **신경망 기반**의 메타러닝!

+ end-to-end 학습 메커니즘을 검토
+ 메타러닝에는 어떤 방법론이 있는가?
+ 메타러닝의 응용분야에는 무엇이 있는가?
