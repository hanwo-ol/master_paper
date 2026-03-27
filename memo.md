# 포스터 개요 가이드라인

``` txt
1. 서론
 - 연구 배경
 - 연구 목적
2. 재료 및 방법
 - 재료 설명  
 - 이상치 탐지 방법론 
 - 선행연구 및 차별성
3. 실험
 - 실험 방법 및 구상
 - 데이터 정제 디테일
 - 탐지 디테일
4. 실험 결과
 - 정제 결과
 - 탐지 결과
 - 간략하게 소결
  =~가 ~보다 좋았더라 기존보다 탐지 성능이 좋더라 이래서 이걸 쓰면 향후에 이런부분에 좋겠더라 하는 실험 마무리
 5. 결론
 - 요약, 제언
 - 향후 계획
```

# 코딩 규칙
- main에는 구동부 및 argparse외에는 정의하지 않아야 한다.
- 모듈별 코드파일 분리를 통해 수정 및 개선이 용이해야 한다.
- 추가되는 기능에 따라 기존 기능을 삭제하지 말고, config를 통해 변경점을 비교할 수 있게 설계해야 한다.

# 논문 파악 7 질문
(1) what is new in the work

(2) why is the work important

(3) what is the literature gap

(4) how is the gap filled

(5) what is achieved with the new method

(6) what data are used

(7) what are the limitations
