# reproduce_fda_analysis.py 프로세스 분석 문서

해당 파이썬 스크립트(`reproduce_fda_analysis.py`)는 파킨슨병(PD) 환자와 정상 대조군(HC)의 6미터 보행 모션 캡처 데이터를 바탕으로 기능적 데이터 분석(FDA, Functional Data Analysis)을 수행하는 데이터 파이프라인입니다.

전체 프로세스는 크게 **1) 데이터 로드 및 전처리**, **2) 연령/성별 보정 그룹간 차이 분석 (FDA)**, **3) 주성분 분석(FPCA) 및 임상 지표 상관관계 분석**, **4) 결과 병합 및 시각화/테이블 저장**의 4단계로 구성되어 있습니다.

아래는 각 단계별 상세 프로세스입니다.

---

## 1단계: 데이터 로드 및 전처리 (Data Loading & Preprocessing)

1. **디렉토리 설정 (`ensure_dirs`)**
   - 분석 결과를 저장할 상위 디렉토리(`OUT_ROOT`) 하위에 시각화 및 표 저장을 위한 폴더들을 자동 생성합니다. (예: `tables`, `adjusted_FDA_plots`, `fpc_loadings`, `score_distribution_by_group` 등)
2. **임상 데이터 로드 (`load_clinical`)**
   - `01_clinical_crf` 폴더 내의 CSV 파일들을 읽어옵니다.
   - 한글로 된 컬럼명을 분석하기 쉽도록 영문으로 변환합니다 (나이, 성별, 키, 몸무게, HY 단계, UPDRS, 약물 복용 여부 등).
   - 문자열 공백 제거 및 숫자형 변환 등 기본적인 데이터 정제를 수행합니다.
3. **모션 데이터 로드 및 보간 (`load_trial_curves`, `read_mot`, `interpolate_curve`)**
   - `02_raw_mocap_deid` 폴더 내의 `6m_*.mot` (6미터 보행 모션) 파일들을 탐색합니다.
   - 골반의 단순 이동(translation) 변수(`pelvis_tx`, `pelvis_ty`, `pelvis_tz`)와 `time`을 제외한 모든 관절 각도(Joint Angle) 변수들을 추출합니다.
   - 추출된 각각의 시계열 커브는 보행 시작부터 끝까지를 0~100% 구간으로 스케일링하는 **선형 보간법(Linear Interpolation)**을 거쳐 표준화된 타임 그리드(`GRID` = 101 points)로 정렬됩니다.
   - 임상 데이터에 존재하는 피험자 중 Parkinson과 Control 그룹만 필터링하고 연령/성별 결측치를 제거하여 최종 메타데이터를 구성합니다.

## 2단계: 연령 및 성별 보정 그룹 간 차이 분석 (Age/Sex Adjusted OLS)

1. **선형 회귀 모델 피팅 (`plot_adjusted_difference`, `ols_fit`, `design_matrix`)**
   - 각 변수(관절 각도) 및 시점(0~100% 중 특정 % 지점) 마다 그룹(PD 여부), 나이, 성별을 독립 변수로 하는 OLS(Ordinary Least Squares) 다중 선형 회귀 모델을 적합합니다.
   - 이를 통해 연령 및 성별의 영향을 통제한 상태에서 정상군과 파킨슨군 간의 보정된 평균(Adjusted Mean) 차이를 계산합니다.
2. **유의성 검정 및 다중 검정 보정 (`bh_fdr`, `significant_spans`)**
   - 모델에서 얻은 그룹 변수의 p-value 값을 Benjamini-Hochberg 방법론을 사용해 **FDR(False Discovery Rate)** q-value로 보정합니다.
   - 보정된 q-value가 0.05 미만인 구간(Significant span)을 도출합니다.
3. **시각화 (Adjusted FDA Plots)**
   - 두 그룹의 연령/성별 보정 평균 커브를 하나의 그래프에 그리고, 통계적으로 유의미한 차이가 있는 보행 구간(%)에 배경색을 칠하여 시각화하고 PNG 파일로 저장합니다.

## 3단계: 기능적 주성분 분석 (FPCA) 및 임상 상관성 분석

1. **잔차 커브 추출 (`age_sex_adjust_curves`)**
   - 앞선 선형 모델에서 나이와 성별의 효과를 뺀 '보정된 커브 데이터'를 생성합니다.
2. **PCA 수행 및 시각화 (`run_pca_and_plots`)**
   - 보정된 커브들에 대해 **PCA(Principal Component Analysis)**를 수행하여 기능적 주성분(FPC)을 추출합니다.
   - 제1, 2 주성분(FPC1, FPC2)의 로딩(Loading) 커브를 그래프로 그려 저장합니다.
   - 각 보행 트라이얼(Trial)별 FPC 점수를 구하고, 피험자(Subject)별로 평균을 내어 피험자 레벨의 FPC 점수(Feature)를 계산합니다.
3. **그룹 간 FPC 점수 분포 시각화 (`make_boxplot`)**
   - 산출된 주성분 점수(FPC1, FPC2 등)가 파킨슨군과 정상군 사이에서 어떻게 분포하는지 Boxplot으로 시각화합니다.
4. **PD 환자군 내 임상 지표와의 상관관계 (`make_scatter`)**
   - 파킨슨 환자 데이터만 필터링하여 FPC 점수와 임상 지표(UPDRS 점수, HY stage) 간의 **스피어만 상관계수(Spearman correlation)**를 계산하고 산점도로 그립니다.
   - 약물 복용 여부(Medication O/X)에 따른 FPC 점수 차이는 Mann-Whitney U 검정(비모수 검정)을 통해 확인하고 Boxplot을 생성합니다.

## 4단계: 결과 병합 및 최종 리포트 출력

1. **Feature Table 병합 (`merge_feature_frames`)**
   - 모든 변수(예: hip_flexion, knee_angle 등)에 대한 피험자 단위 FPC 점수들을 하나의 거대한 Feature Table(DataFrame)로 병합합니다.
2. **중증도(HY Stage) 기반 산점도 시각화 (`plot_hy_scatter`)**
   - 각 주요 관절 각도 변수들에 대해 FPC1 vs FPC2의 2D 산점도를 그립니다. 
   - 이때 점의 색상을 Control과 파킨슨병 중증도(HY stage 1~5)별로 구분하여 시각화합니다.
3. **결과 파일(CSV, JSON) 저장**
   - `dataset_summary.json`: 분석에 사용된 데이터셋 규모 (피험자 수, 트라이얼 수, 변수 수 등) 요약
   - `adjusted_FDA_group_difference_summary.csv`: FDA 분석에서 도출된 변수별 두 그룹 간 차이(최대 차이, 평균 차이, 유의미한 구간 비율 등) 요약 테이블
   - `pca_summary_by_variable.csv`: 변수별 FPC 설명력(Variance Ratio) 요약 테이블
   - `clinical_relationships_PD_only_summary.csv`: 임상 변수들과의 상관계수, p-value, fdr q-value 요약 테이블
   - `trial_level_FPC_scores.csv` & `subject_level_FDA_feature_table.csv`: 머신러닝이나 추가 통계 분석에 바로 사용할 수 있는 트라이얼/피험자 단위 최종 데이터셋 저장.
