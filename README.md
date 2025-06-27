# NSL-KDD 데이터셋을 이용한 네트워크 이상 탐지 프로젝트

## 1. 프로젝트 목표

이 프로젝트는 머신러닝을 사용하여 NSL-KDD 데이터셋의 네트워크 트래픽이 **정상(Normal)**인지 **이상(Anomaly)**인지를 분류하는 이진 분류 모델을 개발하는 것을 목표로 합니다. 개발된 모델의 성능을 평가하고, 그 결과를 바탕으로 분석 리포트를 생성합니다.

## 2. 데이터셋 소개

- **데이터셋:** NSL-KDD (Network Security Laboratory - Knowledge Discovery and Data Mining)
- **출처:** [Kaggle NSL-KDD Dataset](https://www.kaggle.com/datasets/hassan06/nslkdd)
- **설명:** 네트워크 침입 탐지 시스템(Intrusion Detection System) 연구를 위해 널리 사용되는 데이터셋입니다. 실제 네트워크 환경에서 수집된 다양한 유형의 정상 트래픽과 공격(이상) 트래픽 데이터를 포함하고 있습니다.

## 3. 프로젝트 진행 계획

1.  **환경 설정 및 데이터 준비:**
    -   필요한 라이브러리(Pandas, Scikit-learn 등)를 설치합니다.
    -   Kaggle API 또는 직접 다운로드를 통해 NSL-KDD 데이터셋을 준비합니다.

2.  **데이터 전처리 및 탐색적 데이터 분석 (EDA):**
    -   데이터셋의 구조와 통계적 특성을 파악합니다.
    -   결측치를 확인하고 처리합니다.
    -   범주형(Categorical) 피처를 수치형(Numerical)으로 변환합니다. (예: One-Hot Encoding)
    -   피처 스케일링을 통해 데이터의 범위를 조정합니다. (예: StandardScaler)
    -   정상/이상 데이터의 분포를 시각화하여 확인합니다.

3.  **머신러닝 모델 개발:**
    -   이진 분류에 적합한 모델(예: 로지스틱 회귀, 랜덤 포레스트, XGBoost 등)을 선택하여 학습합니다.
    -   학습 데이터(Train set)와 검증 데이터(Validation set)로 나누어 모델 성능을 평가하고 하이퍼파라미터를 튜닝합니다.

4.  **모델 평가:**
    -   테스트 데이터(Test set)를 사용하여 최종 모델의 성능을 평가합니다.
    -   평가 지표로는 정확도(Accuracy), 정밀도(Precision), 재현율(Recall), F1-Score, ROC Curve/AUC 등을 사용합니다.
    -    혼동 행렬(Confusion Matrix)을 통해 모델의 예측 결과를 분석합니다.

5.  **리포트 생성:**
    -   모델의 최종 성능 평가 지표를 정리합니다.
    -   중요 피처(Feature Importance)를 분석하여 어떤 특성이 이상 탐지에 큰 영향을 미치는지 확인합니다.
    -   결과를 시각화 자료와 함께 정리하여 최종 리포트를 작성합니다.

6.  **서빙**
    - gRPC


## 4. 기술 스택

-   Python
-   Pandas & NumPy
-   Scikit-learn
-   Matplotlib & Seaborn
-   Jupyter Notebook
-   gRPC