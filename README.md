# 머신러닝 성능 향상을 위한 특성 공학(Feature Engineering) 파이프라인 설계 및 비교 분석

**Kaggle Titanic (`891 × 12`)** 공개 데이터셋을 활용하여 탐색적 데이터 분석(EDA)부터 데이터 누수 없는(Leak-free) 전처리 파이프라인(`Pipeline` + `ColumnTransformer`), 도메인 파생변수 생성, 변수 선택(Feature Selection), 4대 분류 모델 평가, 그리고 심화 분석(`GridSearchCV`, `SHAP`, `LIME`, `FLAML AutoML`)까지 전체 머신러닝 파이프라인을 구현하고 전략별 성능 차이를 비교·분석한 프로젝트입니다.

---

## 디렉터리 구조

```text
BigData_FE_HW/
├── data/
│   └── titanic.csv                        # Kaggle Titanic 공개 데이터셋 (891행 × 12열)
├── src/
│   ├── utils.py                           # 공통 시드(42), 한글 시각화 설정, 5대 평가지표 산출 유틸리티
│   ├── transformers.py                    # 커스텀 변환기 (TitanicFeatureEngineer, GroupMedianAgeImputer 등)
│   ├── eda.py                             # STEP 01 데이터 구조 요약 및 STEP 02 EDA 시각화 생성
│   ├── experiments.py                     # STEP 03 전처리 비교 · STEP 04 변수선택 · STEP 05 필수 실험 비교
│   ├── bonus_analysis.py                  # 가산점 심화: GridSearchCV, 다각도 중요도, SHAP, LIME, AutoML
│   ├── build_notebook.py                  # 재현용 통합 Jupyter Notebook 자동 생성 및 실행 검증
│   └── generate_pdf_report.py             # 최종 제출용 A4 PDF 보고서 컴파일러
├── figures/                               # 고해상도(300 DPI) 분석 및 비교 시각화 차트 (fig01 ~ fig13)
├── results/                               # 단계별 정량 실험 결과표 (.csv / .json)
├── titanic_fe_pipeline.ipynb              # 전체 파이프라인 실행 결과가 포함된 Jupyter Notebook
├── BigData_Feature_Engineering_Report.pdf # 최종 제출용 보고서 PDF
├── run_all.py                             # 원클릭 전체 실험 재현 스크립트
└── requirements.txt                       # Python 패키지 의존성 명세
```

---

## 빠른 재현 방법 (Reproducibility)

모든 실험은 `RANDOM_STATE = 42` 및 `StratifiedKFold(n_splits=5)`로 고정되어 동일하게 재현됩니다.

```bash
# 1. 가상환경 생성 및 패키지 설치
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 2. 전체 파이프라인(STEP 01 ~ STEP 05 + 가산점 심화 분석) 일괄 실행
python run_all.py
```

---

## 구현 범위 요약

1. **STEP 01. 데이터 준비**: Kaggle Titanic(`891 × 12`) 로드, 타겟(`Survived`) 정의, 전체 컬럼 명세표 산출
2. **STEP 02. 탐색적 데이터 분석 (EDA)**: 결측치 비율 분석, IQR 기반 이상치 탐색, Histogram · Boxplot · Heatmap · Countplot/Barplot 시각화 및 데이터 품질·불균형 진단
3. **STEP 03. 특성 공학 파이프라인 구현 (`sklearn.pipeline.Pipeline`)**:
   - **3-1 결측치 처리 비교**: `Drop NA`, `Mean`, `Median`, `Most Frequent`, `Group-Median(Pclass×Sex)`
   - **3-2 범주형 인코딩 비교**: `Label(Ordinal)`, `One-Hot`, `Target Encoding`
   - **3-3 스케일링 비교**: `No Scaling`, `StandardScaler`, `MinMaxScaler`, `RobustScaler`
   - **3-4 도메인 파생 변수 생성 (7종)**: `FamilySize`, `IsAlone`, `FarePerPerson`, `Title`, `HasCabin`, `Deck`, `AgeGroup`
4. **STEP 04. 변수 선택 (Feature Selection)**: `Random Forest Importance`, `SelectKBest(MI)`, `Correlation-based`, `RFE` 4종 구현 및 제거 전/후 성능·과적합 격차 비교
5. **STEP 05 & 섹션 5. 모델 학습 및 필수 실험 조합 비교**: `Logistic Regression`, `Random Forest`, `XGBoost`, `LightGBM` 4종 × 5대 평가지표(`Accuracy`, `Precision`, `Recall`, `F1-score`, `ROC-AUC`) 및 `Base`, `Exp-1`, `Exp-2`, `Exp-3`, `Exp-4(최적 제안)` 종합 비교
6. **섹션 8. 추가 가산점 심화 분석**:
   - `Pipeline` + `GridSearchCV` 하이퍼파라미터 최적화
   - 4대 중요도 지표(`RF 불순도`, `Permutation Importance`, `Mean |SHAP|`, `Mutual Information`) 교차 합의 시각화
   - `SHAP` 전역/국소 분석 및 `LIME` 개별 승객 예측 설명(XAI)
   - `FLAML AutoML` 비교 실험 (원본 특성 vs 도메인 파생변수 결합 비교)
