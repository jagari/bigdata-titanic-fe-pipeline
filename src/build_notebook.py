"""Generate and execute a clean, well-documented Korean Jupyter Notebook (.ipynb) for GitHub submission."""

from __future__ import annotations

import nbformat as nbf
from nbclient import NotebookClient

from src.utils import PROJECT_ROOT


def create_and_execute_notebook() -> None:
    nb = nbf.v4.new_notebook()
    cells = []

    cells.append(
        nbf.v4.new_markdown_cell(
            "# 머신러닝 성능 향상을 위한 특성 공학(Feature Engineering) 파이프라인 설계 및 비교 분석\n\n"
            "- **데이터셋**: Kaggle Titanic - Machine Learning from Disaster (`891행 × 12열`)\n"
            "- **목표**: 탐색적 데이터 분석(EDA)부터 결측치 처리, 범주형 인코딩, 스케일링, 도메인 파생변수 생성, 변수 선택(Feature Selection), 모델 학습·평가 및 설명 가능한 AI(SHAP/LIME)·AutoML 비교까지 전체 ML 파이프라인을 누수 없이(Leak-free) 구현하고 성능을 정량 비교한다."
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell("## STEP 01. 데이터 준비 (Data Preparation)")
    )
    cells.append(
        nbf.v4.new_code_cell(
            "import warnings\n"
            "warnings.filterwarnings('ignore')\n"
            "import pandas as pd\n"
            "import numpy as np\n"
            "from IPython.display import display, Image\n"
            "from src.utils import configure_korean_font, DATA_DIR, RESULTS_DIR, FIG_DIR\n\n"
            "configure_korean_font()\n"
            "df = pd.read_csv(DATA_DIR / 'titanic.csv')\n"
            "print(f'데이터셋 Shape: {df.shape} (행: {df.shape[0]}, 열: {df.shape[1]})')\n"
            "display(df.head())\n\n"
            "col_summary = pd.read_csv(RESULTS_DIR / 'step01_column_summary.csv')\n"
            "display(col_summary)"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## STEP 02. 탐색적 데이터 분석 (EDA)\n"
            "- 결측치 비율 분석 (`Cabin` 77.10%, `Age` 19.87%, `Embarked` 0.22%)\n"
            "- IQR 기반 수치형 이상치 탐색 (`Fare` 우측 왜도 4.79)\n"
            "- 필수 시각화 4종: Histogram, Boxplot, Correlation Heatmap, Countplot/Barplot"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "outlier_summary = pd.read_csv(RESULTS_DIR / 'step02_outlier_summary.csv')\n"
            "display(outlier_summary)\n"
            "for fig_name in [\n"
            "    'fig01_missing_and_target.png',\n"
            "    'fig02_histograms.png',\n"
            "    'fig03_boxplots.png',\n"
            "    'fig04_correlation_heatmap.png',\n"
            "    'fig05_categorical_bars.png',\n"
            "]:\n"
            "    display(Image(filename=str(FIG_DIR / fig_name), width=820))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## STEP 03. 특성 공학 파이프라인 구현 및 전략별 비교 실험\n"
            "- `sklearn.pipeline.Pipeline` + `ColumnTransformer` + 커스텀 `TitanicFeatureEngineer`를 결합하여 교차검증 폴드 내 데이터 누수(Data Leakage) 원천 차단\n"
            "- **3-1 결측치 처리 비교**: Drop NA, Mean, Median, Most Frequent, Group-Median(`Pclass×Sex`)\n"
            "- **3-2 범주형 인코딩 비교**: Label(Ordinal), One-Hot, Target Encoding\n"
            "- **3-3 스케일링 비교**: No Scaling, StandardScaler, MinMaxScaler, RobustScaler\n"
            "- **3-4 파생 변수 생성 (7종)**: `FamilySize`, `IsAlone`, `FarePerPerson`, `Title`, `HasCabin`, `Deck`, `AgeGroup`"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "from src.experiments import make_preprocessor\n"
            "sample_pipe = make_preprocessor(use_derived=True, num_imputer='median', encoder='onehot', scaler='robust', use_group_age=True)\n"
            "print(sample_pipe)\n\n"
            "for csv_name, title in [\n"
            "    ('step03_1_imputation.csv', '[3-1] 결측치 처리 비교'),\n"
            "    ('step03_2_encoding.csv', '[3-2] 범주형 인코딩 비교'),\n"
            "    ('step03_3_scaling.csv', '[3-3] 스케일링 비교'),\n"
            "    ('step03_4_derived_features.csv', '[3-4] 파생변수 생성 전/후 비교'),\n"
            "]:\n"
            "    print(f'\\n=== {title} ===')\n"
            "    sub = pd.read_csv(RESULTS_DIR / csv_name)\n"
            "    cols = [c for c in sub.columns if not c.startswith('train_') and 'std' not in c]\n"
            "    display(sub[cols].round(4))\n\n"
            "display(Image(filename=str(FIG_DIR / 'fig06_derived_features_eda.png'), width=820))\n"
            "display(Image(filename=str(FIG_DIR / 'fig07_step03_comparisons.png'), width=820))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## STEP 04. 변수 선택 (Feature Selection)\n"
            "- Random Forest Importance, SelectKBest(Mutual Information), Correlation-based Selection, RFE(Recursive Feature Elimination) 4종 비교 및 제거 전/후 과적합 격차(`Train AUC - Val AUC`) 분석"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "fs_df = pd.read_csv(RESULTS_DIR / 'step04_feature_selection.csv')\n"
            "display(fs_df[['변수 선택 방법', '모델', 'cv_roc_auc_mean', 'test_accuracy', 'test_f1', 'test_roc_auc', 'overfit_gap_roc_auc']].round(4))\n"
            "display(Image(filename=str(FIG_DIR / 'fig08_step04_feature_selection.png'), width=860))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## STEP 05 & 5. 필수 실험 비교표 (Base, Exp-1, Exp-2, Exp-3 + 최적 제안 Exp-4)\n"
            "- 4대 분류 모델(`Logistic Regression`, `Random Forest`, `XGBoost`, `LightGBM`) × 5대 평가지표(`Accuracy`, `Precision`, `Recall`, `F1-score`, `ROC-AUC`)"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "mand_df = pd.read_csv(RESULTS_DIR / 'step05_mandatory_comparison.csv')\n"
            "cols_show = ['실험', '결측치 처리', '인코딩', '스케일링', 'Feature Selection', '모델',\n"
            "             'test_accuracy', 'test_precision', 'test_recall', 'test_f1', 'test_roc_auc', 'cv_roc_auc_mean']\n"
            "display(mand_df[cols_show].round(4))\n"
            "display(Image(filename=str(FIG_DIR / 'fig09_mandatory_exp_comparison.png'), width=860))"
        )
    )

    cells.append(
        nbf.v4.new_markdown_cell(
            "## 8. 추가 가산점 심화 분석 (GridSearchCV, 다각도 Feature Importance, SHAP, LIME, AutoML)"
        )
    )
    cells.append(
        nbf.v4.new_code_cell(
            "gs_df = pd.read_csv(RESULTS_DIR / 'bonus_gridsearch.csv')\n"
            "automl_df = pd.read_csv(RESULTS_DIR / 'bonus_automl_comparison.csv')\n"
            "print('=== GridSearchCV 튜닝 결과 ===')\n"
            "display(gs_df)\n"
            "print('=== FLAML AutoML 비교 결과 ===')\n"
            "display(automl_df)\n"
            "for fig_name in [\n"
            "    'fig10_advanced_feature_importance.png',\n"
            "    'fig11_shap_analysis.png',\n"
            "    'fig12_lime_explanations.png',\n"
            "    'fig13_automl_and_gridsearch.png',\n"
            "]:\n"
            "    display(Image(filename=str(FIG_DIR / fig_name), width=860))"
        )
    )

    nb["cells"] = cells
    nb["metadata"]["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }

    nb_path = PROJECT_ROOT / "titanic_fe_pipeline.ipynb"
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)

    client = NotebookClient(
        nb,
        timeout=300,
        kernel_name="python3",
        resources={"metadata": {"path": str(PROJECT_ROOT)}},
    )
    client.execute()
    with open(nb_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Notebook generated and executed cleanly: {nb_path}")


if __name__ == "__main__":
    create_and_execute_notebook()
