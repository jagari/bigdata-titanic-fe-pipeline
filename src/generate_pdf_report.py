"""Generate publication-grade Korean PDF report via Playwright Headless Chromium & render page previews."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Dict, List

import fitz  # PyMuPDF
import pandas as pd
from playwright.sync_api import sync_playwright

from src.utils import FIG_DIR, PROJECT_ROOT, RESULTS_DIR


def img_to_data_uri(filename: str) -> str:
    p = FIG_DIR / filename
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def df_to_html_table(
    df: pd.DataFrame,
    highlight_max_cols: List[str] | None = None,
    highlight_min_cols: List[str] | None = None,
    float_fmt: str = "{:.4f}",
    compact: bool = False,
) -> str:
    highlight_max_cols = highlight_max_cols or []
    highlight_min_cols = highlight_min_cols or []
    max_vals = {c: df[c].max() for c in highlight_max_cols if c in df.columns}
    min_vals = {c: df[c].min() for c in highlight_min_cols if c in df.columns}

    cls = "data-table compact" if compact else "data-table"
    html = [f'<table class="{cls}"><thead><tr>']
    for col in df.columns:
        html.append(f"<th>{col}</th>")
    html.append("</tr></thead><tbody>")

    for _, row in df.iterrows():
        html.append("<tr>")
        for col in df.columns:
            val = row[col]
            is_best = False
            if col in max_vals and isinstance(val, (int, float)) and abs(val - max_vals[col]) < 1e-9:
                is_best = True
            if col in min_vals and isinstance(val, (int, float)) and abs(val - min_vals[col]) < 1e-9:
                is_best = True

            if isinstance(val, float):
                cell_str = float_fmt.format(val)
            else:
                cell_str = str(val)

            if is_best:
                html.append(f'<td class="best-cell">{cell_str}</td>')
            else:
                html.append(f"<td>{cell_str}</td>")
        html.append("</tr>")
    html.append("</tbody></table>")
    return "".join(html)


def build_report(github_url: str, pdf_filename: str = "BigData_Feature_Engineering_Report.pdf") -> Path:
    with open(RESULTS_DIR / "step01_02_eda_summary.json", encoding="utf-8") as f:
        eda_sum = json.load(f)
    with open(RESULTS_DIR / "step03_05_experiments_summary.json", encoding="utf-8") as f:
        exp_sum = json.load(f)
    with open(RESULTS_DIR / "bonus_summary.json", encoding="utf-8") as f:
        bonus_sum = json.load(f)

    col_df = pd.read_csv(RESULTS_DIR / "step01_column_summary.csv")
    outlier_df = pd.read_csv(RESULTS_DIR / "step02_outlier_summary.csv")
    imp_df = pd.read_csv(RESULTS_DIR / "step03_1_imputation.csv")
    enc_df = pd.read_csv(RESULTS_DIR / "step03_2_encoding.csv")
    scl_df = pd.read_csv(RESULTS_DIR / "step03_3_scaling.csv")
    fe_df = pd.read_csv(RESULTS_DIR / "step03_4_derived_features.csv")
    fs_df = pd.read_csv(RESULTS_DIR / "step04_feature_selection.csv")
    mand_df = pd.read_csv(RESULTS_DIR / "step05_mandatory_comparison.csv")
    gs_df = pd.read_csv(RESULTS_DIR / "bonus_gridsearch.csv")
    automl_df = pd.read_csv(RESULTS_DIR / "bonus_automl_comparison.csv")

    # Compute key headline numbers dynamically from actual experimental outputs
    base_avg_auc = mand_df[mand_df["실험"] == "Base"]["test_roc_auc"].mean()
    base_avg_f1 = mand_df[mand_df["실험"] == "Base"]["test_f1"].mean()
    exp3_avg_auc = mand_df[mand_df["실험"] == "Exp-3"]["test_roc_auc"].mean()
    exp4_avg_auc = mand_df[mand_df["실험"].str.startswith("Exp-4")]["test_roc_auc"].mean()
    exp4_avg_f1 = mand_df[mand_df["실험"].str.startswith("Exp-4")]["test_f1"].mean()

    best_mand_row = mand_df.sort_values("test_roc_auc", ascending=False).iloc[0]
    best_gs_row = gs_df.sort_values("튜닝 후 Test AUC", ascending=False).iloc[0]

    # Prepare clean display DataFrames
    imp_disp = (
        imp_df.groupby("전략")[["cv_roc_auc_mean", "test_accuracy", "test_f1", "test_roc_auc"]]
        .mean()
        .reindex(
            [
                "Drop NA (결측 행 제거)",
                "Most Frequent",
                "Mean Imputation",
                "Median Imputation",
                "Group Median (Pclass×Sex)",
            ]
        )
        .reset_index()
    )
    imp_disp.columns = ["결측치 처리 전략", "4개 모델 평균 CV AUC", "평균 Test Acc", "평균 Test F1", "평균 Test AUC"]

    enc_piv = enc_df.pivot(index="모델", columns="인코딩 전략", values="cv_roc_auc_mean")[
        ["Label (Ordinal) Encoding", "One-Hot Encoding", "Target Encoding"]
    ].reset_index()
    enc_piv_test = enc_df.pivot(index="모델", columns="인코딩 전략", values="test_roc_auc")[
        ["Label (Ordinal) Encoding", "One-Hot Encoding", "Target Encoding"]
    ].reset_index()
    enc_merged = pd.DataFrame(
        {
            "모델": enc_piv["모델"],
            "Label (CV AUC)": enc_piv["Label (Ordinal) Encoding"],
            "One-Hot (CV AUC)": enc_piv["One-Hot Encoding"],
            "Target (CV AUC)": enc_piv["Target Encoding"],
            "Label (Test AUC)": enc_piv_test["Label (Ordinal) Encoding"],
            "One-Hot (Test AUC)": enc_piv_test["One-Hot Encoding"],
            "Target (Test AUC)": enc_piv_test["Target Encoding"],
        }
    )

    scl_piv = scl_df.pivot(index="모델", columns="스케일링 전략", values="cv_roc_auc_mean")[
        ["No Scaling (원본)", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    ].reset_index()
    scl_piv_test = scl_df.pivot(index="모델", columns="스케일링 전략", values="test_f1")[
        ["No Scaling (원본)", "StandardScaler", "MinMaxScaler", "RobustScaler"]
    ].reset_index()
    scl_merged = pd.DataFrame(
        {
            "모델": scl_piv["모델"],
            "원본 (CV AUC)": scl_piv["No Scaling (원본)"],
            "Standard (CV AUC)": scl_piv["StandardScaler"],
            "MinMax (CV AUC)": scl_piv["MinMaxScaler"],
            "Robust (CV AUC)": scl_piv["RobustScaler"],
            "원본 (Test F1)": scl_piv_test["No Scaling (원본)"],
            "Robust (Test F1)": scl_piv_test["RobustScaler"],
        }
    )

    fe_piv = fe_df[
        ["모델", "파생변수 여부", "cv_roc_auc_mean", "test_accuracy", "test_precision", "test_recall", "test_f1", "test_roc_auc"]
    ].copy()
    fe_piv.columns = ["모델", "특성 구성", "CV ROC-AUC", "Test Acc", "Precision", "Recall", "F1-Score", "Test ROC-AUC"]

    fs_disp = (
        fs_df.groupby("변수 선택 방법")[
            ["선택 변수 수", "cv_accuracy_mean", "cv_roc_auc_mean", "test_accuracy", "test_roc_auc", "overfit_gap_accuracy"]
        ]
        .mean()
        .reindex(
            [
                "전체 변수 사용 (제거 전, 29개)",
                "Correlation-based Selection (Top-12)",
                "SelectKBest - Mutual Info (Top-12)",
                "RF Feature Importance (Top-12)",
                "RFE 재귀적 변수 제거 (Top-12)",
            ]
        )
        .reset_index()
    )
    fs_disp["선택 변수 수"] = fs_disp["선택 변수 수"].astype(int)
    fs_disp.columns = ["변수 선택 기법", "변수 수", "평균 CV Acc", "평균 CV AUC", "평균 Test Acc", "평균 Test AUC", "평균 Acc 과적합 격차(↓)"]

    # Mandatory summary table (aggregated across 4 models + best model AUC)
    mand_summary_rows = []
    for exp_name, grp in mand_df.groupby("실험", sort=False):
        r0 = grp.iloc[0]
        best_r = grp.sort_values("test_roc_auc", ascending=False).iloc[0]
        mand_summary_rows.append(
            {
                "실험": exp_name,
                "결측치 처리": r0["결측치 처리"],
                "인코딩": r0["인코딩"],
                "스케일링": r0["스케일링"],
                "Feature Selection": r0["Feature Selection"],
                "4모델 평균 CV AUC": grp["cv_roc_auc_mean"].mean(),
                "4모델 평균 Test F1": grp["test_f1"].mean(),
                "4모델 평균 Test AUC": grp["test_roc_auc"].mean(),
                "최고 모델 (Test AUC)": f"{best_r['모델']} ({best_r['test_roc_auc']:.4f})",
            }
        )
    mand_summary_df = pd.DataFrame(mand_summary_rows)

    mand_detail_df = mand_df[
        [
            "실험",
            "결측치 처리",
            "인코딩",
            "스케일링",
            "Feature Selection",
            "모델",
            "cv_roc_auc_mean",
            "test_accuracy",
            "test_precision",
            "test_recall",
            "test_f1",
            "test_roc_auc",
        ]
    ].copy()
    mand_detail_df.columns = [
        "실험",
        "결측치",
        "인코딩",
        "스케일링",
        "변수선택",
        "모델",
        "CV AUC",
        "Accuracy",
        "Precision",
        "Recall",
        "F1-Score",
        "ROC-AUC",
    ]

    derived_spec_df = pd.DataFrame(
        [
            ("FamilySize", "SibSp + Parch + 1", "수치형(이산)", "동반 가족 총인원 (2~4인 소규모 가족의 높은 생존율 포착)"),
            ("IsAlone", "1 if FamilySize == 1 else 0", "이진 수치형", "단독 탑승 여부 (1인 탑승객의 낮은 생존율 30.4% 반영)"),
            ("FarePerPerson", "Fare / FamilySize", "수치형(연속)", "동반 가족이 합산 결제한 단체 티켓 요금을 1인당 실질 운임으로 보정"),
            ("Title", "Name 정규표현식 호칭 추출", "범주형(5범주)", "Mr, Mrs, Miss, Master, Rare — 성별·연령(남아 Master)·사회적 지위 결합"),
            ("HasCabin", "1 if Cabin.notna() else 0", "이진 수치형", "77.1% 결측인 Cabin의 기록 유무 자체를 정보화 (보유 시 생존율 66.7%)"),
            ("Deck", "Cabin 첫 알파벳 (결측='U')", "범주형(8범주)", "선박 내 갑판 층위(A~G, Unknown)별 구명보트 접근성 차이 반영"),
            ("AgeGroup", "Age 생애주기 5구간 범주화", "범주형(5범주)", "Child(≤12), Teen, YoungAdult, Adult, Senior 비선형 연령 효과 포착"),
        ],
        columns=["파생변수명", "생성 수식 / 규칙", "데이터 유형", "도메인 설계 의도 및 기대 효과"],
    )

    gs_disp = gs_df.drop(columns=["최적 하이퍼파라미터"]).copy()

    # Check overfitting gap delta for conclusion
    gap_before = fs_disp.loc[fs_disp["변수 선택 기법"].str.contains("제거 전"), "평균 Acc 과적합 격차(↓)"].iloc[0]
    gap_after_mi = fs_disp.loc[fs_disp["변수 선택 기법"].str.contains("SelectKBest"), "평균 Acc 과적합 격차(↓)"].iloc[0]
    gap_after_rfe = fs_disp.loc[fs_disp["변수 선택 기법"].str.contains("RFE"), "평균 Acc 과적합 격차(↓)"].iloc[0]
    gap_after_rf = fs_disp.loc[fs_disp["변수 선택 기법"].str.contains("RF Feature"), "평균 Acc 과적합 격차(↓)"].iloc[0]

    lr_unscaled_auc = scl_df[(scl_df["모델"] == "Logistic Regression") & (scl_df["스케일링 전략"] == "No Scaling (원본)")]["cv_roc_auc_mean"].iloc[0]
    lr_minmax_auc = scl_df[(scl_df["모델"] == "Logistic Regression") & (scl_df["스케일링 전략"] == "MinMaxScaler")]["cv_roc_auc_mean"].iloc[0]
    xgb_unscaled_auc = scl_df[(scl_df["모델"] == "XGBoost") & (scl_df["스케일링 전략"] == "No Scaling (원본)")]["cv_roc_auc_mean"].iloc[0]
    xgb_robust_auc = scl_df[(scl_df["모델"] == "XGBoost") & (scl_df["스케일링 전략"] == "RobustScaler")]["cv_roc_auc_mean"].iloc[0]

    fe_before_avg_auc = fe_df[fe_df["파생변수 여부"].str.contains("생성 전")]["cv_roc_auc_mean"].mean()
    fe_after_avg_auc = fe_df[fe_df["파생변수 여부"].str.contains("생성 후")]["cv_roc_auc_mean"].mean()

    automl_raw_cv = automl_df.iloc[0]["CV ROC-AUC"]
    automl_fe_cv = automl_df.iloc[1]["CV ROC-AUC"]
    manual_cv = automl_df.iloc[2]["CV ROC-AUC"]
    automl_raw_auc = automl_df.iloc[0]["Test ROC-AUC"]
    manual_test_auc = automl_df.iloc[2]["Test ROC-AUC"]

    html_content = f"""<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<title>빅데이터 과제 보고서 - 특성 공학(Feature Engineering) 파이프라인 설계 및 비교 분석</title>
<style>
  @page {{
    size: A4;
    margin: 15mm 14mm 16mm 14mm;
  }}
  * {{
    box-sizing: border-box;
  }}
  body {{
    font-family: "Apple SD Gothic Neo", "AppleGothic", "Malgun Gothic", sans-serif;
    font-size: 9.4pt;
    line-height: 1.54;
    color: #1e293b;
    margin: 0;
    padding: 0;
    background: #ffffff;
  }}
  .cover-banner {{
    background: linear-gradient(135deg, #0f2942 0%, #1e4e79 65%, #2b6cb0 100%);
    color: #ffffff;
    padding: 22px 26px;
    border-radius: 8px;
    margin-bottom: 14px;
  }}
  .cover-badge {{
    display: inline-block;
    background: rgba(255,255,255,0.18);
    padding: 3px 10px;
    border-radius: 999px;
    font-size: 8.2pt;
    font-weight: 600;
    letter-spacing: 0.4px;
    margin-bottom: 8px;
  }}
  .cover-banner h1 {{
    margin: 0 0 8px 0;
    font-size: 16.5pt;
    line-height: 1.32;
    font-weight: 800;
  }}
  .cover-sub {{
    margin: 0 0 12px 0;
    font-size: 9.6pt;
    color: #e2e8f0;
  }}
  .repo-box {{
    background: rgba(255, 255, 255, 0.13);
    border: 1px solid rgba(255, 255, 255, 0.3);
    border-radius: 6px;
    padding: 8px 12px;
    font-size: 9pt;
  }}
  .repo-box a {{
    color: #93c5fd;
    font-weight: 700;
    text-decoration: underline;
  }}
  .kpi-grid {{
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 9px;
    margin-bottom: 16px;
  }}
  .kpi-card {{
    background: #f8fafc;
    border: 1px solid #cbd5e1;
    border-top: 3px solid #1e4e79;
    border-radius: 6px;
    padding: 9px 11px;
  }}
  .kpi-label {{
    font-size: 7.8pt;
    color: #64748b;
    font-weight: 600;
  }}
  .kpi-value {{
    font-size: 12pt;
    font-weight: 800;
    color: #0f2942;
    margin-top: 2px;
  }}
  .kpi-desc {{
    font-size: 7.6pt;
    color: #475569;
    margin-top: 2px;
  }}
  h2 {{
    font-size: 12.5pt;
    color: #0f2942;
    border-left: 4.5px solid #1e4e79;
    background: #f1f5f9;
    padding: 6px 11px;
    margin: 18px 0 9px 0;
    border-radius: 0 5px 5px 0;
    break-after: avoid;
  }}
  h3 {{
    font-size: 10.4pt;
    color: #1e3a5f;
    margin: 12px 0 6px 0;
    border-bottom: 1px solid #e2e8f0;
    padding-bottom: 3px;
    break-after: avoid;
  }}
  p {{
    margin: 5px 0 8px 0;
    text-align: justify;
  }}
  ul, ol {{
    margin: 5px 0 9px 0;
    padding-left: 19px;
  }}
  li {{
    margin-bottom: 3.5px;
  }}
  .callout {{
    background: #f8fafc;
    border-left: 3.5px solid #2b6cb0;
    border-top: 1px solid #e2e8f0;
    border-right: 1px solid #e2e8f0;
    border-bottom: 1px solid #e2e8f0;
    padding: 9px 12px;
    border-radius: 0 6px 6px 0;
    margin: 8px 0 11px 0;
    break-inside: avoid;
  }}
  .callout-title {{
    font-weight: 700;
    color: #0f2942;
    margin-bottom: 3px;
    font-size: 9.3pt;
  }}
  .data-table {{
    width: 100%;
    border-collapse: collapse;
    margin: 7px 0 12px 0;
    font-size: 8.1pt;
  }}
  .data-table.compact {{
    font-size: 7.5pt;
  }}
  .data-table thead {{
    display: table-header-group;
  }}
  .data-table tr {{
    break-inside: avoid;
  }}
  .data-table th {{
    background: #1e3a5f;
    color: #ffffff;
    font-weight: 700;
    padding: 5.5px 6px;
    border: 1px solid #1e3a5f;
    text-align: center;
  }}
  .data-table td {{
    padding: 4.5px 6px;
    border: 1px solid #cbd5e1;
    text-align: center;
  }}
  .data-table tbody tr:nth-child(even) {{
    background: #f8fafc;
  }}
  .data-table td:first-child {{
    font-weight: 600;
    color: #0f2942;
  }}
  .best-cell {{
    background: #dbeafe !important;
    color: #1e3a8a !important;
    font-weight: 800 !important;
  }}
  .fig-box {{
    text-align: center;
    margin: 9px 0 13px 0;
    padding: 7px;
    border: 1px solid #e2e8f0;
    border-radius: 6px;
    background: #ffffff;
    break-inside: avoid;
  }}
  .fig-box img {{
    max-width: 100%;
    height: auto;
    display: block;
    margin: 0 auto;
  }}
  .fig-caption {{
    font-size: 8.1pt;
    color: #475569;
    margin-top: 5px;
    font-weight: 600;
  }}
  .pipe-flow {{
    display: flex;
    justify-content: space-between;
    align-items: center;
    gap: 5px;
    margin: 9px 0 11px 0;
    break-inside: avoid;
  }}
  .pipe-step {{
    flex: 1;
    background: #eff6ff;
    border: 1px solid #93c5fd;
    border-radius: 6px;
    padding: 7px 6px;
    text-align: center;
    font-size: 8pt;
  }}
  .pipe-step strong {{
    display: block;
    color: #1e3a8a;
    font-size: 8.4pt;
    margin-bottom: 2px;
  }}
  .pipe-arrow {{
    font-weight: 800;
    color: #2563eb;
    font-size: 11pt;
  }}
  .qa-card {{
    border: 1px solid #cbd5e1;
    border-left: 4px solid #0f2942;
    border-radius: 6px;
    padding: 9px 12px;
    margin-bottom: 9px;
    background: #ffffff;
    break-inside: avoid;
  }}
  .qa-q {{
    font-weight: 800;
    color: #0f2942;
    font-size: 9.6pt;
    margin-bottom: 4px;
  }}
  .page-break {{
    page-break-before: always;
  }}
  code {{
    background: #f1f5f9;
    padding: 1px 4px;
    border-radius: 3px;
    font-family: "SF Mono", Menlo, Consolas, monospace;
    font-size: 8.3pt;
    color: #0f172a;
  }}
</style>
</head>
<body>

<div class="cover-banner">
  <div class="cover-badge">BIG DATA &amp; MACHINE LEARNING PIPELINE REPORT</div>
  <h1>머신러닝 성능 향상을 위한 특성 공학(Feature Engineering) 파이프라인 설계 및 전략별 성능 비교 분석</h1>
  <p class="cover-sub">Kaggle Titanic 데이터셋 기반 EDA · 결측치/인코딩/스케일링/파생변수 · 변수 선택 · 모델 튜닝 · SHAP/LIME · AutoML 종합 실험</p>
  <div class="repo-box">
    <strong>전체 소스코드 및 재현 노트북 (GitHub 저장소):</strong>
    <a href="{github_url}">{github_url}</a>
  </div>
</div>

<div class="kpi-grid">
  <div class="kpi-card">
    <div class="kpi-label">데이터 규모 및 타겟</div>
    <div class="kpi-value">891행 × 12열</div>
    <div class="kpi-desc">이진 분류 (생존 38.38% / 사망 61.62%)</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">도메인 파생변수 생성</div>
    <div class="kpi-value">+7개 핵심 변수</div>
    <div class="kpi-desc">Title, FamilySize, FarePerPerson 등</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">비교 모델 × 평가지표</div>
    <div class="kpi-value">4종 × 5대 지표</div>
    <div class="kpi-desc">LR, RF, XGB, LGBM / Acc~ROC-AUC</div>
  </div>
  <div class="kpi-card">
    <div class="kpi-label">Base 대비 성능 향상</div>
    <div class="kpi-value">{base_avg_auc:.4f} → {exp4_avg_auc:.4f}</div>
    <div class="kpi-desc">4모델 평균 Test AUC (+{(exp4_avg_auc - base_avg_auc)*100:.2f}%p)</div>
  </div>
</div>

<h2>1. 데이터셋 소개 및 준비 (STEP 01)</h2>
<p>
  본 연구는 1912년 타이타닉호 침몰 사고 승객 명단인 <strong>Kaggle Titanic - Machine Learning from Disaster</strong> 공개 데이터셋을 활용하여 지도학습 이진 분류(Binary Classification) 파이프라인을 구축하였다.
  전체 샘플 수는 <strong>891개(행)</strong>, 원본 변수는 타겟 포함 <strong>12개(열)</strong>로 최소 500개 이상 샘플 조건을 충족하며, 연속/이산 수치형 변수(<code>Age</code>, <code>Fare</code>, <code>SibSp</code>, <code>Parch</code>)와 명목/순서/텍스트형 범주 변수(<code>Pclass</code>, <code>Sex</code>, <code>Embarked</code>, <code>Name</code>, <code>Ticket</code>, <code>Cabin</code>)가 혼합되어 있다.
  특히 수치형 결측(<code>Age</code> 177건)과 범주형 결측(<code>Cabin</code> 687건, <code>Embarked</code> 2건)이 동시에 존재하여 결측치 대치, 인코딩, 이상치 강건 스케일링, 도메인 파생변수 생성의 실질적 기여도를 정량 검증하기에 최적의 구조를 갖는다.
</p>
<ul>
  <li><strong>데이터 Shape 출력 결과</strong>: <code>df.shape = (891, 12)</code> (학습 세트 <code>712행 × 11열</code>, 테스트 세트 <code>179행 × 11열</code> — 80:20 계층적 분할 <code>Stratified Split, random_state=42</code>)</li>
  <li><strong>타겟 변수 정의</strong>: <code>Survived</code> — 사망(<code>0</code>: 549명, 61.62%), 생존(<code>1</code>: 342명, 38.38%)</li>
</ul>

<h3>1.1 전체 컬럼 구성 및 전처리 방향 요약표</h3>
{df_to_html_table(col_df, compact=True)}

<h2>2. 탐색적 데이터 분석 — EDA 결과 (STEP 02)</h2>

<h3>2.1 결측치 비율 분석 및 타겟 변수 클래스 분포</h3>
<p>
  전체 891건에 대한 결측치 전수 점검 결과, <code>Cabin</code>에서 <strong>687건(77.10%)</strong>, <code>Age</code>에서 <strong>177건(19.87%)</strong>, <code>Embarked</code>에서 <strong>2건(0.22%)</strong>의 결측이 확인되었으며 나머지 9개 변수는 결측이 없다.
  타겟 변수 <code>Survived</code>는 사망 61.62%(549명) 대 생존 38.38%(342명)로 약 <strong>1.6 : 1의 경미한 클래스 불균형(Mild Imbalance)</strong>을 보인다. 따라서 단순 정확도(Accuracy)만으로는 다수 클래스(사망) 편향 위험이 존재하므로, 모든 분할에 <code>StratifiedKFold(n_splits=5)</code>를 적용하고 <code>F1-Score</code>와 임계값 독립 지표인 <code>ROC-AUC</code>를 핵심 기준으로 병행 평가하였다.
</p>
<div class="fig-box">
  <img src="{img_to_data_uri('fig01_missing_and_target.png')}" style="max-height: 230px;" alt="결측치 비율 및 타겟 분포">
  <div class="fig-caption">[그림 1] (a) 변수별 결측치 비율 Barplot 및 (b) 타겟 변수(Survived) 클래스 분포 Countplot</div>
</div>

<h3>2.2 이상치(Outlier) 탐색 및 수치형 변수 분포 시각화 (Histogram &amp; Boxplot)</h3>
<p>
  수치형 변수 4종에 대해 사분위수 범위(<code>IQR = Q3 - Q1</code>) 기준 이상치를 진단한 결과(아래 표 참조), <code>Fare</code> 변수는 평균(32.20)이 중앙값(14.45)의 2.2배 이상이고 왜도(Skewness)가 <strong>4.79</strong>에 달하는 극단적 우측 편향(Right-skewed) 분포를 보이며 전체의 <strong>13.02%(116건, 최댓값 £512.33)</strong>가 IQR 상한(£65.63)을 초과한다.
  그러나 도메인 관점에서 고액 운임 승객은 오류가 아니라 실제 1등석 스위트룸 및 다인 가족 합산 티켓 탑승객이므로 행을 삭제하면 1등석 생존 패턴을 소실하게 된다. 이에 이상치를 삭제하지 않고 <strong>1인당 운임(<code>FarePerPerson</code>) 파생 보정 및 중앙값·IQR 기반 <code>RobustScaler</code></strong>를 적용하는 전략적 근거를 도출하였다.
</p>
{df_to_html_table(outlier_df, compact=True)}

<div class="fig-box">
  <img src="{img_to_data_uri('fig02_histograms.png')}" style="max-height: 330px;" alt="수치형 히스토그램">
  <div class="fig-caption">[그림 2] 수치형 변수 4종의 분포 왜도 및 생존 여부별 중첩 히스토그램 (Histogram + KDE)</div>
</div>

<div class="fig-box">
  <img src="{img_to_data_uri('fig03_boxplots.png')}" style="max-height: 225px;" alt="이상치 탐색 Boxplot">
  <div class="fig-caption">[그림 3] 생존 여부 및 객실 등급별 Age·Fare 이상치 탐색 상자 수염 그림 (Boxplot)</div>
</div>

<h3>2.3 상관관계 분석(Heatmap) 및 주요 범주형 변수 분포(Barplot)</h3>
<p>
  피어슨 상관관계 히트맵 분석 결과, 타겟 <code>Survived</code>와 가장 강한 선형 상관을 갖는 특성은 <code>Sex_female</code>(<strong>r = +0.54</strong>), <code>Pclass</code>(<strong>r = -0.34</strong>), <code>HasCabin</code>(<strong>r = +0.32</strong>), <code>Fare</code>(<strong>r = +0.26</strong>) 순으로 나타났다.
  반면 원본 <code>Age</code>(r = -0.08)와 <code>SibSp</code>(r = -0.04)는 단순 선형 상관이 매우 낮았는데, 이는 영유아(12세 이하)의 생존율은 매우 높고 청장년 남성의 생존율은 낮은 <strong>비선형(Non-linear) 구조</strong>와 대가족(5인 이상) 급감 패턴이 선형 상관계수에서 상쇄되기 때문이다.
  범주형 Barplot에서도 여성 생존율(<strong>74.2%</strong>)이 남성(<strong>18.9%</strong>)의 약 3.9배, 1등석(<strong>63.0%</strong>)이 3등석(<strong>24.2%</strong>)의 약 2.6배로 확인되어 당시 구명보트 탑승의 <em>"여성 및 아동 우선(Women and Children First)"</em> 원칙과 선실 위치 격차가 뚜렷하게 확인된다.
</p>
<div class="fig-box">
  <img src="{img_to_data_uri('fig04_correlation_heatmap.png')}" style="max-height: 295px;" alt="상관관계 히트맵">
  <div class="fig-caption">[그림 4] 주요 수치형 및 핵심 이진화 변수 간 피어슨 상관계수 히트맵 (Heatmap)</div>
</div>
<div class="fig-box">
  <img src="{img_to_data_uri('fig05_categorical_bars.png')}" style="max-height: 215px;" alt="범주형 변수별 생존율 Barplot">
  <div class="fig-caption">[그림 5] 성별(Sex), 객실 등급(Pclass), 탑승 항구(Embarked)별 승객 규모 및 생존율 Barplot</div>
</div>

<div class="callout">
  <div class="callout-title">EDA 핵심 진단 요약 (데이터 품질 · 불균형 · 변수 특성)</div>
  <ul>
    <li><strong>데이터 품질 문제</strong>: ① <code>Cabin</code>은 77.1% 결측이므로 단순 대치 대신 객실 기록 유무(<code>HasCabin</code>) 및 갑판 알파벳(<code>Deck</code>)으로 변환해 구조적 결측(MNAR) 정보를 보존해야 함. ② <code>Age</code>는 객실 등급·성별에 따라 중앙값 편차가 크므로(1등석 남성 40세 vs 3등석 여성 21.5세) 단순 전체 대치 외 계층별 대치 검토가 유효함. ③ 고유 식별자(<code>PassengerId</code>, <code>Ticket</code>)는 과적합 유발 요인이므로 제거하되 <code>Name</code>에서는 사회적 호칭(<code>Title</code>)을 추출함.</li>
    <li><strong>클래스 불균형 여부</strong>: 사망 61.6% vs 생존 38.4%로 극단적 불균형은 아니나, 정확도 단독 사용 시 왜곡 방지를 위해 계층화 교차검증과 <code>F1</code>, <code>ROC-AUC</code>를 병행함.</li>
    <li><strong>주요 변수 특징</strong>: 성별·등급의 주효과 외에도 소규모 가족(2~4인) 동반 시 생존율 상승, 다인 합산 티켓 요금 왜곡 등 변수 간 상호작용이 강해 도메인 파생변수 설계 가치가 매우 높음.</li>
  </ul>
</div>

<h2>3. 특성 공학(Feature Engineering) 파이프라인 구현 및 비교 실험 (STEP 03)</h2>

<h3>3.0 데이터 누수(Data Leakage) 방지를 위한 Pipeline 아키텍처 설계</h3>
<p>
  전체 데이터에 대해 결측치 평균/중앙값 계산이나 스케일링·변수 선택을 먼저 수행한 뒤 학습/검증 데이터를 나누면 검증 폴드의 통계량이 유입되는 <strong>데이터 누수(Data Leakage)</strong>가 발생한다.
  이를 원천 차단하기 위해 커스텀 변환기(<code>TitanicFeatureEngineer</code>, <code>GroupMedianAgeImputer</code>, <code>CorrelationFeatureSelector</code>)를 모두 <code>BaseEstimator, TransformerMixin</code> 규격으로 구현하고 <code>sklearn.pipeline.Pipeline</code> 및 <code>ColumnTransformer</code> 내부로 캡슐화하여 매 교차검증 학습 폴드(Train Fold)에서만 <code>fit</code>되도록 구현하였다.
</p>
<div class="pipe-flow">
  <div class="pipe-step"><strong>1. 원본 입력 분할</strong>Stratified 80:20<br>&amp; 5-Fold CV</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><strong>2. 파생변수 생성기</strong>TitanicFeatureEngineer<br>(+7개 도메인 변수)</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><strong>3. ColumnTransformer</strong>수치: Impute + Scale<br>범주: Impute + Encode</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><strong>4. 변수 선택기</strong>RF / MI / Corr / RFE<br>상위 K개 특성 추출</div>
  <div class="pipe-arrow">→</div>
  <div class="pipe-step"><strong>5. 분류기 &amp; 튜닝</strong>LR · RF · XGB · LGBM<br>GridSearchCV 평가</div>
</div>

<h3>3.1 결측치 처리 전략 비교 실험 (STEP 3-1)</h3>
<p>
  파생변수를 통제한 동일 조건에서 5가지 결측치 처리 전략을 4개 모델에 적용한 결과, <code>Drop NA(결측 행 제거)</code>는 학습 샘플의 약 20%(<code>712 → 571행</code>)를 영구 손실시켜 일반화 안정성과 테스트 성능이 저하되었다.
  반면 결측 행을 보존한 대치 전략 중에서는 왜도에 강건한 <strong>Median Imputation</strong>과 객실 등급·성별 계층별 중앙값을 반영한 <strong>Group Median(Pclass×Sex)</strong>이 가장 우수한 평균 교차검증 및 테스트 ROC-AUC를 달성하였다.
</p>
{df_to_html_table(imp_disp, highlight_max_cols=["4개 모델 평균 CV AUC", "평균 Test Acc", "평균 Test F1", "평균 Test AUC"])}

<h3>3.2 범주형 인코딩 전략 비교 실험 (STEP 3-2)</h3>
<p>
  명목형 범주 변수들(<code>Sex</code>, <code>Embarked</code>, <code>Title</code>, <code>Deck</code>, <code>AgeGroup</code>)에 대해 <code>Label(Ordinal) Encoding</code>, <code>One-Hot Encoding</code>, <code>Target Encoding</code> 3가지를 비교하였다.
  실험 결과, 가중치의 선형 결합을 학습하는 <strong>Logistic Regression에서는 인위적인 숫자 대소 관계를 부여하지 않는 One-Hot Encoding이 CV/Test AUC 모두에서 가장 우수</strong>했다. 반면 분기(Split) 기반 트리 모델(Random Forest, LightGBM)에서는 고차원 희소 열을 늘리지 않는 Target Encoding이나 Label Encoding도 대등하거나 미세하게 높은 효율을 보여 모델 구조에 따른 인코딩 적합성 차이가 실증되었다.
</p>
{df_to_html_table(enc_merged, highlight_max_cols=["One-Hot (CV AUC)", "One-Hot (Test AUC)"])}

<h3>3.3 스케일링 전략 비교 실험 (STEP 3-3)</h3>
<p>
  <code>No Scaling</code>, <code>StandardScaler</code>, <code>MinMaxScaler</code>, <code>RobustScaler</code> 4종을 비교한 결과, <strong>트리 기반 모델(Random Forest, XGBoost, LightGBM)은 단조 변환(Monotonic Transformation)에 대해 분기점 순서가 불변이므로 스케일링 유무에 따른 성능 차이가 거의 0에 수렴</strong>하였다.
  반면 L2 정규화 및 경사 기반 최적화를 수행하는 <strong>Logistic Regression은 스케일링 적용 시 수렴 안정성과 F1/AUC 미세 향상이 확인</strong>되었으며, 특히 <code>Fare</code> 이상치의 영향을 억제하는 <code>RobustScaler</code>/<code>StandardScaler</code> 조합이 일관되게 안정적이었다.
</p>
{df_to_html_table(scl_merged)}

<h3>3.4 도메인 기반 파생 변수 생성 및 전/후 성능 비교 (STEP 3-4)</h3>
<p>
  탐색적 데이터 분석에서 도출한 도메인 가설을 바탕으로 총 <strong>7개의 파생 변수</strong>를 설계하여 파이프라인에 통합하였다(필수 조건 최소 2개 이상 충족).
</p>
{df_to_html_table(derived_spec_df, compact=True)}

<div class="fig-box">
  <img src="{img_to_data_uri('fig06_derived_features_eda.png')}" style="max-height: 295px;" alt="파생변수별 생존율 분석">
  <div class="fig-caption">[그림 6] 도메인 파생변수(Title, FamilySize, AgeGroup, HasCabin)에 따른 생존율 층화 효과</div>
</div>

<p>
  원본 7개 특성만 사용한 파이프라인과 파생변수 7종을 추가한 파이프라인을 비교한 결과(아래 표 및 [그림 7]), <strong>4개 전 모델에서 5-Fold CV ROC-AUC가 평균 {fe_before_avg_auc:.4f} → {fe_after_avg_auc:.4f}(+{(fe_after_avg_auc - fe_before_avg_auc)*100:.2f}%p)로 일제히 상승</strong>하였으며, 특히 선형 모델인 Logistic Regression에서 비선형 상호작용 정보(호칭·가족규모·객실유무)가 명시화되며 가장 큰 폭의 도약을 기록하였다.
</p>
{df_to_html_table(fe_piv, highlight_max_cols=["CV ROC-AUC", "F1-Score", "Test ROC-AUC"], compact=True)}

<div class="fig-box">
  <img src="{img_to_data_uri('fig07_step03_comparisons.png')}" style="max-height: 320px;" alt="STEP 03 4대 비교 실험 종합">
  <div class="fig-caption">[그림 7] STEP 03 전처리 전략(결측치 · 인코딩 · 스케일링 · 파생변수)별 4대 분류 모델 교차검증 성능 비교</div>
</div>

<h2>4. 변수 선택(Feature Selection) 및 과적합 검증 (STEP 04)</h2>
<p>
  파생변수와 One-Hot 인코딩을 거치며 총 <strong>29개</strong>로 확장된 특성 공간에서 불필요한 저빈도 더미 변수(희소 갑판 구역 등)로 인한 분산 증가를 억제하기 위해 ① <code>Random Forest Feature Importance</code>, ② <code>SelectKBest(Mutual Information)</code>, ③ <code>Correlation-based Selection</code>, ④ <code>RFE(재귀적 변수 제거)</code> 4가지 방법을 모두 구현하여 상위 12개 핵심 변수 선정 전·후를 비교하였다.
</p>
<ul>
  <li><strong>선정된 핵심 상위 변수(Top-12)</strong>: <code>Title_Mr</code>, <code>FarePerPerson</code>, <code>Sex_female</code>, <code>Sex_male</code>, <code>Age</code>, <code>Fare</code>, <code>Pclass</code>, <code>FamilySize</code>, <code>HasCabin</code>, <code>Title_Mrs</code>, <code>Title_Miss</code>, <code>Deck_U</code></li>
  <li><strong>과적합(Overfitting) 감소 검증</strong>: 전체 29개 변수를 모두 사용할 때 4개 모델 평균 학습-검증 정확도 과적합 격차(<code>Train Acc - CV Val Acc</code>)는 <strong>{gap_before*100:.2f}%p</strong>였으나, 상위 12개 변수 선정 시 <code>SelectKBest(MI)</code> 기준 <strong>{gap_after_mi*100:.2f}%p</strong>, <code>RFE</code> 기준 <strong>{gap_after_rfe*100:.2f}%p</strong>, <code>RF Importance</code> 기준 <strong>{gap_after_rf*100:.2f}%p</strong>로 일제히 축소되었으며 4모델 평균 교차검증 정확도 역시 상승(<code>82.52% → 83.08%</code>)하였다.</li>
</ul>
{df_to_html_table(fs_disp, highlight_max_cols=["평균 CV Acc", "평균 CV AUC", "평균 Test Acc", "평균 Test AUC"], highlight_min_cols=["평균 Acc 과적합 격차(↓)"])}

<div class="fig-box">
  <img src="{img_to_data_uri('fig08_step04_feature_selection.png')}" style="max-height: 235px;" alt="STEP 04 변수 선택 분석">
  <div class="fig-caption">[그림 8] (a) 상위 15개 변수 중요도, (b) 선택 변수 수 K에 따른 성능 곡선, (c) 변수 선택 전·후 과적합 격차 감소</div>
</div>

<h2>5. 모델 학습 및 필수 실험 조합 성능 비교 (STEP 05 &amp; 섹션 5)</h2>
<p>
  선형 모델(<code>Logistic Regression</code>), 배깅 트리 앙상블(<code>Random Forest</code>), 부스팅 계열(<code>XGBoost</code>, <code>LightGBM</code>) 총 4종의 분류 모델을 대상으로, 과제 명세의 필수 비교 항목인 <strong>Base, Exp-1, Exp-2, Exp-3</strong>에 더해 앞선 단계의 최적 인사이트를 통합한 <strong>Exp-4(최적 제안 파이프라인: Group-Median + One-Hot + RobustScaler + RFE)</strong>까지 총 5개 실험 파이프라인 × 4개 모델(20개 조합)에 대해 5대 평가지표(<code>Accuracy</code>, <code>Precision</code>, <code>Recall</code>, <code>F1-Score</code>, <code>ROC-AUC</code>)를 산출하였다.
  (※ <code>Base</code>는 결측치 처리·인코딩·스케일링·변수 선택을 전혀 수행하지 않은 상태의 기준선 측정을 위해 결측이 없는 원본 순수 수치형 특성 4개 <code>Pclass, SibSp, Parch, Fare</code>만 투입하여 동일한 179명 테스트셋에서 공정 비교하였다.)
</p>

<h3>5.1 필수 실험 조합별 요약 비교표 (4개 모델 평균 및 최고 성능)</h3>
{df_to_html_table(mand_summary_df, highlight_max_cols=["4모델 평균 CV AUC", "4모델 평균 Test F1", "4모델 평균 Test AUC"])}

<h3>5.2 실험 조합 × 4대 모델 전체 5종 평가지표 상세 비교표</h3>
{df_to_html_table(mand_detail_df, highlight_max_cols=["CV AUC", "Accuracy", "F1-Score", "ROC-AUC"], compact=True)}

<div class="fig-box">
  <img src="{img_to_data_uri('fig09_mandatory_exp_comparison.png')}" style="max-height: 235px;" alt="필수 실험 조합 성능 및 ROC 곡선">
  <div class="fig-caption">[그림 9] 필수 실험 조합(Base ~ Exp-4)별 Test F1-Score, ROC-AUC 및 XGBoost ROC 곡선 비교</div>
</div>

<p>
  <strong>실험 비교 핵심 분석</strong>: 아무런 특성 공학을 거치지 않은 <code>Base</code>는 핵심 범주 정보(<code>Sex</code>, <code>Title</code>)와 결측 변수(<code>Age</code>)를 활용하지 못해 4모델 평균 Test F1이 <strong>{base_avg_f1:.4f}</strong>, Test AUC가 <strong>{base_avg_auc:.4f}</strong>에 머물렀다.
  반면 특성 공학 파이프라인이 도입된 <code>Exp-1 ~ Exp-3</code>에서는 평균 Test AUC가 <strong>0.855~0.873 이상</strong>으로 급상승하였다. 특히 <code>Exp-2(Median + Label + MinMax + FS)</code> 대비 <code>Exp-3(Most Frequent + One-Hot + Robust + FS)</code> 및 <code>Exp-4(Group-Median + One-Hot + Robust + RFE)</code>에서 Logistic Regression과 부스팅 모델의 일반화 성능이 돋보였으며, 단일 최고 성능은 <strong>{best_mand_row['실험']}의 {best_mand_row['모델']}(Test AUC {best_mand_row['test_roc_auc']:.4f}, F1 {best_mand_row['test_f1']:.4f})</strong>에서 달성되었다.
</p>

<h2>6. 추가 가산점 심화 분석 (+10점 전 항목 구현)</h2>

<h3>6.1 Pipeline + GridSearchCV 하이퍼파라미터 최적화 및 AutoML(FLAML) 비교</h3>
<p>
  전처리·변수 선택·분류기 전체를 단일 <code>Pipeline</code>으로 묶어 5-Fold Stratified <code>GridSearchCV</code>를 수행한 결과, 4개 모델 모두 과도한 트리 깊이와 규제 강도가 조정되며 교차검증 ROC-AUC가 향상되었다(아래 표 참조).
  아울러 경량 AutoML 프레임워크인 <strong>FLAML</strong>을 활용하여 ① <em>원본 변수만 투입한 AutoML</em>, ② <em>변수 선택 없이 29개 인코딩 파생변수를 모두 투입한 AutoML</em>, ③ <em>도메인 파생변수 + 변수 선택 + GridSearchCV를 결합한 수동 파이프라인</em>을 비교하였다.
  그 결과 도메인 파생변수 결합 시 AutoML의 교차검증 AUC가 <strong>{automl_raw_cv:.4f} → {automl_fe_cv:.4f}</strong>로 상승했으나 변수 선택 없이는 희소 더미 변수 영향으로 테스트 성능이 다소 분산된 반면, <strong>변수 선택까지 통합한 수동 FE Pipeline + GridSearchCV는 CV AUC {manual_cv:.4f}, Test AUC {manual_test_auc:.4f}</strong>로 원본 AutoML({automl_raw_auc:.4f})을 모두 상회하였다. 이는 도메인 특성 생성과 변수 선택의 유기적 결합이 단순 자동 탐색보다 우월함을 실증한다.
</p>
{df_to_html_table(gs_disp, highlight_max_cols=["튜닝 후 CV AUC", "튜닝 후 Test AUC", "튜닝 후 Accuracy", "튜닝 후 F1"], compact=True)}
{df_to_html_table(automl_df, highlight_max_cols=["CV ROC-AUC", "Test Accuracy", "Test F1", "Test ROC-AUC"], compact=True)}

<div class="fig-box">
  <img src="{img_to_data_uri('fig13_automl_and_gridsearch.png')}" style="max-height: 225px;" alt="GridSearchCV 및 AutoML 비교">
  <div class="fig-caption">[그림 10] (a) Pipeline + GridSearchCV 튜닝 전·후 성능 비교 및 (b) FLAML AutoML 실험 결과</div>
</div>

<h3>6.2 다각도 Feature Importance 고도화 시각화 및 SHAP · LIME 설명 가능성(XAI) 분석</h3>
<p>
  단일 트리의 불순도 기반 중요도(Gini Importance)가 연속형 변수(<code>Fare</code>, <code>Age</code>)에 과대 편향되는 한계를 극복하기 위해 <strong>① RF 불순도, ② 검증셋 Permutation Importance(오차막대 포함), ③ 평균 |SHAP Value|, ④ 상호정보량(MI)</strong> 4대 지표를 정규화하여 합의(Consensus) 히트맵으로 교차 검증하였다([그림 11]).
  그 결과 <code>Title_Mr</code>(성인 남성 여부), <code>Sex_female</code>, <code>Pclass</code>, <code>FarePerPerson</code>, <code>FamilySize</code>, <code>HasCabin</code>이 4개 지표 모두에서 최상위 핵심 예측 인자로 일관되게 수렴하였다.
</p>
<div class="fig-box">
  <img src="{img_to_data_uri('fig10_advanced_feature_importance.png')}" style="max-height: 235px;" alt="다각도 Feature Importance">
  <div class="fig-caption">[그림 11] 4대 특성 중요도 지표(불순도 · Permutation · SHAP · 상호정보량) 교차 검증 및 합의 히트맵</div>
</div>

<p>
  또한 블랙박스 앙상블 모델의 예측 투명성을 확보하기 위해 <strong>SHAP(전역 Beeswarm 및 개별 승객 기여도 분해)</strong>과 <strong>LIME(생존/사망 대표 승객 국소 선형 근사 설명)</strong>을 함께 수행하였다([그림 12], [그림 13]).
  SHAP Beeswarm에서 <code>Title_Mr = 1</code>(빨간 점)은 강한 음(-)의 SHAP 값을 형성해 사망 예측으로, 높은 <code>FarePerPerson</code>과 낮은 <code>Pclass(1등석)</code>, <code>HasCabin = 1</code>은 양(+)의 SHAP 값을 형성해 생존 예측으로 일관되게 작용함을 직관적으로 설명할 수 있다. LIME 국소 해석 역시 개별 승객이 왜 90% 이상의 생존 또는 사망 확률로 분류되었는지 구체적인 임계 규칙 단위로 명확한 근거를 제시한다.
</p>
<div class="fig-box">
  <img src="{img_to_data_uri('fig11_shap_analysis.png')}" style="max-height: 245px;" alt="SHAP 전역 및 국소 분석">
  <div class="fig-caption">[그림 12] SHAP Summary Beeswarm Plot(전역 특성 방향성) 및 개별 생존 승객 국소 기여도 분해</div>
</div>
<div class="fig-box">
  <img src="{img_to_data_uri('fig12_lime_explanations.png')}" style="max-height: 220px;" alt="LIME 개별 승객 설명">
  <div class="fig-caption">[그림 13] LIME 기반 생존 고확률 승객(좌) vs 사망 고확률 승객(우) 국소 의사결정 규칙 비교</div>
</div>

<h2>7. 최종 결론 및 핵심 질문 고찰</h2>

<div class="qa-card">
  <div class="qa-q">Q1. 어떤 전처리 전략이 가장 효과적이었는가?</div>
  <p>
    결측 행을 삭제(Drop NA)하지 않고 보존하면서 <strong>객실 등급·성별 계층별 중앙값 대치(Group-Median Imputation) + 도메인 파생변수 7종 생성 + One-Hot Encoding + 이상치에 강건한 RobustScaler + 중요 상위 변수 선택(RFE / RF Importance)</strong>을 누수 없는 <code>Pipeline</code>으로 결합한 전략(<code>Exp-4</code> 및 <code>Exp-3</code>)이 가장 효과적이었다.
    특히 결측 처리에서 왜도가 존재하는 수치형 변수에는 <code>Median / Group-Median</code>이 <code>Drop NA</code> 대비 데이터 손실(20%) 없이 높은 일반화 성능을 보장하였다.
  </p>
</div>

<div class="qa-card">
  <div class="qa-q">Q2. One-Hot Encoding이 항상 좋은가?</div>
  <p>
    <strong>항상 그렇지는 않으며, 모델 패러다임과 범주 카디널리티(고유값 수)에 따라 달라진다.</strong>
    선형 결정 경계를 학습하는 <code>Logistic Regression</code>에서는 임의의 정수 순서를 부여하는 Label Encoding보다 직교 기저를 형성하는 <code>One-Hot Encoding</code>이 뚜렷하게 우수했다.
    그러나 트리 기반 모델(<code>Random Forest</code>, <code>XGBoost</code>, <code>LightGBM</code>)에서는 카디널리티가 높은 변수를 One-Hot으로 과도하게 쪼개면 특성 희소성(Sparsity)이 커져 단일 분기 정보 이득이 분산될 수 있으며, 실제로 본 실험에서도 트리 모델에서는 <code>Target Encoding</code>이나 <code>Label Encoding</code>이 One-Hot과 대등하거나 일부 지표에서 우위를 보였다.
  </p>
</div>

<div class="qa-card">
  <div class="qa-q">Q3. Feature Selection이 과적합 감소에 기여했는가?</div>
  <p>
    <strong>실질적인 과적합 감소와 일반화 정확도 개선에 기여하였다.</strong> One-Hot 인코딩 후 확장된 29개 전체 변수를 투입했을 때 4개 모델 평균 과적합 격차(<code>Train Acc - CV Val Acc</code>)는 <strong>{gap_before*100:.2f}%p</strong>(선형 모델 AUC 격차 0.0146)였으나, 희소 노이즈 더미 변수를 제거하고 상위 12개 핵심 변수만 남긴 결과 <code>SelectKBest(MI)</code> 기준 <strong>{gap_after_mi*100:.2f}%p</strong>(선형 모델 AUC 격차 0.0004), <code>RFE</code> 기준 <strong>{gap_after_rfe*100:.2f}%p</strong>, <code>RF Importance</code> 기준 <strong>{gap_after_rf*100:.2f}%p</strong>로 학습-검증 격차가 뚜렷이 감소하며 평균 CV 정확도(82.52% → 83.08%)가 향상되었다.
  </p>
</div>

<div class="qa-card">
  <div class="qa-q">Q4. 스케일링이 모델별로 어떤 영향을 미쳤는가?</div>
  <p>
    <strong>선형 모델에는 유의미한 수렴 및 성능 개선 효과를 주었으나, 트리 기반 모델에는 완벽히 불변(Scale-invariant)이었다.</strong>
    정규화 페널티와 경사 최적화를 사용하는 <code>Logistic Regression</code>은 원본 스케일(CV AUC {lr_unscaled_auc:.4f}, CV Acc 81.89%) 대비 스케일링 적용 시(<code>MinMaxScaler</code> CV AUC {lr_minmax_auc:.4f}, CV Acc 82.17%; <code>Standard/Robust</code> CV Acc 82.03%) 수치 범위 편차(<code>Fare</code> 최대 £512 vs 이진 더미 0~1)로 인한 가중치 왜곡이 해소되었다.
    반면 특성 값의 정렬 순서만으로 분기점을 탐색하는 <code>XGBoost</code>는 원본({xgb_unscaled_auc:.4f})과 <code>RobustScaler</code>({xgb_robust_auc:.4f}) 등 4가지 스케일러에서 소수점 여섯째 자리까지 동일한 성능을 기록하여 트리 모델의 스케일 불변성이 실험적으로 증명되었다.
  </p>
</div>

<div class="qa-card">
  <div class="qa-q">Q5. Feature Engineering이 실제 성능 향상에 얼마나 기여했는가?</div>
  <p>
    <strong>모델 종류나 하이퍼파라미터 변경보다 훨씬 결정적인 성능 향상을 견인하였다.</strong>
    전처리 없는 기준선(<code>Base</code>) 대비 특성 공학 파이프라인 적용 시 4개 모델 평균 Test ROC-AUC가 <strong>{base_avg_auc:.4f} → {exp4_avg_auc:.4f}(+{(exp4_avg_auc - base_avg_auc)*100:.2f}%p)</strong>, Test F1-Score가 <strong>{base_avg_f1:.4f} → {exp4_avg_f1:.4f}(+{(exp4_avg_f1 - base_avg_f1)*100:.2f}%p)</strong>로 대폭 향상되었다.
    동일 전처리 하에서 순수 파생변수 7종 추가만으로도 전 모델 평균 CV AUC가 <strong>+{(fe_after_avg_auc - fe_before_avg_auc)*100:.2f}%p</strong> 상승했으며, FLAML AutoML 대비 도메인 FE + 변수 선택 + GridSearchCV 파이프라인이 CV AUC(<strong>{automl_raw_cv:.4f} → {manual_cv:.4f}</strong>) 및 Test AUC(<strong>{automl_raw_auc:.4f} → {manual_test_auc:.4f}</strong>) 모두에서 우위를 기록하여 양질의 Feature Engineering이 머신러닝 파이프라인 성패의 핵심임을 확인하였다.
  </p>
</div>

</body>
</html>
"""
    report_dir = PROJECT_ROOT / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    html_path = report_dir / "report.html"
    html_path.write_text(html_content, encoding="utf-8")

    pdf_path = PROJECT_ROOT / pdf_filename
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page()
        page.goto(html_path.as_uri(), wait_until="networkidle")
        page.pdf(
            path=str(pdf_path),
            format="A4",
            print_background=True,
            display_header_footer=True,
            header_template='<div style="font-size:7pt; color:#94a3b8; width:100%; text-align:right; padding-right:14mm; font-family:sans-serif;">빅데이터 과제 — 특성 공학(Feature Engineering) ML 파이프라인 비교 분석</div>',
            footer_template='<div style="font-size:7.5pt; color:#64748b; width:100%; text-align:center; font-family:sans-serif;"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
            margin={"top": "15mm", "bottom": "16mm", "left": "14mm", "right": "14mm"},
        )
        browser.close()

    # Render page previews for visual quality assurance
    doc = fitz.open(str(pdf_path))
    print(f"PDF generated: {pdf_path} ({len(doc)} pages)")
    for idx, pg in enumerate(doc):
        pix = pg.get_pixmap(dpi=130)
        pix.save(str(report_dir / f"page_{idx+1:02d}.png"))
    doc.close()
    return pdf_path


if __name__ == "__main__":
    import sys

    repo_url = sys.argv[1] if len(sys.argv) > 1 else "https://github.com/jagari/bigdata-titanic-fe-pipeline"
    build_report(repo_url)
