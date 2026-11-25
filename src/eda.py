"""STEP 01 (Data Preparation) & STEP 02 (Exploratory Data Analysis) execution & visualizations."""

from __future__ import annotations

from typing import Dict, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils import (
    DATA_DIR,
    FIG_DIR,
    PALETTE_BINARY,
    PALETTE_MAIN,
    RESULTS_DIR,
    configure_korean_font,
    ensure_dirs,
    save_json,
)


COLUMN_DESCRIPTIONS = [
    ("PassengerId", "수치형 (ID)", "승객 고유 식별 번호 (1 ~ 891)", "식별자 (모델 학습 제외)"),
    ("Survived", "범주형 (이진 타겟)", "생존 여부 (0 = 사망, 1 = 생존)", "타겟 변수 (Target)"),
    ("Pclass", "범주형 (순서형)", "객실 등급 (1 = 1등석, 2 = 2등석, 3 = 3등석)", "사회·경제적 지위 대리 변수"),
    ("Name", "텍스트 (범주형)", "승객 성명 및 사회적 호칭(Mr, Mrs, Miss 등) 포함", "파생변수 Title 추출 후 제거"),
    ("Sex", "범주형 (명목형)", "성별 (male, female)", "핵심 예측 변수"),
    ("Age", "수치형 (연속형)", "승객 나이 (세, 유아는 소수점 표기)", "약 19.87% 결측 존재"),
    ("SibSp", "수치형 (이산형)", "동승한 형제자매 및 배우자 수", "가족 규모 파생에 활용"),
    ("Parch", "수치형 (이산형)", "동승한 부모 및 자녀 수", "가족 규모 파생에 활용"),
    ("Ticket", "텍스트 (범주형)", "티켓 고유 번호 (681개 고유값)", "고카디널리티 변수 (제거)"),
    ("Fare", "수치형 (연속형)", "지불 운임 요금 (영국 파운드, £)", "우측 편향(왜도) 및 이상치 존재"),
    ("Cabin", "범주형 (텍스트)", "객실 번호 (A~G 갑판 구역 + 호실)", "약 77.10% 고결측 → HasCabin/Deck 변환"),
    ("Embarked", "범주형 (명목형)", "탑승 항구 (C=Cherbourg, Q=Queenstown, S=Southampton)", "2건(0.22%) 결측 존재"),
]


def run_step1_and_step2() -> Dict[str, Any]:
    """Run data inspection and full EDA, generating figures 01-05 and structured JSON/CSV summaries."""
    ensure_dirs()
    font_used = configure_korean_font()
    df = pd.read_csv(DATA_DIR / "titanic.csv")

    # --- STEP 01: Column metadata table ---
    col_rows = []
    for col, dtype_desc, desc, note in COLUMN_DESCRIPTIONS:
        s = df[col]
        n_miss = int(s.isna().sum())
        miss_pct = round(n_miss / len(df) * 100, 2)
        n_unique = int(s.nunique(dropna=True))
        sample_vals = ", ".join(map(str, s.dropna().unique()[:3]))
        col_rows.append(
            {
                "컬럼명": col,
                "데이터 유형": dtype_desc,
                "실제 dtype": str(s.dtype),
                "결측치 수(비율)": f"{n_miss} ({miss_pct:.2f}%)",
                "고유값 수": n_unique,
                "예시 값": sample_vals,
                "설명 및 전처리 방향": f"{desc} / {note}",
            }
        )
    col_df = pd.DataFrame(col_rows)
    col_df.to_csv(RESULTS_DIR / "step01_column_summary.csv", index=False, encoding="utf-8-sig")

    # --- STEP 02: Missing values & Outlier IQR statistics ---
    num_cols = ["Age", "SibSp", "Parch", "Fare"]
    outlier_rows = []
    for col in num_cols:
        s = df[col].dropna()
        q1, q3 = float(s.quantile(0.25)), float(s.quantile(0.75))
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        outliers = s[(s < lower) | (s > upper)]
        outlier_rows.append(
            {
                "변수명": col,
                "유효 샘플 수": int(len(s)),
                "평균": round(float(s.mean()), 2),
                "중앙값": round(float(s.median()), 2),
                "표준편차": round(float(s.std()), 2),
                "왜도(Skewness)": round(float(s.skew()), 2),
                "Q1 / Q3": f"{q1:.2f} / {q3:.2f}",
                "IQR 정상 범위": f"[{lower:.2f}, {upper:.2f}]",
                "이상치 개수(비율)": f"{len(outliers)} ({len(outliers)/len(s)*100:.2f}%)",
                "최댓값": round(float(s.max()), 2),
            }
        )
    outlier_df = pd.DataFrame(outlier_rows)
    outlier_df.to_csv(RESULTS_DIR / "step02_outlier_summary.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # Figure 01: Missing Value Ratio & Target Class Distribution (Countplot/Bar)
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # Left: Missing value barplot
    miss_s = (df.isna().sum() / len(df) * 100).sort_values(ascending=False)
    miss_nonzero = miss_s[miss_s > 0]
    bars = axes[0].barh(
        miss_nonzero.index[::-1],
        miss_nonzero.values[::-1],
        color=["#2e75b6", "#ed7d31", "#c00000"],
        edgecolor="black",
        alpha=0.88,
    )
    axes[0].set_xlim(0, 92)
    axes[0].set_xlabel("결측치 비율 (%)")
    axes[0].set_title("(a) 변수별 결측치 비율 분석 (전체 891행 기준)")
    for bar, col in zip(bars, miss_nonzero.index[::-1]):
        cnt = int(df[col].isna().sum())
        pct = bar.get_width()
        axes[0].text(
            pct + 1.5,
            bar.get_y() + bar.get_height() / 2,
            f"{cnt}건 ({pct:.2f}%)",
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )

    # Right: Target Countplot
    target_counts = df["Survived"].value_counts().sort_index()
    labels = ["사망 (Survived = 0)", "생존 (Survived = 1)"]
    bars2 = axes[1].bar(
        labels,
        target_counts.values,
        color=PALETTE_BINARY,
        edgecolor="black",
        width=0.52,
        alpha=0.9,
    )
    axes[1].set_ylim(0, 650)
    axes[1].set_ylabel("승객 수 (명)")
    axes[1].set_title("(b) 타겟 변수(Survived) 분포 (Countplot)")
    for bar, cnt in zip(bars2, target_counts.values):
        pct = cnt / len(df) * 100
        axes[1].text(
            bar.get_x() + bar.get_width() / 2,
            cnt + 14,
            f"{cnt}명 ({pct:.2f}%)",
            ha="center",
            fontsize=10,
            fontweight="bold",
        )

    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig01_missing_and_target.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 02: Histograms of Numeric Variables by Survival Status
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.5))
    plot_specs = [
        ("Age", "승객 나이 (Age, 세)", 28, True),
        ("Fare", "운임 요금 (Fare, £)", 35, True),
        ("SibSp", "형제자매·배우자 수 (SibSp)", 9, False),
        ("Parch", "부모·자녀 수 (Parch)", 7, False),
    ]
    for ax, (col, title, bins, kde_flag) in zip(axes.flat, plot_specs):
        sns.histplot(
            data=df,
            x=col,
            hue="Survived",
            bins=bins,
            kde=kde_flag,
            palette=PALETTE_BINARY,
            alpha=0.55,
            ax=ax,
        )
        skew_val = df[col].dropna().skew()
        med_val = df[col].dropna().median()
        mean_val = df[col].dropna().mean()
        ax.axvline(mean_val, color="#1f4e79", linestyle="--", linewidth=1.4, label=f"평균: {mean_val:.1f}")
        ax.axvline(med_val, color="#c00000", linestyle=":", linewidth=1.6, label=f"중앙값: {med_val:.1f}")
        ax.set_title(f"{title} 분포 (왜도={skew_val:.2f})")
        ax.set_xlabel(col)
        ax.set_ylabel("빈도 (명)")
        ax.legend(fontsize=8.5)

    plt.suptitle("수치형 변수의 분포 및 생존 여부별 히스토그램 (Histogram)", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig02_histograms.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 03: Boxplots for Outlier Detection & Group Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8))

    sns.boxplot(
        data=df,
        x="Survived",
        y="Age",
        hue="Survived",
        palette=PALETTE_BINARY,
        legend=False,
        width=0.48,
        ax=axes[0],
    )
    axes[0].set_xticks([0, 1])
    axes[0].set_xticklabels(["사망 (0)", "생존 (1)"])
    axes[0].set_title("(a) 생존 여부별 Age Boxplot")

    sns.boxplot(
        data=df,
        x="Pclass",
        y="Fare",
        hue="Survived",
        palette=PALETTE_BINARY,
        ax=axes[1],
    )
    axes[1].set_title("(b) 객실 등급·생존별 Fare Boxplot (원본 스케일)")
    axes[1].legend(title="Survived", labels=["사망(0)", "생존(1)"], fontsize=8.5)

    df_temp = df.copy()
    df_temp["LogFare"] = np.log1p(df_temp["Fare"])
    sns.boxplot(
        data=df_temp,
        x="Pclass",
        y="LogFare",
        hue="Survived",
        palette=PALETTE_BINARY,
        ax=axes[2],
    )
    axes[2].set_title("(c) log(1+Fare) 변환 후 객실 등급별 분포")
    axes[2].set_ylabel("log(1 + Fare)")
    axes[2].legend(title="Survived", labels=["사망(0)", "생존(1)"], fontsize=8.5)

    plt.suptitle("수치형 변수 이상치 탐색을 위한 상자 수염 그림 (Boxplot)", fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig03_boxplots.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 04: Correlation Heatmap (Numeric + Key Binary Indicators)
    # =========================================================================
    corr_df = pd.DataFrame(
        {
            "Survived": df["Survived"],
            "Pclass": df["Pclass"],
            "Sex_female": (df["Sex"] == "female").astype(int),
            "Age": df["Age"],
            "SibSp": df["SibSp"],
            "Parch": df["Parch"],
            "Fare": df["Fare"],
            "HasCabin": df["Cabin"].notna().astype(int),
            "Embarked_C": (df["Embarked"] == "C").astype(int),
        }
    )
    corr_mat = corr_df.corr(method="pearson")
    corr_mat.to_csv(RESULTS_DIR / "step02_correlation_matrix.csv", encoding="utf-8-sig")

    fig, ax = plt.subplots(figsize=(9.2, 7.5))
    mask = np.triu(np.ones_like(corr_mat, dtype=bool), k=1)
    sns.heatmap(
        corr_mat,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        vmin=-0.65,
        vmax=0.65,
        center=0,
        square=True,
        linewidths=0.8,
        cbar_kws={"shrink": 0.82, "label": "Pearson 상관계수 (r)"},
        ax=ax,
    )
    ax.set_title("주요 수치형·이진화 변수 간 상관관계 히트맵 (Correlation Heatmap)", pad=14)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig04_correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 05: Categorical Countplot & Survival Rate Barplots
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    cat_specs = [
        ("Sex", "성별(Sex)에 따른 생존율", ["male", "female"], ["남성(male)", "여성(female)"]),
        ("Pclass", "객실 등급(Pclass)에 따른 생존율", [1, 2, 3], ["1등석", "2등석", "3등석"]),
        ("Embarked", "탑승 항구(Embarked)에 따른 생존율", ["S", "C", "Q"], ["S (사우샘프턴)", "C (셰르부르)", "Q (퀸스타운)"]),
    ]
    for ax, (col, title, order, xlabels) in zip(axes, cat_specs):
        rates = df.groupby(col)["Survived"].agg(["mean", "count"]).reindex(order)
        bars = ax.bar(
            xlabels,
            rates["mean"] * 100,
            color=PALETTE_MAIN[: len(order)],
            edgecolor="black",
            width=0.55,
            alpha=0.88,
        )
        ax.set_ylim(0, 92)
        ax.set_ylabel("생존율 (%)")
        ax.set_title(title)
        ax.axhline(df["Survived"].mean() * 100, color="#c00000", linestyle="--", linewidth=1.2, label="전체 평균 (38.4%)")
        for bar, (_, row) in zip(bars, rates.iterrows()):
            pct = row["mean"] * 100
            cnt = int(row["count"])
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                pct + 2.2,
                f"{pct:.1f}%\n(n={cnt})",
                ha="center",
                fontsize=9,
                fontweight="bold",
            )
        ax.legend(fontsize=8.5, loc="upper right")

    plt.suptitle("주요 범주형 변수별 승객 규모 및 생존율 비교 (Barplot)", fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig05_categorical_bars.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "font_used": font_used,
        "shape": list(df.shape),
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "target_distribution": {
            "died_0": int((df["Survived"] == 0).sum()),
            "survived_1": int((df["Survived"] == 1).sum()),
            "survival_rate_pct": round(float(df["Survived"].mean() * 100), 2),
        },
        "missing_counts": {k: int(v) for k, v in df.isna().sum().items() if v > 0},
        "missing_pcts": {
            k: round(float(v / len(df) * 100), 2) for k, v in df.isna().sum().items() if v > 0
        },
        "survival_by_sex": {
            k: round(float(v * 100), 2) for k, v in df.groupby("Sex")["Survived"].mean().items()
        },
        "survival_by_pclass": {
            str(k): round(float(v * 100), 2)
            for k, v in df.groupby("Pclass")["Survived"].mean().items()
        },
        "top_correlations_with_survived": {
            k: round(float(v), 4)
            for k, v in corr_mat["Survived"].drop("Survived").sort_values(key=abs, ascending=False).items()
        },
    }
    save_json(summary, "step01_02_eda_summary.json")
    return summary


if __name__ == "__main__":
    res = run_step1_and_step2()
    print("EDA Completed:", res["shape"], res["target_distribution"])
