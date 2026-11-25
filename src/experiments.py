"""STEP 03 (Feature Engineering), STEP 04 (Feature Selection), STEP 05 & Section 5 Experiments."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE, SelectFromModel, SelectKBest, f_classif, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import (
    MinMaxScaler,
    OneHotEncoder,
    OrdinalEncoder,
    RobustScaler,
    StandardScaler,
    TargetEncoder,
)
from xgboost import XGBClassifier

from src.transformers import (
    CorrelationFeatureSelector,
    GroupMedianAgeImputer,
    TitanicFeatureEngineer,
)
from src.utils import (
    DATA_DIR,
    FIG_DIR,
    PALETTE_MAIN,
    RANDOM_STATE,
    RESULTS_DIR,
    configure_korean_font,
    ensure_dirs,
    evaluate_classifier,
    save_json,
)


BASE_NUM_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
BASE_CAT_COLS = ["Sex", "Embarked"]

DERIVED_NUM_COLS = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize", "IsAlone", "FarePerPerson", "HasCabin"]
DERIVED_CAT_COLS = ["Sex", "Embarked", "Title", "Deck", "AgeGroup"]


def get_models() -> Dict[str, Any]:
    """Return 4 standard classification models with fixed random seeds."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
        ),
        "XGBoost": XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=1,
        ),
        "LightGBM": LGBMClassifier(
            n_estimators=180,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=RANDOM_STATE,
            n_jobs=1,
            verbose=-1,
        ),
    }


def make_preprocessor(
    use_derived: bool = True,
    num_imputer: str = "median",
    cat_imputer: str = "most_frequent",
    encoder: str = "onehot",
    scaler: str = "standard",
    use_group_age: bool = False,
) -> Pipeline:
    """Build modular leak-free preprocessing pipeline."""
    num_cols = DERIVED_NUM_COLS if use_derived else BASE_NUM_COLS
    cat_cols = DERIVED_CAT_COLS if use_derived else BASE_CAT_COLS

    # Numeric steps
    num_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=num_imputer)),
    ]
    if scaler == "standard":
        num_steps.append(("scaler", StandardScaler()))
    elif scaler == "minmax":
        num_steps.append(("scaler", MinMaxScaler()))
    elif scaler == "robust":
        num_steps.append(("scaler", RobustScaler()))
    elif scaler == "none":
        pass
    else:
        raise ValueError(f"Unknown scaler: {scaler}")

    # Categorical steps
    cat_steps: List[Tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy=cat_imputer)),
    ]
    if encoder == "onehot":
        cat_steps.append(("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)))
    elif encoder == "label":
        cat_steps.append(
            ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
        )
    elif encoder == "target":
        cat_steps.append(
            ("encoder", TargetEncoder(cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)))
        )
    else:
        raise ValueError(f"Unknown encoder: {encoder}")

    col_tf = ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline(cat_steps), cat_cols),
        ],
        remainder="drop",
    )

    steps: List[Tuple[str, Any]] = []
    if use_group_age:
        steps.append(("group_age", GroupMedianAgeImputer()))
    steps.append(("fe", TitanicFeatureEngineer(add_derived=use_derived, drop_identifiers=True)))
    steps.append(("prep", col_tf))
    return Pipeline(steps)


def make_selector(method: str, k: int = 10) -> Optional[Any]:
    """Create feature selector step."""
    if method == "none":
        return None
    if method == "rf_importance":
        rf = RandomForestClassifier(n_estimators=150, max_depth=7, random_state=RANDOM_STATE, n_jobs=1)
        return SelectFromModel(rf, max_features=k, threshold=-np.inf)
    if method == "selectkbest_mi":
        return SelectKBest(
            score_func=lambda X, y: mutual_info_classif(X, y, random_state=RANDOM_STATE), k=k
        )
    if method == "selectkbest_f":
        return SelectKBest(score_func=f_classif, k=k)
    if method == "correlation":
        return CorrelationFeatureSelector(k=k)
    if method == "rfe":
        est = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=RANDOM_STATE, n_jobs=1)
        return RFE(estimator=est, n_features_to_select=k, step=1)
    raise ValueError(f"Unknown selector: {method}")


def plot_derived_features_eda(df: pd.DataFrame) -> None:
    """Visualize survival rates across key derived features (Figure 06)."""
    fe = TitanicFeatureEngineer(add_derived=True, drop_identifiers=False)
    df_fe = fe.fit_transform(df)
    df_fe["Survived"] = df["Survived"].values

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # 1. Title survival rate
    title_order = ["Mrs", "Miss", "Master", "Rare", "Mr"]
    t_stats = df_fe.groupby("Title")["Survived"].agg(["mean", "count"]).reindex(title_order)
    bars = axes[0, 0].bar(
        t_stats.index, t_stats["mean"] * 100, color=PALETTE_MAIN[:5], edgecolor="black", alpha=0.88
    )
    axes[0, 0].set_ylim(0, 95)
    axes[0, 0].set_ylabel("생존율 (%)")
    axes[0, 0].set_title("(a) 파생변수 Title(호칭)별 생존율")
    for b, (_, r) in zip(bars, t_stats.iterrows()):
        axes[0, 0].text(
            b.get_x() + b.get_width() / 2,
            r["mean"] * 100 + 2,
            f"{r['mean']*100:.1f}%\n(n={int(r['count'])})",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    # 2. FamilySize survival rate
    f_stats = df_fe.groupby("FamilySize")["Survived"].agg(["mean", "count"])
    bars2 = axes[0, 1].bar(
        f_stats.index.astype(str),
        f_stats["mean"] * 100,
        color="#2e75b6",
        edgecolor="black",
        alpha=0.88,
    )
    axes[0, 1].axhline(df["Survived"].mean() * 100, color="#c00000", linestyle="--", label="평균 (38.4%)")
    axes[0, 1].set_ylim(0, 88)
    axes[0, 1].set_xlabel("가족 규모 (FamilySize = SibSp + Parch + 1)")
    axes[0, 1].set_ylabel("생존율 (%)")
    axes[0, 1].set_title("(b) 파생변수 FamilySize별 생존율 (2~4인 가족 우세)")
    axes[0, 1].legend(fontsize=8.5)
    for b, (_, r) in zip(bars2, f_stats.iterrows()):
        axes[0, 1].text(
            b.get_x() + b.get_width() / 2,
            r["mean"] * 100 + 2,
            f"{r['mean']*100:.0f}%",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    # 3. AgeGroup survival rate
    age_order = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
    age_labels = ["Child\n(≤12)", "Teen\n(13~19)", "YoungAdult\n(20~35)", "Adult\n(36~55)", "Senior\n(56+)"]
    a_stats = df_fe.groupby("AgeGroup")["Survived"].agg(["mean", "count"]).reindex(age_order)
    bars3 = axes[1, 0].bar(
        age_labels, a_stats["mean"] * 100, color="#70ad47", edgecolor="black", alpha=0.88
    )
    axes[1, 0].set_ylim(0, 75)
    axes[1, 0].set_ylabel("생존율 (%)")
    axes[1, 0].set_title("(c) 파생변수 AgeGroup(연령대)별 생존율")
    for b, (_, r) in zip(bars3, a_stats.iterrows()):
        axes[1, 0].text(
            b.get_x() + b.get_width() / 2,
            r["mean"] * 100 + 2,
            f"{r['mean']*100:.1f}%\n(n={int(r['count'])})",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    # 4. FarePerPerson & HasCabin comparison
    cabin_stats = df_fe.groupby(["Pclass", "HasCabin"])["Survived"].mean().unstack() * 100
    cabin_stats.plot(
        kind="bar",
        color=["#d95f02", "#1b9e77"],
        edgecolor="black",
        alpha=0.88,
        ax=axes[1, 1],
        rot=0,
    )
    axes[1, 1].set_ylim(0, 90)
    axes[1, 1].set_xlabel("객실 등급 (Pclass)")
    axes[1, 1].set_ylabel("생존율 (%)")
    axes[1, 1].set_title("(d) 객실 등급 및 HasCabin(객실 기록 유무)별 생존율")
    axes[1, 1].legend(["객실 미기재 (0)", "객실 보유 (1)"], fontsize=8.5)

    plt.suptitle("STEP 3-4: 도메인 기반 주요 파생 변수와 생존율의 관계 분석", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig06_derived_features_eda.png", bbox_inches="tight")
    plt.close(fig)


def run_all_experiments() -> Dict[str, Any]:
    """Run STEP 03, STEP 04, STEP 05, and Section 5 Mandatory Comparison."""
    ensure_dirs()
    configure_korean_font()
    df = pd.read_csv(DATA_DIR / "titanic.csv")
    plot_derived_features_eda(df)

    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    # =========================================================================
    # 3-1. Missing Value Imputation Comparison
    # =========================================================================
    imp_rows = []
    # Strategy A: Drop NA (drop Cabin first, then drop rows with missing Age/Embarked)
    train_dropna_mask = X_train[["Age", "Embarked", "Fare"]].notna().all(axis=1)
    test_dropna_mask = X_test[["Age", "Embarked", "Fare"]].notna().all(axis=1)
    X_tr_drop = X_train.loc[train_dropna_mask]
    y_tr_drop = y_train.loc[train_dropna_mask]
    X_te_drop = X_test.loc[test_dropna_mask]
    y_te_drop = y_test.loc[test_dropna_mask]

    for model_name, model in get_models().items():
        pipe_drop = Pipeline(
            [
                ("prep", make_preprocessor(use_derived=False, num_imputer="median", encoder="onehot", scaler="standard")),
                ("clf", model),
            ]
        )
        m_drop = evaluate_classifier(pipe_drop, X_tr_drop, y_tr_drop, X_te_drop, y_te_drop)
        imp_rows.append(
            {"전략": "Drop NA (결측 행 제거)", "샘플 수(Train/Test)": f"{len(X_tr_drop)}/{len(X_te_drop)}", "모델": model_name, **m_drop}
        )

    imp_configs = [
        ("Mean Imputation", "mean", "most_frequent", False),
        ("Median Imputation", "median", "most_frequent", False),
        ("Most Frequent", "most_frequent", "most_frequent", False),
        ("Group Median (Pclass×Sex)", "median", "most_frequent", True),
    ]
    for label, num_imp, cat_imp, grp_flag in imp_configs:
        for model_name, model in get_models().items():
            pipe = Pipeline(
                [
                    (
                        "prep",
                        make_preprocessor(
                            use_derived=False,
                            num_imputer=num_imp,
                            cat_imputer=cat_imp,
                            encoder="onehot",
                            scaler="standard",
                            use_group_age=grp_flag,
                        ),
                    ),
                    ("clf", model),
                ]
            )
            m = evaluate_classifier(pipe, X_train, y_train, X_test, y_test)
            imp_rows.append(
                {"전략": label, "샘플 수(Train/Test)": f"{len(X_train)}/{len(X_test)}", "모델": model_name, **m}
            )
    imp_df = pd.DataFrame(imp_rows)
    imp_df.to_csv(RESULTS_DIR / "step03_1_imputation.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # 3-2. Categorical Encoding Comparison (with derived categoricals included)
    # =========================================================================
    enc_rows = []
    for enc_label, enc_code in [
        ("Label (Ordinal) Encoding", "label"),
        ("One-Hot Encoding", "onehot"),
        ("Target Encoding", "target"),
    ]:
        for model_name, model in get_models().items():
            pipe = Pipeline(
                [
                    ("prep", make_preprocessor(use_derived=True, num_imputer="median", encoder=enc_code, scaler="standard")),
                    ("clf", model),
                ]
            )
            m = evaluate_classifier(pipe, X_train, y_train, X_test, y_test)
            enc_rows.append({"인코딩 전략": enc_label, "모델": model_name, **m})
    enc_df = pd.DataFrame(enc_rows)
    enc_df.to_csv(RESULTS_DIR / "step03_2_encoding.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # 3-3. Feature Scaling Comparison
    # =========================================================================
    scl_rows = []
    for scl_label, scl_code in [
        ("No Scaling (원본)", "none"),
        ("StandardScaler", "standard"),
        ("MinMaxScaler", "minmax"),
        ("RobustScaler", "robust"),
    ]:
        for model_name, model in get_models().items():
            pipe = Pipeline(
                [
                    ("prep", make_preprocessor(use_derived=True, num_imputer="median", encoder="onehot", scaler=scl_code)),
                    ("clf", model),
                ]
            )
            m = evaluate_classifier(pipe, X_train, y_train, X_test, y_test)
            scl_rows.append({"스케일링 전략": scl_label, "모델": model_name, **m})
    scl_df = pd.DataFrame(scl_rows)
    scl_df.to_csv(RESULTS_DIR / "step03_3_scaling.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # 3-4. Derived Features Before vs After Comparison
    # =========================================================================
    fe_rows = []
    for fe_label, fe_flag in [
        ("파생변수 생성 전 (원본 7개 특성)", False),
        ("파생변수 생성 후 (+7개 도메인 특성)", True),
    ]:
        for model_name, model in get_models().items():
            pipe = Pipeline(
                [
                    ("prep", make_preprocessor(use_derived=fe_flag, num_imputer="median", encoder="onehot", scaler="robust")),
                    ("clf", model),
                ]
            )
            m = evaluate_classifier(pipe, X_train, y_train, X_test, y_test)
            fe_rows.append({"파생변수 여부": fe_label, "모델": model_name, **m})
    fe_df = pd.DataFrame(fe_rows)
    fe_df.to_csv(RESULTS_DIR / "step03_4_derived_features.csv", index=False, encoding="utf-8-sig")

    # Plot Figure 07: 2x2 Summary of STEP 03 Comparisons (CV ROC-AUC)
    fig, axes = plt.subplots(2, 2, figsize=(14.5, 10))
    sns.barplot(
        data=imp_df, x="모델", y="cv_roc_auc_mean", hue="전략", palette="Blues_d", edgecolor="black", ax=axes[0, 0]
    )
    axes[0, 0].set_ylim(0.78, 0.90)
    axes[0, 0].set_ylabel("5-Fold CV ROC-AUC")
    axes[0, 0].set_title("(a) 3-1. 결측치 처리 전략별 모델 성능 비교")
    axes[0, 0].legend(fontsize=8, loc="lower right")
    axes[0, 0].tick_params(axis="x", rotation=10)

    sns.barplot(
        data=enc_df, x="모델", y="cv_roc_auc_mean", hue="인코딩 전략", palette="Set2", edgecolor="black", ax=axes[0, 1]
    )
    axes[0, 1].set_ylim(0.80, 0.90)
    axes[0, 1].set_ylabel("5-Fold CV ROC-AUC")
    axes[0, 1].set_title("(b) 3-2. 범주형 인코딩 전략별 모델 성능 비교")
    axes[0, 1].legend(fontsize=8.5, loc="lower right")
    axes[0, 1].tick_params(axis="x", rotation=10)

    sns.barplot(
        data=scl_df, x="모델", y="cv_roc_auc_mean", hue="스케일링 전략", palette="Paired", edgecolor="black", ax=axes[1, 0]
    )
    axes[1, 0].set_ylim(0.82, 0.90)
    axes[1, 0].set_ylabel("5-Fold CV ROC-AUC")
    axes[1, 0].set_title("(c) 3-3. 스케일링 전략별 모델 성능 비교")
    axes[1, 0].legend(fontsize=8.5, loc="lower right")
    axes[1, 0].tick_params(axis="x", rotation=10)

    sns.barplot(
        data=fe_df, x="모델", y="cv_roc_auc_mean", hue="파생변수 여부", palette=["#a5a5a5", "#1f4e79"], edgecolor="black", ax=axes[1, 1]
    )
    axes[1, 1].set_ylim(0.80, 0.90)
    axes[1, 1].set_ylabel("5-Fold CV ROC-AUC")
    axes[1, 1].set_title("(d) 3-4. 도메인 파생변수 추가 전/후 성능 향상")
    axes[1, 1].legend(fontsize=8.5, loc="lower right")
    axes[1, 1].tick_params(axis="x", rotation=10)

    plt.suptitle("STEP 03: 전처리 및 파생변수 전략별 교차검증(5-Fold CV ROC-AUC) 비교", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig07_step03_comparisons.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # STEP 04: Feature Selection (4 Methods + Before/After + K Sweep)
    # =========================================================================
    fs_methods = [
        ("전체 변수 사용 (제거 전, 29개)", "none", 29),
        ("RF Feature Importance (Top-12)", "rf_importance", 12),
        ("SelectKBest - Mutual Info (Top-12)", "selectkbest_mi", 12),
        ("Correlation-based Selection (Top-12)", "correlation", 12),
        ("RFE 재귀적 변수 제거 (Top-12)", "rfe", 12),
    ]
    fs_rows = []
    for fs_label, fs_code, k_val in fs_methods:
        for model_name, model in get_models().items():
            steps_list: List[Tuple[str, Any]] = [
                ("prep", make_preprocessor(use_derived=True, num_imputer="median", encoder="onehot", scaler="robust", use_group_age=True))
            ]
            sel = make_selector(fs_code, k=k_val)
            if sel is not None:
                steps_list.append(("select", sel))
            steps_list.append(("clf", model))
            pipe = Pipeline(steps_list)
            m = evaluate_classifier(pipe, X_train, y_train, X_test, y_test)
            fs_rows.append({"변수 선택 방법": fs_label, "선택 변수 수": k_val, "모델": model_name, **m})
    fs_df = pd.DataFrame(fs_rows)
    fs_df.to_csv(RESULTS_DIR / "step04_feature_selection.csv", index=False, encoding="utf-8-sig")

    # Top-K sweep curve (K = 4..26) for RF & Logistic Regression using RFE & RF Importance
    prep_fit = make_preprocessor(
        use_derived=True, num_imputer="median", encoder="onehot", scaler="robust", use_group_age=True
    )
    X_tr_trans = prep_fit.fit_transform(X_train, y_train)
    raw_feat_names = list(prep_fit.named_steps["prep"].get_feature_names_out())
    clean_feat_names = [n.replace("num__", "").replace("cat__", "") for n in raw_feat_names]

    rf_full = RandomForestClassifier(n_estimators=250, max_depth=8, random_state=RANDOM_STATE, n_jobs=1)
    rf_full.fit(X_tr_trans, y_train)
    feat_imp_series = pd.Series(rf_full.feature_importances_, index=clean_feat_names).sort_values(ascending=False)
    feat_imp_series.to_csv(RESULTS_DIR / "step04_rf_feature_ranking.csv", header=["rf_importance"], encoding="utf-8-sig")

    k_candidates = [4, 6, 8, 10, 12, 15, 18, 22, len(clean_feat_names)]
    k_sweep_rows = []
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    for k in k_candidates:
        for m_name in ["Logistic Regression", "Random Forest", "XGBoost"]:
            model = get_models()[m_name]
            pipe_k = Pipeline(
                [
                    ("prep", make_preprocessor(use_derived=True, num_imputer="median", encoder="onehot", scaler="robust", use_group_age=True)),
                    ("select", make_selector("rf_importance", k=k)),
                    ("clf", model),
                ]
            )
            scores = cross_val_score(pipe_k, X_train, y_train, cv=cv5, scoring="roc_auc", n_jobs=1)
            k_sweep_rows.append({"k": k, "모델": m_name, "cv_roc_auc": float(np.mean(scores))})
    k_sweep_df = pd.DataFrame(k_sweep_rows)

    # Plot Figure 08: Feature Selection Analysis (Top-15 Ranking, K-Sweep, Overfitting Gap Reduction)
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    top15 = feat_imp_series.head(15)[::-1]
    axes[0].barh(top15.index, top15.values, color="#1f4e79", edgecolor="black", alpha=0.88)
    axes[0].set_xlabel("Random Forest 불순도 감소 중요도")
    axes[0].set_title("(a) 상위 15개 핵심 특성 중요도 순위")

    for m_name, color in zip(
        ["Logistic Regression", "Random Forest", "XGBoost"], ["#1f4e79", "#ed7d31", "#70ad47"]
    ):
        sub = k_sweep_df[k_sweep_df["모델"] == m_name]
        axes[1].plot(sub["k"], sub["cv_roc_auc"], marker="o", linewidth=2, label=m_name, color=color)
    axes[1].axvline(12, color="#c00000", linestyle="--", linewidth=1.3, label="선정 K=12")
    axes[1].set_xlabel("선택된 상위 변수 개수 (K)")
    axes[1].set_ylabel("5-Fold CV ROC-AUC")
    axes[1].set_title("(b) 선택 변수 수(K) 변화에 따른 성능 곡선")
    axes[1].legend(fontsize=8.5)

    gap_sub = fs_df[
        fs_df["변수 선택 방법"].isin(["전체 변수 사용 (제거 전, 29개)", "RFE 재귀적 변수 제거 (Top-12)", "RF Feature Importance (Top-12)"])
    ]
    sns.barplot(
        data=gap_sub,
        x="모델",
        y="overfit_gap_roc_auc",
        hue="변수 선택 방법",
        palette=["#d95f02", "#2e75b6", "#1b9e77"],
        edgecolor="black",
        ax=axes[2],
    )
    axes[2].set_ylabel("과적합 격차 (Train AUC - CV Val AUC)")
    axes[2].set_title("(c) 변수 선택 전/후 과적합(Overfitting) 감소 효과")
    axes[2].tick_params(axis="x", rotation=12)
    axes[2].legend(fontsize=7.8)

    plt.suptitle("STEP 04: 중요 특성 선정(Top-N) 및 변수 제거 전·후 성능/과적합 비교", fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig08_step04_feature_selection.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Section 5: Mandatory Experiment Comparison Table (Base, Exp-1, Exp-2, Exp-3 + Exp-4)
    # =========================================================================
    mandatory_rows = []
    roc_curves_for_plot: Dict[str, Tuple[np.ndarray, np.ndarray, float]] = {}

    # Define exact configurations from Section 5 + Proposed Optimal Exp-4
    # Base: 결측치 처리 없음 / 인코딩 없음 / 스케일링 없음 / Feature Selection 없음 / 파생변수 없음
    #       -> 결측치가 없는 순수 수치형 원본 특성 4개 (Pclass, SibSp, Parch, Fare) 사용
    pure_num_cols = ["Pclass", "SibSp", "Parch", "Fare"]

    exp_specs = [
        {
            "exp_id": "Base",
            "결측치 처리": "없음",
            "인코딩": "없음",
            "스케일링": "없음",
            "Feature Selection": "없음 (X)",
            "파생변수": "없음 (X)",
        },
        {
            "exp_id": "Exp-1",
            "결측치 처리": "Mean",
            "인코딩": "One-Hot",
            "스케일링": "Standard",
            "Feature Selection": "X",
            "파생변수": "적용 (O)",
            "prep_kwargs": dict(use_derived=True, num_imputer="mean", cat_imputer="most_frequent", encoder="onehot", scaler="standard", use_group_age=False),
            "sel": ("none", 0),
        },
        {
            "exp_id": "Exp-2",
            "결측치 처리": "Median",
            "인코딩": "Label",
            "스케일링": "MinMax",
            "Feature Selection": "O (Top-10)",
            "파생변수": "적용 (O)",
            "prep_kwargs": dict(use_derived=True, num_imputer="median", cat_imputer="most_frequent", encoder="label", scaler="minmax", use_group_age=False),
            "sel": ("rf_importance", 10),
        },
        {
            "exp_id": "Exp-3",
            "결측치 처리": "Most Frequent",
            "인코딩": "One-Hot",
            "스케일링": "Robust",
            "Feature Selection": "O (Top-12)",
            "파생변수": "적용 (O)",
            "prep_kwargs": dict(use_derived=True, num_imputer="most_frequent", cat_imputer="most_frequent", encoder="onehot", scaler="robust", use_group_age=False),
            "sel": ("rf_importance", 12),
        },
        {
            "exp_id": "Exp-4 (최적 제안)",
            "결측치 처리": "Group-Median",
            "인코딩": "One-Hot",
            "스케일링": "Robust",
            "Feature Selection": "O (RFE Top-14)",
            "파생변수": "적용 (O)",
            "prep_kwargs": dict(use_derived=True, num_imputer="median", cat_imputer="most_frequent", encoder="onehot", scaler="robust", use_group_age=True),
            "sel": ("rfe", 14),
        },
    ]

    for spec in exp_specs:
        exp_id = spec["exp_id"]
        for model_name, model in get_models().items():
            if exp_id == "Base":
                pipe = Pipeline(
                    [
                        ("select_raw_num", ColumnTransformer([("raw_num", "passthrough", pure_num_cols)], remainder="drop")),
                        ("clf", model),
                    ]
                )
            else:
                steps_e: List[Tuple[str, Any]] = [("prep", make_preprocessor(**spec["prep_kwargs"]))]
                sel_obj = make_selector(spec["sel"][0], k=spec["sel"][1])
                if sel_obj is not None:
                    steps_e.append(("select", sel_obj))
                steps_e.append(("clf", model))
                pipe = Pipeline(steps_e)

            metrics = evaluate_classifier(pipe, X_train, y_train, X_test, y_test)
            mandatory_rows.append(
                {
                    "실험": exp_id,
                    "결측치 처리": spec["결측치 처리"],
                    "인코딩": spec["인코딩"],
                    "스케일링": spec["스케일링"],
                    "Feature Selection": spec["Feature Selection"],
                    "모델": model_name,
                    **metrics,
                }
            )
            if model_name == "XGBoost":
                pipe.fit(X_train, y_train)
                probs = pipe.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, probs)
                roc_curves_for_plot[exp_id] = (fpr, tpr, float(roc_auc_score(y_test, probs)))

    mandatory_df = pd.DataFrame(mandatory_rows)
    mandatory_df.to_csv(RESULTS_DIR / "step05_mandatory_comparison.csv", index=False, encoding="utf-8-sig")

    # Plot Figure 09: Mandatory Experiment Comparison (Accuracy/F1/ROC-AUC across Base..Exp-4 + ROC Curve)
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.2))

    sns.barplot(
        data=mandatory_df,
        x="실험",
        y="test_f1",
        hue="모델",
        palette=PALETTE_MAIN[:4],
        edgecolor="black",
        ax=axes[0],
    )
    axes[0].set_ylim(0.42, 0.88)
    axes[0].set_ylabel("Test F1-Score")
    axes[0].set_title("(a) 실험 조합(Base ~ Exp-4)별 Test F1-Score 비교")
    axes[0].tick_params(axis="x", rotation=12)
    axes[0].legend(fontsize=8, loc="lower right")

    sns.barplot(
        data=mandatory_df,
        x="실험",
        y="test_roc_auc",
        hue="모델",
        palette=PALETTE_MAIN[:4],
        edgecolor="black",
        ax=axes[1],
    )
    axes[1].set_ylim(0.65, 0.92)
    axes[1].set_ylabel("Test ROC-AUC")
    axes[1].set_title("(b) 실험 조합(Base ~ Exp-4)별 Test ROC-AUC 비교")
    axes[1].tick_params(axis="x", rotation=12)
    axes[1].legend(fontsize=8, loc="lower right")

    colors_roc = ["#a5a5a5", "#2e75b6", "#ed7d31", "#70ad47", "#c00000"]
    for (exp_id, (fpr, tpr, auc_v)), c in zip(roc_curves_for_plot.items(), colors_roc):
        lw = 2.4 if "Exp-4" in exp_id or exp_id == "Exp-3" else 1.7
        axes[2].plot(fpr, tpr, label=f"{exp_id} (AUC={auc_v:.4f})", color=c, linewidth=lw)
    axes[2].plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.6)
    axes[2].set_xlabel("False Positive Rate (1 - Specificity)")
    axes[2].set_ylabel("True Positive Rate (Recall)")
    axes[2].set_title("(c) XGBoost 기준 실험별 Test ROC Curve 비교")
    axes[2].legend(fontsize=8.5, loc="lower right")

    plt.suptitle("5. 필수 실험 비교 (Base, Exp-1, Exp-2, Exp-3 및 최적 제안 Exp-4) 종합 결과", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig09_mandatory_exp_comparison.png", bbox_inches="tight")
    plt.close(fig)

    summary = {
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "top_12_features": feat_imp_series.head(12).round(4).to_dict(),
        "mandatory_best_by_test_auc": mandatory_df.sort_values("test_roc_auc", ascending=False)
        .head(5)[["실험", "모델", "test_accuracy", "test_f1", "test_roc_auc", "cv_roc_auc_mean"]]
        .to_dict(orient="records"),
    }
    save_json(summary, "step03_05_experiments_summary.json")
    return summary


if __name__ == "__main__":
    out = run_all_experiments()
    print("Experiments finished. Top results:", out["mandatory_best_by_test_auc"][:2])
