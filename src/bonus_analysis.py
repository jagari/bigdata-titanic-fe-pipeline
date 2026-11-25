"""Section 8 Bonus Analyses:
1. Pipeline + GridSearchCV hyperparameter tuning
2. Advanced Multi-Method Feature Importance Visualization (Gini, Permutation, Mean |SHAP|, Mutual Info)
3. SHAP Global & Local Explainability Analysis
4. LIME Local Instance Explainability Analysis
5. FLAML AutoML Comparison (Raw AutoML vs Domain-FE AutoML vs Manual Tuned Pipeline)
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import shap
from flaml import AutoML
from lightgbm import LGBMClassifier
from lime.lime_tabular import LimeTabularExplainer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.experiments import get_models, make_preprocessor, make_selector
from src.transformers import GroupMedianAgeImputer, TitanicFeatureEngineer
from src.utils import (
    DATA_DIR,
    FIG_DIR,
    PALETTE_BINARY,
    PALETTE_MAIN,
    RANDOM_STATE,
    RESULTS_DIR,
    configure_korean_font,
    ensure_dirs,
    evaluate_classifier,
    save_json,
)

warnings.filterwarnings("ignore")


def run_bonus_analyses() -> Dict[str, Any]:
    """Execute all bonus items (+10 pts) and save tables + Figures 10-13."""
    ensure_dirs()
    configure_korean_font()

    df = pd.read_csv(DATA_DIR / "titanic.csv")
    X = df.drop(columns=["Survived"])
    y = df["Survived"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    # =========================================================================
    # 1. Pipeline + GridSearchCV Hyperparameter Tuning
    # =========================================================================
    search_configs = [
        (
            "Logistic Regression",
            LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
            {
                "select__max_features": [12, 16, 22],
                "clf__C": [0.1, 0.5, 1.0, 2.0, 5.0],
            },
        ),
        (
            "Random Forest",
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=1),
            {
                "select__max_features": [12, 16, 22],
                "clf__n_estimators": [150, 300],
                "clf__max_depth": [5, 7, 10],
                "clf__min_samples_leaf": [2, 4],
            },
        ),
        (
            "XGBoost",
            XGBClassifier(eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=1),
            {
                "select__max_features": [14, 20],
                "clf__n_estimators": [120, 200],
                "clf__max_depth": [3, 4, 5],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__subsample": [0.85],
                "clf__colsample_bytree": [0.85],
            },
        ),
        (
            "LightGBM",
            LGBMClassifier(random_state=RANDOM_STATE, n_jobs=1, verbose=-1),
            {
                "select__max_features": [14, 20],
                "clf__n_estimators": [120, 200],
                "clf__max_depth": [3, 5, 7],
                "clf__learning_rate": [0.03, 0.05, 0.1],
                "clf__num_leaves": [15, 31],
            },
        ),
    ]

    gs_rows = []
    best_pipelines: Dict[str, Pipeline] = {}
    for model_name, estimator, param_grid in search_configs:
        base_pipe = Pipeline(
            [
                (
                    "prep",
                    make_preprocessor(
                        use_derived=True,
                        num_imputer="median",
                        encoder="onehot",
                        scaler="robust",
                        use_group_age=True,
                    ),
                ),
                ("select", make_selector("rf_importance", k=14)),
                ("clf", get_models()[model_name]),
            ]
        )
        default_metrics = evaluate_classifier(base_pipe, X_train, y_train, X_test, y_test)

        tune_pipe = Pipeline(
            [
                (
                    "prep",
                    make_preprocessor(
                        use_derived=True,
                        num_imputer="median",
                        encoder="onehot",
                        scaler="robust",
                        use_group_age=True,
                    ),
                ),
                ("select", make_selector("rf_importance", k=14)),
                ("clf", estimator),
            ]
        )
        gs = GridSearchCV(
            tune_pipe,
            param_grid=param_grid,
            cv=cv5,
            scoring="roc_auc",
            n_jobs=1,
            refit=True,
        )
        gs.fit(X_train, y_train)
        best_pipe = gs.best_estimator_
        best_pipelines[model_name] = best_pipe

        y_pred = best_pipe.predict(X_test)
        y_prob = best_pipe.predict_proba(X_test)[:, 1]

        gs_rows.append(
            {
                "모델": model_name,
                "튜닝 전 CV AUC": round(default_metrics["cv_roc_auc_mean"], 4),
                "튜닝 후 CV AUC": round(float(gs.best_score_), 4),
                "튜닝 전 Test AUC": round(default_metrics["test_roc_auc"], 4),
                "튜닝 후 Test AUC": round(float(roc_auc_score(y_test, y_prob)), 4),
                "튜닝 후 Accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
                "튜닝 후 Precision": round(float(precision_score(y_test, y_pred)), 4),
                "튜닝 후 Recall": round(float(recall_score(y_test, y_pred)), 4),
                "튜닝 후 F1": round(float(f1_score(y_test, y_pred)), 4),
                "최적 하이퍼파라미터": str(gs.best_params_),
            }
        )
    gs_df = pd.DataFrame(gs_rows)
    gs_df.to_csv(RESULTS_DIR / "bonus_gridsearch.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # 2 & 3. Prepare Clean Feature Matrix for Advanced Importance, SHAP & LIME
    # =========================================================================
    prep_full = make_preprocessor(
        use_derived=True,
        num_imputer="median",
        encoder="onehot",
        scaler="robust",
        use_group_age=True,
    )
    X_tr_arr = prep_full.fit_transform(X_train, y_train)
    X_te_arr = prep_full.transform(X_test)
    feat_names = [
        n.replace("num__", "").replace("cat__", "")
        for n in prep_full.named_steps["prep"].get_feature_names_out()
    ]
    X_tr_df = pd.DataFrame(X_tr_arr, columns=feat_names, index=X_train.index)
    X_te_df = pd.DataFrame(X_te_arr, columns=feat_names, index=X_test.index)

    lgbm_xai = LGBMClassifier(
        n_estimators=180,
        max_depth=4,
        learning_rate=0.05,
        num_leaves=15,
        subsample=0.85,
        colsample_bytree=0.85,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbose=-1,
    )
    lgbm_xai.fit(X_tr_df, y_train)

    rf_xai = RandomForestClassifier(
        n_estimators=250, max_depth=7, min_samples_leaf=2, random_state=RANDOM_STATE, n_jobs=1
    )
    rf_xai.fit(X_tr_df, y_train)

    # SHAP values using LightGBM TreeExplainer
    explainer = shap.TreeExplainer(lgbm_xai)
    shap_vals_raw = explainer.shap_values(X_te_df)
    if isinstance(shap_vals_raw, list):
        shap_vals = np.asarray(shap_vals_raw[1])
        base_val = float(explainer.expected_value[1])
    elif shap_vals_raw.ndim == 3:
        shap_vals = shap_vals_raw[:, :, 1]
        base_val = float(explainer.expected_value[1])
    else:
        shap_vals = np.asarray(shap_vals_raw)
        ev = explainer.expected_value
        base_val = float(ev[0] if isinstance(ev, (list, np.ndarray)) else ev)

    mean_abs_shap = pd.Series(np.abs(shap_vals).mean(axis=0), index=feat_names)
    rf_gini = pd.Series(rf_xai.feature_importances_, index=feat_names)

    perm_res = permutation_importance(
        rf_xai, X_te_df, y_test, scoring="roc_auc", n_repeats=20, random_state=RANDOM_STATE, n_jobs=1
    )
    perm_mean = pd.Series(perm_res.importances_mean, index=feat_names)
    perm_std = pd.Series(perm_res.importances_std, index=feat_names)

    mi_scores = pd.Series(
        mutual_info_classif(X_tr_df, y_train, random_state=RANDOM_STATE), index=feat_names
    )

    imp_comp_df = pd.DataFrame(
        {
            "RF_Gini": rf_gini,
            "Permutation_AUC": perm_mean.clip(lower=0),
            "Mean_Abs_SHAP": mean_abs_shap,
            "Mutual_Info": mi_scores,
        }
    )
    # Normalized 0~1 for multi-method consensus comparison
    imp_norm_df = imp_comp_df.div(imp_comp_df.max(axis=0).replace(0, 1), axis=1)
    imp_norm_df["Consensus_Score"] = imp_norm_df.mean(axis=1)
    imp_norm_df = imp_norm_df.sort_values("Consensus_Score", ascending=False)
    imp_norm_df.to_csv(RESULTS_DIR / "bonus_multi_feature_importance.csv", encoding="utf-8-sig")

    # =========================================================================
    # Figure 10: Advanced Multi-Angle Feature Importance Visualization
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.8))

    top12_names = imp_norm_df.head(12).index[::-1]

    # (a) Permutation Importance with Error Bars
    axes[0].barh(
        top12_names,
        perm_mean.loc[top12_names],
        xerr=perm_std.loc[top12_names],
        color="#2e75b6",
        edgecolor="black",
        alpha=0.88,
        capsize=3,
    )
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[0].set_xlabel("ROC-AUC 감소량 (평균 ± 표준편차)")
    axes[0].set_title("(a) Permutation Importance (검증셋 기반)")

    # (b) Mean |SHAP| Importance
    axes[1].barh(
        top12_names,
        mean_abs_shap.loc[top12_names],
        color="#ed7d31",
        edgecolor="black",
        alpha=0.88,
    )
    axes[1].set_xlabel("평균 |SHAP Value| (로그 오즈 기여도)")
    axes[1].set_title("(b) SHAP 기반 전역 특성 영향력")

    # (c) Normalized 4-Method Consensus Heatmap
    hm_data = imp_norm_df.head(12)[["RF_Gini", "Permutation_AUC", "Mean_Abs_SHAP", "Mutual_Info"]]
    hm_data.columns = ["RF 불순도", "Permutation", "Mean |SHAP|", "상호정보량(MI)"]
    sns.heatmap(
        hm_data,
        annot=True,
        fmt=".2f",
        cmap="YlGnBu",
        cbar_kws={"label": "정규화 중요도 (0~1)"},
        linewidths=0.6,
        ax=axes[2],
    )
    axes[2].set_title("(c) 4대 중요도 지표 정규화 합의(Consensus) 히트맵")

    plt.suptitle("가산점 심화: 다각도 특성 중요도(Feature Importance) 교차 검증 및 고도화 시각화", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig10_advanced_feature_importance.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 11: SHAP Beeswarm Summary & Local Instance Waterfall-style Plot
    # =========================================================================
    fig = plt.figure(figsize=(15.5, 6.2))
    ax1 = fig.add_subplot(1, 2, 1)
    plt.sca(ax1)
    shap.summary_plot(
        shap_vals,
        X_te_df,
        max_display=12,
        show=False,
        plot_size=None,
        color_bar_label="특성 값 크기 (낮음 → 높음)",
    )
    configure_korean_font()
    ax1.set_title("(a) SHAP Summary Beeswarm Plot (상위 12개 변수)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("SHAP Value (생존 예측 방향 기여도: +생존 / -사망)")

    # Local explanation for 1 high-prob survivor in Test set
    probs_te = lgbm_xai.predict_proba(X_te_df)[:, 1]
    surv_idx = int(np.argmax(probs_te))
    sv_instance = pd.Series(shap_vals[surv_idx], index=feat_names)
    top_local = sv_instance.reindex(sv_instance.abs().sort_values(ascending=False).head(10).index)[::-1]

    ax2 = fig.add_subplot(1, 2, 2)
    bar_colors = ["#1b9e77" if v > 0 else "#d95f02" for v in top_local.values]
    bars = ax2.barh(top_local.index, top_local.values, color=bar_colors, edgecolor="black", alpha=0.9)
    ax2.axvline(0, color="black", linewidth=1)
    ax2.set_xlabel("개별 승객에 대한 SHAP 기여도 (로그 오즈)")
    ax2.set_title(
        f"(b) 개별 생존 승객 국소 설명 (예측 생존확률={probs_te[surv_idx]*100:.1f}%, 실제={int(y_test.iloc[surv_idx])})",
        fontsize=11.5,
        fontweight="bold",
    )
    for bar, v in zip(bars, top_local.values):
        offset = 0.04 if v >= 0 else -0.04
        ha = "left" if v >= 0 else "right"
        ax2.text(
            v + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{v:+.2f}",
            va="center",
            ha=ha,
            fontsize=8.5,
            fontweight="bold",
        )
    ax2.set_xlim(min(top_local.values) - 0.55, max(top_local.values) + 0.55)

    plt.suptitle("가산점 심화: SHAP 기반 전역(Global) 및 개별 승객(Local) 설명 가능성(XAI) 분석", fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig11_shap_analysis.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # Figure 12: LIME Local Explanations (Predicted Survivor vs Non-Survivor)
    # =========================================================================
    lime_explainer = LimeTabularExplainer(
        training_data=X_tr_df.values,
        feature_names=feat_names,
        class_names=["사망(0)", "생존(1)"],
        mode="classification",
        random_state=RANDOM_STATE,
    )
    died_idx = int(np.argmin(probs_te))

    exp_surv = lime_explainer.explain_instance(
        X_te_df.iloc[surv_idx].values, rf_xai.predict_proba, num_features=8
    )
    exp_died = lime_explainer.explain_instance(
        X_te_df.iloc[died_idx].values, rf_xai.predict_proba, num_features=8
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))
    for ax, lime_exp, idx_case, case_title in [
        (axes[0], exp_surv, surv_idx, "사례 A: 생존 고확률 승객 LIME 규칙 기여도"),
        (axes[1], exp_died, died_idx, "사례 B: 사망 고확률 승객 LIME 규칙 기여도"),
    ]:
        items = lime_exp.as_list()[::-1]
        rules = [it[0] for it in items]
        weights = [it[1] for it in items]
        colors = ["#1b9e77" if w > 0 else "#d95f02" for w in weights]
        ax.barh(rules, weights, color=colors, edgecolor="black", alpha=0.88)
        ax.axvline(0, color="black", linewidth=1)
        p_surv = rf_xai.predict_proba(X_te_df.iloc[[idx_case]])[0, 1]
        ax.set_title(f"{case_title}\n(RF 생존확률={p_surv*100:.1f}%, 실제 타겟={int(y_test.iloc[idx_case])})", fontsize=11)
        ax.set_xlabel("LIME 국소 선형 근사 가중치 (+생존 / -사망)")

    plt.suptitle("가산점 심화: LIME(Local Interpretable Model-agnostic Explanations) 개별 예측 해석", fontsize=13.5, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig12_lime_explanations.png", bbox_inches="tight")
    plt.close(fig)

    # =========================================================================
    # 5. FLAML AutoML Comparison (Raw Features vs Domain FE Features)
    # =========================================================================
    raw_cols = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
    X_tr_raw = X_train[raw_cols].copy()
    X_te_raw = X_test[raw_cols].copy()
    for c in ["Sex", "Embarked"]:
        X_tr_raw[c] = X_tr_raw[c].fillna("Missing").astype(str)
        X_te_raw[c] = X_te_raw[c].fillna("Missing").astype(str)

    automl_raw = AutoML()
    automl_raw.fit(
        X_train=X_tr_raw,
        y_train=y_train,
        task="classification",
        metric="roc_auc",
        time_budget=20,
        eval_method="cv",
        n_splits=5,
        seed=RANDOM_STATE,
        verbose=0,
    )
    prob_automl_raw = automl_raw.predict_proba(X_te_raw)[:, 1]
    pred_automl_raw = automl_raw.predict(X_te_raw)

    # AutoML with our Domain Feature Engineering matrix
    automl_fe = AutoML()
    automl_fe.fit(
        X_train=X_tr_df,
        y_train=y_train,
        task="classification",
        metric="roc_auc",
        time_budget=20,
        eval_method="cv",
        n_splits=5,
        seed=RANDOM_STATE,
        verbose=0,
    )
    prob_automl_fe = automl_fe.predict_proba(X_te_df)[:, 1]
    pred_automl_fe = automl_fe.predict(X_te_df)

    # Compare with best manual GridSearchCV pipeline
    best_manual_row = gs_df.sort_values("튜닝 후 CV AUC", ascending=False).iloc[0]
    best_manual_name = best_manual_row["모델"]
    best_manual_pipe = best_pipelines[best_manual_name]
    prob_manual = best_manual_pipe.predict_proba(X_test)[:, 1]
    pred_manual = best_manual_pipe.predict(X_test)

    automl_comp_rows = [
        {
            "비교 방식": f"FLAML AutoML (원본 변수만 사용, Best={automl_raw.best_estimator})",
            "CV ROC-AUC": round(float(1.0 - automl_raw.best_loss), 4),
            "Test Accuracy": round(float(accuracy_score(y_test, pred_automl_raw)), 4),
            "Test Precision": round(float(precision_score(y_test, pred_automl_raw)), 4),
            "Test Recall": round(float(recall_score(y_test, pred_automl_raw)), 4),
            "Test F1": round(float(f1_score(y_test, pred_automl_raw)), 4),
            "Test ROC-AUC": round(float(roc_auc_score(y_test, prob_automl_raw)), 4),
        },
        {
            "비교 방식": f"FLAML AutoML (도메인 파생변수 결합, Best={automl_fe.best_estimator})",
            "CV ROC-AUC": round(float(1.0 - automl_fe.best_loss), 4),
            "Test Accuracy": round(float(accuracy_score(y_test, pred_automl_fe)), 4),
            "Test Precision": round(float(precision_score(y_test, pred_automl_fe)), 4),
            "Test Recall": round(float(recall_score(y_test, pred_automl_fe)), 4),
            "Test F1": round(float(f1_score(y_test, pred_automl_fe)), 4),
            "Test ROC-AUC": round(float(roc_auc_score(y_test, prob_automl_fe)), 4),
        },
        {
            "비교 방식": f"수동 설계 FE Pipeline + GridSearchCV ({best_manual_name})",
            "CV ROC-AUC": float(best_manual_row["튜닝 후 CV AUC"]),
            "Test Accuracy": round(float(accuracy_score(y_test, pred_manual)), 4),
            "Test Precision": round(float(precision_score(y_test, pred_manual)), 4),
            "Test Recall": round(float(recall_score(y_test, pred_manual)), 4),
            "Test F1": round(float(f1_score(y_test, pred_manual)), 4),
            "Test ROC-AUC": round(float(roc_auc_score(y_test, prob_manual)), 4),
        },
    ]
    automl_df = pd.DataFrame(automl_comp_rows)
    automl_df.to_csv(RESULTS_DIR / "bonus_automl_comparison.csv", index=False, encoding="utf-8-sig")

    # =========================================================================
    # Figure 13: GridSearchCV Before/After & AutoML Comparison Chart
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2))

    x_pos = np.arange(len(gs_df))
    w = 0.35
    axes[0].bar(
        x_pos - w / 2,
        gs_df["튜닝 전 CV AUC"],
        width=w,
        label="튜닝 전 (기본 하이퍼파라미터)",
        color="#a5a5a5",
        edgecolor="black",
    )
    axes[0].bar(
        x_pos + w / 2,
        gs_df["튜닝 후 CV AUC"],
        width=w,
        label="GridSearchCV 최적 튜닝 후",
        color="#1f4e79",
        edgecolor="black",
    )
    axes[0].set_xticks(x_pos)
    axes[0].set_xticklabels(gs_df["모델"], rotation=10)
    axes[0].set_ylim(0.83, 0.90)
    axes[0].set_ylabel("5-Fold CV ROC-AUC")
    axes[0].set_title("(a) Pipeline + GridSearchCV 튜닝 전/후 성능 비교")
    axes[0].legend(fontsize=8.8, loc="lower right")
    for i, r in gs_df.iterrows():
        axes[0].text(
            i + w / 2,
            r["튜닝 후 CV AUC"] + 0.0015,
            f"{r['튜닝 후 CV AUC']:.4f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    short_labels = ["AutoML\n(원본 특성)", "AutoML\n(+도메인 파생변수)", f"수동 FE Pipeline\n+GridSearch({best_manual_name})"]
    x2 = np.arange(3)
    axes[1].bar(
        x2 - w / 2,
        automl_df["CV ROC-AUC"],
        width=w,
        label="5-Fold CV ROC-AUC",
        color="#2e75b6",
        edgecolor="black",
    )
    axes[1].bar(
        x2 + w / 2,
        automl_df["Test ROC-AUC"],
        width=w,
        label="Test ROC-AUC",
        color="#70ad47",
        edgecolor="black",
    )
    axes[1].set_xticks(x2)
    axes[1].set_xticklabels(short_labels)
    axes[1].set_ylim(0.80, 0.92)
    axes[1].set_ylabel("ROC-AUC")
    axes[1].set_title("(b) FLAML AutoML vs 도메인 특성 공학 파이프라인 비교")
    axes[1].legend(fontsize=8.8, loc="lower right")
    for i, r in automl_df.iterrows():
        axes[1].text(
            i + w / 2,
            r["Test ROC-AUC"] + 0.002,
            f"{r['Test ROC-AUC']:.4f}",
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )

    plt.suptitle("가산점 심화: GridSearchCV 최적화 및 AutoML(FLAML) 비교 실험 결과", fontsize=14, fontweight="bold", y=1.03)
    plt.tight_layout()
    plt.savefig(FIG_DIR / "fig13_automl_and_gridsearch.png", bbox_inches="tight")
    plt.close(fig)

    bonus_summary = {
        "gridsearch": gs_rows,
        "automl": automl_comp_rows,
        "automl_raw_best_estimator": str(automl_raw.best_estimator),
        "automl_fe_best_estimator": str(automl_fe.best_estimator),
        "top_10_consensus_features": imp_norm_df.head(10)["Consensus_Score"].round(4).to_dict(),
    }
    save_json(bonus_summary, "bonus_summary.json")
    return bonus_summary


if __name__ == "__main__":
    res = run_bonus_analyses()
    print("Bonus analyses finished:", res["automl"])
