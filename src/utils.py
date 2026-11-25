"""Utility functions, plotting style setup, and evaluation helpers."""

from __future__ import annotations

import json
import os
import platform
from pathlib import Path
from typing import Any, Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate

RANDOM_STATE = 42
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
FIG_DIR = PROJECT_ROOT / "figures"
RESULTS_DIR = PROJECT_ROOT / "results"

PALETTE_MAIN = ["#1f4e79", "#2e75b6", "#ed7d31", "#70ad47", "#a5a5a5", "#ffc000"]
PALETTE_BINARY = ["#d95f02", "#1b9e77"]  # 0: 사망, 1: 생존


def configure_korean_font() -> str:
    """Configure Matplotlib to render Korean text cleanly without minus-sign breakage."""
    sns.set_theme(style="whitegrid", context="notebook")
    candidates = [
        "Apple SD Gothic Neo",
        "AppleGothic",
        "NanumGothic",
        "Malgun Gothic",
        "Noto Sans CJK KR",
        "Noto Sans KR",
    ]
    available = {f.name for f in fm.fontManager.ttflist}
    chosen = "DejaVu Sans"
    for cand in candidates:
        if cand in available:
            chosen = cand
            break

    plt.rcParams["font.family"] = chosen
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["figure.dpi"] = 150
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["axes.titleweight"] = "bold"
    plt.rcParams["axes.labelsize"] = 10
    return chosen


def ensure_dirs() -> None:
    """Ensure output directories exist."""
    for d in (DATA_DIR, FIG_DIR, RESULTS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def evaluate_classifier(
    pipeline: Any,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    cv_splits: int = 5,
) -> Dict[str, float]:
    """Fit pipeline, evaluate on train/test sets, and compute Stratified K-Fold CV metrics."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_res = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=1,
    )

    pipeline.fit(X_train, y_train)
    y_pred_test = pipeline.predict(X_test)
    y_pred_train = pipeline.predict(X_train)

    if hasattr(pipeline, "predict_proba"):
        y_prob_test = pipeline.predict_proba(X_test)[:, 1]
        y_prob_train = pipeline.predict_proba(X_train)[:, 1]
    else:
        y_prob_test = pipeline.decision_function(X_test)
        y_prob_train = pipeline.decision_function(X_train)

    return {
        "test_accuracy": float(accuracy_score(y_test, y_pred_test)),
        "test_precision": float(precision_score(y_test, y_pred_test, zero_division=0)),
        "test_recall": float(recall_score(y_test, y_pred_test, zero_division=0)),
        "test_f1": float(f1_score(y_test, y_pred_test, zero_division=0)),
        "test_roc_auc": float(roc_auc_score(y_test, y_prob_test)),
        "train_accuracy": float(accuracy_score(y_train, y_pred_train)),
        "train_f1": float(f1_score(y_train, y_pred_train, zero_division=0)),
        "train_roc_auc": float(roc_auc_score(y_train, y_prob_train)),
        "cv_accuracy_mean": float(np.mean(cv_res["test_accuracy"])),
        "cv_accuracy_std": float(np.std(cv_res["test_accuracy"])),
        "cv_precision_mean": float(np.mean(cv_res["test_precision"])),
        "cv_recall_mean": float(np.mean(cv_res["test_recall"])),
        "cv_f1_mean": float(np.mean(cv_res["test_f1"])),
        "cv_f1_std": float(np.std(cv_res["test_f1"])),
        "cv_roc_auc_mean": float(np.mean(cv_res["test_roc_auc"])),
        "cv_roc_auc_std": float(np.std(cv_res["test_roc_auc"])),
        "cv_train_roc_auc_mean": float(np.mean(cv_res["train_roc_auc"])),
        "overfit_gap_roc_auc": float(
            np.mean(cv_res["train_roc_auc"]) - np.mean(cv_res["test_roc_auc"])
        ),
        "overfit_gap_accuracy": float(
            np.mean(cv_res["train_accuracy"]) - np.mean(cv_res["test_accuracy"])
        ),
    }


def save_json(data: Any, filename: str) -> Path:
    """Save dictionary or list to JSON in RESULTS_DIR."""
    ensure_dirs()
    out = RESULTS_DIR / filename
    with open(out, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    return out
