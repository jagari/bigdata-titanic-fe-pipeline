"""Master end-to-end execution script for full reproducibility."""

from __future__ import annotations

from src.eda import run_step1_and_step2
from src.experiments import run_all_experiments
from src.bonus_analysis import run_bonus_analyses


def main() -> None:
    print("[1/3] Running STEP 01 & STEP 02 (Data Prep & EDA)...")
    eda_res = run_step1_and_step2()
    print(f"      Done -> shape={eda_res['shape']}, survival_rate={eda_res['target_distribution']['survival_rate_pct']}%")

    print("[2/3] Running STEP 03, STEP 04 & STEP 05 (Feature Engineering, Selection & Model Experiments)...")
    exp_res = run_all_experiments()
    print(f"      Done -> top result={exp_res['mandatory_best_by_test_auc'][0]}")

    print("[3/3] Running Section 8 Bonus Analyses (GridSearchCV, SHAP, LIME, AutoML, Multi-Importance)...")
    bonus_res = run_bonus_analyses()
    print(f"      Done -> AutoML & GridSearch completed ({len(bonus_res['gridsearch'])} models tuned).")


if __name__ == "__main__":
    main()
