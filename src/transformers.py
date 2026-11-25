"""Custom scikit-learn compatible transformers for leak-free Feature Engineering & Selection."""

from __future__ import annotations

import re
from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class TitanicFeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Domain-driven derived feature generator for the Titanic dataset.
    Operates row-wise without target leakage so it can sit directly inside sklearn Pipeline.

    Derived features created (6 total):
      1. FamilySize    : SibSp + Parch + 1
      2. IsAlone       : 1 if FamilySize == 1 else 0
      3. FarePerPerson : Fare / FamilySize
      4. Title         : Social title extracted from Name (Mr, Miss, Mrs, Master, Rare)
      5. HasCabin      : 1 if Cabin is non-null else 0
      6. Deck          : First letter of Cabin ('U' for Unknown)
      (+ AgeGroup is created post-Age-imputation via AgeGroupAdder or approximated safely)
    """

    def __init__(self, add_derived: bool = True, drop_identifiers: bool = True):
        self.add_derived = add_derived
        self.drop_identifiers = drop_identifiers

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "TitanicFeatureEngineer":
        if "Age" in X.columns:
            self.median_age_ = float(X["Age"].median())
        else:
            self.median_age_ = 28.0
        return self

    @staticmethod
    def _extract_title(name: str) -> str:
        if not isinstance(name, str):
            return "Rare"
        match = re.search(r",\s*([^\.]+)\.", name)
        if not match:
            return "Rare"
        raw = match.group(1).strip()
        mapping = {
            "Mlle": "Miss",
            "Ms": "Miss",
            "Mme": "Mrs",
            "Lady": "Rare",
            "the Countess": "Rare",
            "Countess": "Rare",
            "Capt": "Rare",
            "Col": "Rare",
            "Don": "Rare",
            "Dr": "Rare",
            "Major": "Rare",
            "Rev": "Rare",
            "Sir": "Rare",
            "Jonkheer": "Rare",
            "Dona": "Rare",
        }
        title = mapping.get(raw, raw)
        if title not in {"Mr", "Miss", "Mrs", "Master"}:
            return "Rare"
        return title

    @staticmethod
    def _bin_age(age_series: pd.Series) -> pd.Series:
        bins = [-np.inf, 12, 19, 35, 55, np.inf]
        labels = ["Child", "Teen", "YoungAdult", "Adult", "Senior"]
        return pd.cut(age_series, bins=bins, labels=labels).astype(str)

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if self.add_derived:
            df["FamilySize"] = df["SibSp"] + df["Parch"] + 1
            df["IsAlone"] = (df["FamilySize"] == 1).astype(int)
            df["FarePerPerson"] = df["Fare"] / df["FamilySize"].replace(0, 1)
            if "Name" in df.columns:
                df["Title"] = df["Name"].apply(self._extract_title)
            if "Cabin" in df.columns:
                df["HasCabin"] = df["Cabin"].notna().astype(int)
                df["Deck"] = (
                    df["Cabin"]
                    .fillna("U")
                    .astype(str)
                    .str.strip()
                    .str[0]
                    .replace({"T": "U"})
                )
            if "Age" in df.columns:
                # Fill temporarily ONLY for categorical AgeGroup binning using train median;
                # raw numeric 'Age' column retains its NaNs so downstream Imputers can be compared fairly!
                filled_age = df["Age"].fillna(self.median_age_)
                df["AgeGroup"] = self._bin_age(filled_age)

        if self.drop_identifiers:
            drop_cols = [c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns]
            df = df.drop(columns=drop_cols)

        # Ensure string/categorical columns use object dtype with standard np.nan for sklearn compatibility
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or df[col].dtype == object:
                df[col] = df[col].astype(object).where(df[col].notna(), np.nan)
        return df


class GroupMedianAgeImputer(BaseEstimator, TransformerMixin):
    """Imputes missing Age using group-wise median by (Pclass, Sex) learned strictly on train fold."""

    def fit(self, X: pd.DataFrame, y: Optional[pd.Series] = None) -> "GroupMedianAgeImputer":
        self.group_medians_ = X.groupby(["Pclass", "Sex"])["Age"].median().to_dict()
        self.global_median_ = float(X["Age"].median())
        self.embarked_mode_ = str(X["Embarked"].mode().iloc[0])
        self.fare_median_ = float(X["Fare"].median())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()
        if "Age" in df.columns:
            def _fill_age(row: pd.Series) -> float:
                if pd.notna(row["Age"]):
                    return float(row["Age"])
                return float(
                    self.group_medians_.get((row["Pclass"], row["Sex"]), self.global_median_)
                )

            df["Age"] = df.apply(_fill_age, axis=1)
        if "Embarked" in df.columns:
            df["Embarked"] = df["Embarked"].fillna(self.embarked_mode_)
        if "Fare" in df.columns:
            df["Fare"] = df["Fare"].fillna(self.fare_median_)
        return df


class CorrelationFeatureSelector(BaseEstimator, TransformerMixin):
    """Selects top-k features with highest absolute Pearson correlation with target y on train fold."""

    def __init__(self, k: int = 10):
        self.k = k

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CorrelationFeatureSelector":
        X_arr = np.asarray(X, dtype=float)
        y_arr = np.asarray(y, dtype=float)
        corrs = []
        for idx in range(X_arr.shape[1]):
            col = X_arr[:, idx]
            if np.std(col) < 1e-12:
                corrs.append(0.0)
            else:
                c = np.corrcoef(col, y_arr)[0, 1]
                corrs.append(0.0 if np.isnan(c) else abs(float(c)))
        self.scores_ = np.array(corrs)
        k_eff = min(self.k, X_arr.shape[1])
        top_idx = np.argsort(self.scores_)[::-1][:k_eff]
        self.support_mask_ = np.zeros(X_arr.shape[1], dtype=bool)
        self.support_mask_[top_idx] = True
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(X)[:, self.support_mask_]

    def get_support(self, indices: bool = False):
        if indices:
            return np.where(self.support_mask_)[0]
        return self.support_mask_
