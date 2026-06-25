from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

from . import config as C
from . import features as F
from . import metrics_qini as qini


class ConstantProbabilityModel:
    """Fallback classifier for a split that contains only one target class."""

    def __init__(self, probability: float):
        self.probability = float(probability)

    def fit(self, X: pd.DataFrame, y: pd.Series, cat_features: list[str] | None = None):
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        p1 = np.full(len(X), self.probability, dtype=float)
        return np.column_stack([1.0 - p1, p1])


def product_setup(product: str) -> dict[str, str]:
    """Return outcome/treatment setup for one product."""
    outcome_col, treated_value, control_value = qini.product_columns(product)
    return {
        "product": product.strip().lower(),
        "outcome_col": outcome_col,
        "treated_value": treated_value,
        "control_value": control_value,
    }


def default_catboost_params() -> dict[str, Any]:
    return {
        "iterations": 300,
        "learning_rate": 0.05,
        "depth": 5,
        "loss_function": "Logloss",
        "eval_metric": "AUC",
        "random_seed": C.SEED,
        "verbose": 0,
        "allow_writing_files": False,
    }


def _fit_binary_model(
    X: pd.DataFrame,
    y: pd.Series,
    cat_features: list[str],
    params: dict[str, Any],
):
    y_num = pd.to_numeric(y, errors="coerce").dropna()
    if y_num.empty:
        raise ValueError("Cannot fit model because target is empty after dropping missing labels.")

    classes = sorted(y_num.unique().tolist())
    if len(classes) == 1:
        return ConstantProbabilityModel(float(classes[0]))

    model = CatBoostClassifier(**params)
    model.fit(X.loc[y_num.index], y_num.astype(int), cat_features=cat_features)
    return model


def _prepare_catboost_matrix(
    df_engineered: pd.DataFrame,
    feature_columns: list[str] | None = None,
    cat_features: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    X, detected_cats = F.get_catboost_inputs(df_engineered)

    if feature_columns is not None:
        for col in feature_columns:
            if col not in X.columns:
                X[col] = np.nan
        X = X[feature_columns].copy()

    cat_features = detected_cats if cat_features is None else [c for c in cat_features if c in X.columns]

    for col in X.columns:
        if col in cat_features:
            X[col] = X[col].astype("object")
            X[col] = X[col].where(pd.notna(X[col]), "__NA__").astype(str)
        else:
            X[col] = pd.to_numeric(X[col], errors="coerce").astype("float32")

    return X, cat_features


@dataclass
class CatBoostTLearner:
    """
    T-Learner for the two-arm campaign design.

    For product='cash', uplift = P(cash_sign_30d | called cash) -
    P(cash_sign_30d | called card). For product='card', the arms are reversed.
    """

    product: str = "cash"
    model_params: dict[str, Any] | None = None
    miss_cols_: list[str] = field(default_factory=list, init=False)
    feature_columns_: list[str] = field(default_factory=list, init=False)
    cat_features_: list[str] = field(default_factory=list, init=False)
    treated_model_: Any = field(default=None, init=False, repr=False)
    control_model_: Any = field(default=None, init=False, repr=False)
    setup_: dict[str, str] = field(default_factory=dict, init=False)
    fitted_: bool = field(default=False, init=False)

    def fit(self, df: pd.DataFrame) -> "CatBoostTLearner":
        self.setup_ = product_setup(self.product)
        params = default_catboost_params()
        if self.model_params:
            params.update(self.model_params)

        outcome_col = self.setup_["outcome_col"]
        missing = sorted({C.TREATMENT_COL, outcome_col} - set(df.columns))
        if missing:
            raise KeyError(f"Missing required columns for model training: {missing}")

        train_df = df.dropna(subset=[outcome_col]).copy()
        if train_df.empty:
            raise ValueError(f"No labeled rows available for {outcome_col}.")

        train_df[C.TREATMENT_COL] = train_df[C.TREATMENT_COL].astype(str).str.strip().str.lower()
        df_eng, self.miss_cols_ = F.engineer_features(train_df)
        X, self.cat_features_ = _prepare_catboost_matrix(df_eng)
        self.feature_columns_ = X.columns.tolist()

        y = pd.to_numeric(train_df[outcome_col], errors="coerce")
        treated_mask = train_df[C.TREATMENT_COL] == self.setup_["treated_value"]
        control_mask = train_df[C.TREATMENT_COL] == self.setup_["control_value"]

        if treated_mask.sum() == 0 or control_mask.sum() == 0:
            raise ValueError("Training data must contain both treated and control campaign arms.")

        self.treated_model_ = _fit_binary_model(
            X.loc[treated_mask],
            y.loc[treated_mask],
            self.cat_features_,
            params,
        )
        self.control_model_ = _fit_binary_model(
            X.loc[control_mask],
            y.loc[control_mask],
            self.cat_features_,
            params,
        )
        self.fitted_ = True
        return self

    def _transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("CatBoostTLearner must be fitted before prediction.")

        df_eng, _ = F.engineer_features(df.copy(), miss_cols=self.miss_cols_)
        X, _ = _prepare_catboost_matrix(
            df_eng,
            feature_columns=self.feature_columns_,
            cat_features=self.cat_features_,
        )
        return X

    def predict_components(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self._transform(df)
        p_treated = self.treated_model_.predict_proba(X)[:, 1]
        p_control = self.control_model_.predict_proba(X)[:, 1]
        uplift = p_treated - p_control

        product = self.setup_["product"]
        return pd.DataFrame(
            {
                f"p_{product}_treated": p_treated,
                f"p_{product}_control": p_control,
                f"uplift_{product}": uplift,
            },
            index=df.index,
        )

    def predict_uplift(self, df: pd.DataFrame) -> np.ndarray:
        return self.predict_components(df)[f"uplift_{self.setup_['product']}"].to_numpy()

    def evaluate(self, df: pd.DataFrame) -> dict[str, object]:
        scores = self.predict_uplift(df)
        return qini.evaluate_qini(df, scores, product=self.setup_["product"])

    def save(self, path: str | Path) -> Path:
        if not self.fitted_:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        return path


def load_model(path: str | Path) -> CatBoostTLearner:
    return joblib.load(path)


def train_valid_split(
    df: pd.DataFrame,
    valid_size: float | None = None,
    random_state: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split data with treatment stratification for offline Qini evaluation."""
    valid_size = C.VALID_SIZE if valid_size is None else valid_size
    random_state = C.SEED if random_state is None else random_state
    stratify = df[C.TREATMENT_COL] if C.TREATMENT_COL in df.columns else None

    train_idx, valid_idx = train_test_split(
        df.index,
        test_size=valid_size,
        random_state=random_state,
        stratify=stratify,
    )
    return df.loc[train_idx].copy(), df.loc[valid_idx].copy()


def fit_t_learner(
    df: pd.DataFrame,
    product: str = "cash",
    model_params: dict[str, Any] | None = None,
) -> CatBoostTLearner:
    return CatBoostTLearner(product=product, model_params=model_params).fit(df)


def fit_cash_card_learners(
    df: pd.DataFrame,
    model_params: dict[str, Any] | None = None,
) -> dict[str, CatBoostTLearner]:
    return {
        "cash": fit_t_learner(df, product="cash", model_params=model_params),
        "card": fit_t_learner(df, product="card", model_params=model_params),
    }


def score_cash_card(
    df: pd.DataFrame,
    learners: dict[str, CatBoostTLearner],
) -> pd.DataFrame:
    """Return customer_id plus cash/card uplift scores and recommendation."""
    out_cols = [C.ID_COL] if C.ID_COL in df.columns else []
    out = df[out_cols].copy()

    for product, learner in learners.items():
        out = out.join(learner.predict_components(df))

    score_cols = [c for c in ["uplift_cash", "uplift_card"] if c in out.columns]
    if score_cols:
        out["best_product"] = out[score_cols].idxmax(axis=1).str.replace("uplift_", "", regex=False)
        out["best_uplift"] = out[score_cols].max(axis=1)
        out["recommend_treatment"] = np.where(
            out["best_uplift"] > C.DEFAULT_UPLIFT_THRESHOLD,
            out["best_product"],
            "no_call",
        )

    return out


if __name__ == "__main__":
    from . import data_io as io

    df_train = io.load_train().head(5000)
    tr, va = train_valid_split(df_train, valid_size=0.25)
    learner = fit_t_learner(
        tr,
        product="cash",
        model_params={"iterations": 50, "depth": 4},
    )
    result = learner.evaluate(va)
    print(result["summary"])
    print(result["topk"])