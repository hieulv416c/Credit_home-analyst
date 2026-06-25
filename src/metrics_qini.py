from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from . import config as C


def product_columns(product: str) -> tuple[str, str, str]:
    """
    Return outcome, treated value, and control value for one product.

    product='cash': measure effect of calling cash vs card on cash_sign_30d.
    product='card': measure effect of calling card vs cash on card_sign_30d.
    """
    product = str(product).strip().lower()
    if product == "cash":
        return C.CASH_OUTCOME, C.TREAT_CASH, C.TREAT_CARD
    if product == "card":
        return C.CARD_OUTCOME, C.TREAT_CARD, C.TREAT_CASH
    raise ValueError("product must be either 'cash' or 'card'.")


def prepare_qini_frame(
    df: pd.DataFrame,
    uplift_score: str | Iterable[float],
    product: str = "cash",
    score_col: str = "uplift_score",
) -> pd.DataFrame:
    """
    Build a clean evaluation frame with y, treatment flag, and uplift score.

    uplift_score can be a dataframe column name or an array-like score vector.
    Higher score means the customer should be targeted earlier.
    """
    outcome_col, treated_value, control_value = product_columns(product)
    required = {C.TREATMENT_COL, outcome_col}
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"Missing required columns for Qini evaluation: {missing}")

    out = df[[C.TREATMENT_COL, outcome_col]].copy()

    if isinstance(uplift_score, str):
        if uplift_score not in df.columns:
            raise KeyError(f"Score column not found: {uplift_score}")
        out[score_col] = df[uplift_score].to_numpy()
    else:
        score_values = np.asarray(list(uplift_score), dtype=float)
        if len(score_values) != len(df):
            raise ValueError("uplift_score length must match df length.")
        out[score_col] = score_values

    treatment_norm = out[C.TREATMENT_COL].astype("string").str.strip().str.lower()
    out["treatment_flag"] = np.select(
        [
            treatment_norm == str(treated_value).lower(),
            treatment_norm == str(control_value).lower(),
        ],
        [1, 0],
        default=np.nan,
    )
    out["outcome"] = pd.to_numeric(out[outcome_col], errors="coerce")
    out[score_col] = pd.to_numeric(out[score_col], errors="coerce")

    out = out.dropna(subset=["treatment_flag", "outcome", score_col]).copy()
    out["treatment_flag"] = out["treatment_flag"].astype(int)
    out["outcome"] = out["outcome"].astype(float)

    invalid_y = ~out["outcome"].isin([0.0, 1.0])
    if invalid_y.any():
        raise ValueError(f"Outcome column {outcome_col} must contain only 0/1 labels.")

    if out["treatment_flag"].nunique() < 2:
        raise ValueError("Qini evaluation needs both treated and control rows.")

    return out


def qini_curve(
    df: pd.DataFrame,
    uplift_score: str | Iterable[float],
    product: str = "cash",
    score_col: str = "uplift_score",
) -> pd.DataFrame:
    """
    Compute the cumulative Qini curve.

    qini_gain at rank k = treated responders - control responders * n_treat / n_ctrl.
    """
    frame = prepare_qini_frame(df, uplift_score, product=product, score_col=score_col)
    frame = frame.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)

    t = frame["treatment_flag"].to_numpy(dtype=int)
    y = frame["outcome"].to_numpy(dtype=float)
    n = len(frame)

    cum_treat = np.cumsum(t)
    cum_control = np.cumsum(1 - t)
    cum_y_treat = np.cumsum(y * t)
    cum_y_control = np.cumsum(y * (1 - t))

    with np.errstate(divide="ignore", invalid="ignore"):
        control_scaled = np.divide(
            cum_y_control * cum_treat,
            cum_control,
            out=np.zeros(n, dtype=float),
            where=cum_control > 0,
        )
    qini_gain = cum_y_treat - control_scaled

    total_gain = float(qini_gain[-1])
    population = np.arange(1, n + 1) / n
    random_gain = population * total_gain

    curve = pd.DataFrame(
        {
            "rank": np.arange(1, n + 1),
            "population_pct": population,
            "n_treat": cum_treat,
            "n_control": cum_control,
            "cum_y_treat": cum_y_treat,
            "cum_y_control": cum_y_control,
            "qini_gain": qini_gain,
            "random_gain": random_gain,
        }
    )

    start = pd.DataFrame(
        {
            "rank": [0],
            "population_pct": [0.0],
            "n_treat": [0],
            "n_control": [0],
            "cum_y_treat": [0.0],
            "cum_y_control": [0.0],
            "qini_gain": [0.0],
            "random_gain": [0.0],
        }
    )
    return pd.concat([start, curve], ignore_index=True)


def qini_auc(curve: pd.DataFrame) -> dict[str, float]:
    """Return AUUC, random area, and Qini coefficient from a qini_curve output."""
    x = curve["population_pct"].to_numpy(dtype=float)
    qini = curve["qini_gain"].to_numpy(dtype=float)
    random = curve["random_gain"].to_numpy(dtype=float)

    auuc = float(np.trapz(qini, x))
    random_area = float(np.trapz(random, x))
    qini_coef = auuc - random_area

    return {
        "auuc": auuc,
        "random_area": random_area,
        "qini_coef": qini_coef,
        "max_qini_gain": float(np.nanmax(qini)),
        "final_qini_gain": float(qini[-1]),
    }


def uplift_at_k(
    df: pd.DataFrame,
    uplift_score: str | Iterable[float],
    product: str = "cash",
    topk: Iterable[float] | None = None,
    score_col: str = "uplift_score",
) -> pd.DataFrame:
    """Report observed uplift and incremental responders for top-K ranked customers."""
    topk = list(C.TOPK_PERCENTILES if topk is None else topk)
    frame = prepare_qini_frame(df, uplift_score, product=product, score_col=score_col)
    frame = frame.sort_values(score_col, ascending=False, kind="mergesort").reset_index(drop=True)

    rows = []
    n_total = len(frame)
    for k in topk:
        if not 0 < float(k) <= 1:
            raise ValueError("All topk values must be in (0, 1].")

        n_top = max(1, int(np.ceil(n_total * float(k))))
        sub = frame.iloc[:n_top]
        treated = sub[sub["treatment_flag"] == 1]
        control = sub[sub["treatment_flag"] == 0]

        treat_rate = float(treated["outcome"].mean()) if len(treated) else np.nan
        control_rate = float(control["outcome"].mean()) if len(control) else np.nan
        uplift = treat_rate - control_rate if np.isfinite(treat_rate) and np.isfinite(control_rate) else np.nan
        incremental = uplift * len(treated) if np.isfinite(uplift) else np.nan

        rows.append(
            {
                "top_pct": float(k),
                "n_total": int(n_top),
                "n_treat": int(len(treated)),
                "n_control": int(len(control)),
                "treat_response_rate": treat_rate,
                "control_response_rate": control_rate,
                "uplift": uplift,
                "incremental_responders": incremental,
            }
        )

    return pd.DataFrame(rows)


def evaluate_qini(
    df: pd.DataFrame,
    uplift_score: str | Iterable[float],
    product: str = "cash",
    topk: Iterable[float] | None = None,
    score_col: str = "uplift_score",
) -> dict[str, object]:
    """Compute curve, Qini summary metrics, and top-K uplift table."""
    curve = qini_curve(df, uplift_score, product=product, score_col=score_col)
    return {
        "product": product,
        "curve": curve,
        "summary": qini_auc(curve),
        "topk": uplift_at_k(
            df,
            uplift_score,
            product=product,
            topk=topk,
            score_col=score_col,
        ),
    }


def plot_qini_curve(
    curve: pd.DataFrame,
    product: str = "cash",
    name: str | None = None,
) -> Path:
    """Save a Qini curve chart to outputs/charts and return its path."""
    import matplotlib.pyplot as plt

    C.CHART_DIR.mkdir(parents=True, exist_ok=True)
    out_name = name or f"qini_curve_{product}"
    path = C.CHART_DIR / (out_name if out_name.endswith(".png") else f"{out_name}.png")

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curve["population_pct"], curve["qini_gain"], label="Model", color="#2a7fff", lw=2)
    ax.plot(curve["population_pct"], curve["random_gain"], label="Random", color="gray", ls="--")
    ax.axhline(0, color="black", lw=0.8, alpha=0.4)
    ax.set_title(f"Qini Curve - {product.upper()}")
    ax.set_xlabel("Targeted population share")
    ax.set_ylabel("Incremental responders")
    ax.grid(True, alpha=0.15)
    ax.legend()
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return path


if __name__ == "__main__":
    from . import data_io as io

    df_train = io.load_train()
    # Smoke-test baseline: use a deterministic random score, not a real model.
    rng = np.random.default_rng(C.SEED)
    scores = rng.normal(size=len(df_train))
    result = evaluate_qini(df_train, scores, product="cash")
    print(result["summary"])
    print(result["topk"])