from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from . import config as C
from . import data_io as io
from . import metrics_qini as qini
from . import models
from . import validation


@dataclass
class PipelineArtifacts:
    train_scores: pd.DataFrame
    valid_scores: pd.DataFrame
    test_scores: pd.DataFrame
    qini_results: dict[str, dict[str, object]]
    health_reports: dict[str, validation.Report]
    learners: dict[str, models.CatBoostTLearner]
    output_paths: dict[str, Path]


def _write_report_table(df: pd.DataFrame, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path


def _save_qini_outputs(
    qini_results: dict[str, dict[str, object]],
    prefix: str = "valid",
) -> dict[str, Path]:
    paths: dict[str, Path] = {}

    for product, result in qini_results.items():
        curve = result["curve"]
        topk = result["topk"]
        summary = pd.DataFrame([result["summary"]])

        curve_path = C.TABLE_DIR / f"qini_curve_{product}_{prefix}.csv"
        topk_path = C.TABLE_DIR / f"uplift_topk_{product}_{prefix}.csv"
        summary_path = C.TABLE_DIR / f"qini_summary_{product}_{prefix}.csv"

        _write_report_table(curve, curve_path)
        _write_report_table(topk, topk_path)
        _write_report_table(summary, summary_path)
        chart_path = qini.plot_qini_curve(curve, product=product, name=f"qini_curve_{product}_{prefix}")

        paths[f"{product}_curve"] = curve_path
        paths[f"{product}_topk"] = topk_path
        paths[f"{product}_summary"] = summary_path
        paths[f"{product}_chart"] = chart_path

    return paths


def _save_models(learners: dict[str, models.CatBoostTLearner], prefix: str = "catboost_tlearner") -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for product, learner in learners.items():
        paths[f"model_{product}"] = learner.save(C.MODEL_DIR / f"{prefix}_{product}.joblib")
    return paths


def run_health_checks(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame | None = None,
    print_report: bool = True,
) -> dict[str, validation.Report]:
    reports = {"train": validation.check_data_health(df_train, is_train=True)}
    if df_test is not None:
        reports["test"] = validation.check_data_health(df_test, is_train=False)

    if print_report:
        for name, report in reports.items():
            validation.print_health_report(report, title=name)

    return reports


def train_and_evaluate(
    df_train: pd.DataFrame,
    model_params: dict[str, Any] | None = None,
    valid_size: float | None = None,
) -> tuple[
    dict[str, models.CatBoostTLearner],
    pd.DataFrame,
    pd.DataFrame,
    dict[str, dict[str, object]],
]:
    train_part, valid_part = models.train_valid_split(df_train, valid_size=valid_size)
    learners = models.fit_cash_card_learners(train_part, model_params=model_params)

    train_scores = models.score_cash_card(train_part, learners)
    valid_scores = models.score_cash_card(valid_part, learners)

    qini_results = {
        "cash": learners["cash"].evaluate(valid_part),
        "card": learners["card"].evaluate(valid_part),
    }
    return learners, train_scores, valid_scores, qini_results


def score_test(
    df_test: pd.DataFrame,
    learners: dict[str, models.CatBoostTLearner],
) -> pd.DataFrame:
    return models.score_cash_card(df_test, learners)


def run_pipeline(
    model_params: dict[str, Any] | None = None,
    valid_size: float | None = None,
    sample_size: int | None = None,
    save_outputs: bool = True,
    print_reports: bool = True,
) -> PipelineArtifacts:
    """
    Run the modeling workflow end to end.

    sample_size is intended for quick smoke tests. Leave it as None for full data.
    """
    C.ensure_dirs()

    df_train = io.load_train()
    df_test = io.load_test()

    if sample_size is not None:
        df_train = df_train.head(sample_size).copy()
        df_test = df_test.head(max(1, min(sample_size, len(df_test)))).copy()

    health_reports = run_health_checks(df_train, df_test, print_report=print_reports)
    learners, train_scores, valid_scores, qini_results = train_and_evaluate(
        df_train,
        model_params=model_params,
        valid_size=valid_size,
    )
    test_scores = score_test(df_test, learners)

    output_paths: dict[str, Path] = {}
    if save_outputs:
        output_paths["train_scores"] = _write_report_table(
            train_scores,
            C.TABLE_DIR / "train_uplift_scores.csv",
        )
        output_paths["valid_scores"] = _write_report_table(
            valid_scores,
            C.TABLE_DIR / "valid_uplift_scores.csv",
        )
        output_paths["test_scores"] = _write_report_table(
            test_scores,
            C.TABLE_DIR / "test_uplift_scores.csv",
        )
        output_paths.update(_save_qini_outputs(qini_results, prefix="valid"))
        output_paths.update(_save_models(learners))

    return PipelineArtifacts(
        train_scores=train_scores,
        valid_scores=valid_scores,
        test_scores=test_scores,
        qini_results=qini_results,
        health_reports=health_reports,
        learners=learners,
        output_paths=output_paths,
    )


def print_qini_summary(qini_results: dict[str, dict[str, object]]) -> None:
    for product, result in qini_results.items():
        print(f"\n[{product.upper()}] Qini summary")
        print(pd.DataFrame([result["summary"]]).to_string(index=False))
        print("\nTop-K uplift")
        print(result["topk"].to_string(index=False))


if __name__ == "__main__":
    artifacts = run_pipeline(
        model_params={"iterations": 50, "depth": 4},
        sample_size=5000,
        save_outputs=True,
        print_reports=True,
    )
    print_qini_summary(artifacts.qini_results)
    print("\nSaved outputs:")
    for key, path in artifacts.output_paths.items():
        print(f"- {key}: {path}")
