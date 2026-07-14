"""End-to-end pipeline: fetch load data, engineer features, train and
evaluate the forecast model against a naive baseline, and plot the result.
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from stedin_load_forecast.energy_charts import fetch_load
from stedin_load_forecast.features import build_features
from stedin_load_forecast.model import (
    evaluate,
    seasonal_naive_predict,
    time_train_test_split,
    train_regressor,
)
from stedin_load_forecast.visualize import plot_forecast_comparison

logger = logging.getLogger(__name__)

FEATURE_COLUMNS = ["hour", "day_of_week", "is_weekend", "lag_1", "lag_4", "lag_96", "lag_672"]
TARGET_COLUMN = "load_mw"


@dataclass
class PipelineResult:
    model_metrics: dict[str, float]
    naive_metrics: dict[str, float]
    y_true: pd.Series
    model_pred: pd.Series
    naive_pred: pd.Series
    plot_path: Path


def run_pipeline(
    start: date | str,
    end: date | str,
    country: str = "nl",
    test_fraction: float = 0.2,
    output_dir: Path | str = "outputs",
) -> PipelineResult:
    """Run the full pipeline and write a forecast-comparison plot to output_dir."""
    raw = fetch_load(start=start, end=end, country=country)
    featured = build_features(raw, target_col=TARGET_COLUMN)
    train, test = time_train_test_split(featured, test_fraction=test_fraction)

    model = train_regressor(train, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)
    model_pred = pd.Series(model.predict(test[FEATURE_COLUMNS]), index=test.index)
    naive_pred = seasonal_naive_predict(test, season_lag_col="lag_96")

    model_metrics = evaluate(test[TARGET_COLUMN], model_pred)
    naive_metrics = evaluate(test[TARGET_COLUMN], naive_pred)
    logger.info("Model metrics: %s", model_metrics)
    logger.info("Naive baseline metrics: %s", naive_metrics)

    plot_path = plot_forecast_comparison(
        y_true=test[TARGET_COLUMN],
        model_pred=model_pred,
        naive_pred=naive_pred,
        output_path=Path(output_dir) / "forecast_comparison.png",
    )

    return PipelineResult(
        model_metrics=model_metrics,
        naive_metrics=naive_metrics,
        y_true=test[TARGET_COLUMN],
        model_pred=model_pred,
        naive_pred=naive_pred,
        plot_path=plot_path,
    )
