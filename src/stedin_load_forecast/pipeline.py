"""End-to-end pipeline: fetch load data, engineer features, train and
evaluate the forecast model against a naive baseline, and plot the result.

This module is deliberately the *only* place that wires the other modules
together in this order. Each of energy_charts, features, model, and
visualize is independently usable and independently tested; pipeline.py's
job is just orchestration, so that "how do I run this end to end" has one
obvious answer (run_pipeline()) instead of every caller re-inventing the
fetch -> features -> train -> evaluate -> plot sequence themselves.
"""

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import pandas as pd

from stedin_load_forecast.energy_charts import ATTRIBUTION, fetch_load
from stedin_load_forecast.features import DEFAULT_LAGS, build_features, feature_columns
from stedin_load_forecast.model import (
    evaluate,
    seasonal_naive_predict,
    time_train_test_split,
    train_regressor,
)
from stedin_load_forecast.visualize import plot_forecast_comparison

logger = logging.getLogger(__name__)

# Derived from features.feature_columns() rather than hardcoded here, so
# that if the lag set in features.py ever changes, the model always trains
# on exactly the columns build_features() actually produces — the two
# can't silently drift apart.
FEATURE_COLUMNS = feature_columns(DEFAULT_LAGS)
TARGET_COLUMN = "load_mw"

# season_lag_col for the naive baseline: "lag_96" is 96 rows back at the
# source's ~15-minute resolution, i.e. the same time yesterday. This must
# stay one of the lags produced by DEFAULT_LAGS/FEATURE_COLUMNS above.
NAIVE_SEASON_LAG_COLUMN = "lag_96"


@dataclass
class PipelineResult:
    """Everything a caller needs after a pipeline run, without having to
    re-derive it from side effects (log lines, files on disk) — a plain
    dict would work too, but a dataclass makes the available fields
    discoverable (IDE autocomplete, type checking) instead of tribal
    knowledge.
    """

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
    """Fetch load data, train the model, evaluate it, and plot the result.

    This is the single entry point for "give me a working forecast" —
    scripts/run_pipeline.py (the CLI) and the test suite (with fetch_load
    mocked) both call this same function, so there's exactly one code path
    to trust rather than a CLI-only path that might behave differently
    from what's actually tested.

    Args:
        start: Start date for the historical load data to fetch.
        end: End date for the historical load data to fetch.
        country: energy-charts.info country code, e.g. "nl".
        test_fraction: Fraction of rows held out for the chronological
            test split (see model.time_train_test_split for why it's
            chronological, not random).
        output_dir: Where to write the forecast-comparison plot.

    Returns:
        A PipelineResult with both models' metrics, the test-period
        predictions (for further analysis or re-plotting), and the path
        to the saved plot.
    """
    raw = fetch_load(start=start, end=end, country=country)
    featured = build_features(raw, target_col=TARGET_COLUMN)
    train, test = time_train_test_split(featured, test_fraction=test_fraction)

    model = train_regressor(train, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)
    model_pred = pd.Series(model.predict(test[FEATURE_COLUMNS]), index=test.index)
    naive_pred = seasonal_naive_predict(test, season_lag_col=NAIVE_SEASON_LAG_COLUMN)

    model_metrics = evaluate(test[TARGET_COLUMN], model_pred)
    naive_metrics = evaluate(test[TARGET_COLUMN], naive_pred)
    logger.info("Model metrics: %s", model_metrics)
    logger.info("Naive baseline metrics: %s", naive_metrics)

    plot_path = plot_forecast_comparison(
        y_true=test[TARGET_COLUMN],
        model_pred=model_pred,
        naive_pred=naive_pred,
        output_path=Path(output_dir) / "forecast_comparison.png",
        attribution=ATTRIBUTION,
    )

    return PipelineResult(
        model_metrics=model_metrics,
        naive_metrics=naive_metrics,
        y_true=test[TARGET_COLUMN],
        model_pred=model_pred,
        naive_pred=naive_pred,
        plot_path=plot_path,
    )
