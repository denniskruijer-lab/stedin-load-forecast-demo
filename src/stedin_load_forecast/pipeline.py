"""End-to-end pipeline: fetch load data, train and evaluate two forecast
models (GradientBoosting and SARIMA) against a naive baseline, and plot
the results.

This module is deliberately the *only* place that wires the other modules
together in this order. Each of energy_charts, features, model, and
visualize is independently usable and independently tested; pipeline.py's
job is just orchestration, so that "how do I run this end to end" has one
obvious answer (run_pipeline()) instead of every caller re-inventing the
fetch -> features -> train -> evaluate -> plot sequence themselves.

**Why two separate comparisons, not one three-way comparison?** The
GradientBoosting comparison runs at the data's native ~15-minute
resolution; the SARIMA comparison runs on the same period resampled to
hourly. That's not an oversight — see model.py's module docstring for the
empirical timing finding (751s vs. 3.9s) behind why SARIMA specifically
needs the coarser resolution to be tractable here. Each comparison gets
its own naive baseline at its own resolution, since "the same time
yesterday" means a different lag (96 steps vs. 24 steps) at each.
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
    sarima_forecast,
    seasonal_naive_predict,
    split_last_n,
    time_train_test_split,
    train_regressor,
    train_sarima,
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

# SARIMA runs on hourly-resampled data (see module docstring for why), so
# "yesterday" there is 24 steps back, not 96.
SARIMA_LAGS = (24,)
SARIMA_NAIVE_SEASON_LAG_COLUMN = "lag_24"
# See model.py's module docstring for the empirical reasoning behind both
# of these: the "obvious" (1,1,1)x(1,1,1,24) diverges over a multi-day
# horizon, and a fixed 48-hour evaluation window (not the same
# test_fraction the regressor uses) reflects how a model like this is
# actually meant to be used.
SARIMA_ORDER = (0, 0, 0)
SARIMA_SEASONAL_ORDER = (1, 1, 0, 24)
SARIMA_TEST_HOURS = 48


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

    sarima_metrics: dict[str, float]
    sarima_naive_metrics: dict[str, float]
    sarima_plot_path: Path


def run_pipeline(
    start: date | str,
    end: date | str,
    country: str = "nl",
    test_fraction: float = 0.2,
    output_dir: Path | str = "outputs",
) -> PipelineResult:
    """Fetch load data, train both models, evaluate them, and plot the results.

    This is the single entry point for "give me a working forecast" —
    scripts/run_pipeline.py (the CLI) and the test suite (with fetch_load
    mocked) both call this same function, so there's exactly one code path
    to trust rather than a CLI-only path that might behave differently
    from what's actually tested.

    Args:
        start: Start date for the historical load data to fetch.
        end: End date for the historical load data to fetch.
        country: energy-charts.info country code, e.g. "nl".
        test_fraction: Fraction of rows held out for each chronological
            test split (see model.time_train_test_split for why it's
            chronological, not random). Applied independently to the
            15-minute and hourly series.
        output_dir: Where to write the forecast-comparison plots.

    Returns:
        A PipelineResult with both comparisons' metrics, the GradientBoosting
        comparison's test-period predictions (for further analysis or
        re-plotting), and both plot paths.
    """
    raw = fetch_load(start=start, end=end, country=country)

    model_metrics, naive_metrics, y_true, model_pred, naive_pred, plot_path = (
        _run_gradient_boosting_comparison(raw, test_fraction, output_dir)
    )
    sarima_metrics, sarima_naive_metrics, sarima_plot_path = _run_sarima_comparison(
        raw, test_fraction, output_dir
    )

    return PipelineResult(
        model_metrics=model_metrics,
        naive_metrics=naive_metrics,
        y_true=y_true,
        model_pred=model_pred,
        naive_pred=naive_pred,
        plot_path=plot_path,
        sarima_metrics=sarima_metrics,
        sarima_naive_metrics=sarima_naive_metrics,
        sarima_plot_path=sarima_plot_path,
    )


def _run_gradient_boosting_comparison(
    raw: pd.DataFrame, test_fraction: float, output_dir: Path | str
) -> tuple[dict, dict, pd.Series, pd.Series, pd.Series, Path]:
    """The original 15-minute-resolution GradientBoosting vs. naive comparison."""
    featured = build_features(raw, target_col=TARGET_COLUMN)
    train, test = time_train_test_split(featured, test_fraction=test_fraction)

    model = train_regressor(train, feature_cols=FEATURE_COLUMNS, target_col=TARGET_COLUMN)
    model_pred = pd.Series(model.predict(test[FEATURE_COLUMNS]), index=test.index)
    naive_pred = seasonal_naive_predict(test, season_lag_col=NAIVE_SEASON_LAG_COLUMN)

    model_metrics = evaluate(test[TARGET_COLUMN], model_pred)
    naive_metrics = evaluate(test[TARGET_COLUMN], naive_pred)
    logger.info("GradientBoosting metrics: %s", model_metrics)
    logger.info("Naive baseline metrics (15-min): %s", naive_metrics)

    plot_path = plot_forecast_comparison(
        y_true=test[TARGET_COLUMN],
        model_pred=model_pred,
        naive_pred=naive_pred,
        output_path=Path(output_dir) / "forecast_comparison.png",
        attribution=ATTRIBUTION,
        model_label="Model (GradientBoosting)",
        title="NL electricity load: forecast vs actual (test period)",
    )

    return model_metrics, naive_metrics, test[TARGET_COLUMN], model_pred, naive_pred, plot_path


def _run_sarima_comparison(
    raw: pd.DataFrame, test_fraction: float, output_dir: Path | str
) -> tuple[dict, dict, Path]:
    """The SARIMA vs. naive comparison, at hourly resolution — see module docstring.

    test_fraction is accepted but intentionally unused for the split
    itself — SARIMA_TEST_HOURS (a fixed horizon) is used instead, for the
    reasons documented on that constant and in model.py's module
    docstring. It's still accepted here so run_pipeline's single
    test_fraction argument doesn't silently do nothing for half the
    pipeline without an obvious reason why, visible right at the call site.
    """
    hourly_raw = raw[[TARGET_COLUMN]].resample("1h").mean()
    hourly_featured = build_features(hourly_raw, target_col=TARGET_COLUMN, lags=SARIMA_LAGS)
    hourly_train, hourly_test = split_last_n(hourly_featured, SARIMA_TEST_HOURS)

    fitted = train_sarima(
        hourly_train[TARGET_COLUMN], order=SARIMA_ORDER, seasonal_order=SARIMA_SEASONAL_ORDER
    )
    sarima_pred = sarima_forecast(fitted, steps=len(hourly_test), index=hourly_test.index)
    naive_pred = seasonal_naive_predict(
        hourly_test, season_lag_col=SARIMA_NAIVE_SEASON_LAG_COLUMN
    )

    sarima_metrics = evaluate(hourly_test[TARGET_COLUMN], sarima_pred)
    naive_metrics = evaluate(hourly_test[TARGET_COLUMN], naive_pred)
    logger.info("SARIMA metrics (hourly): %s", sarima_metrics)
    logger.info("Naive baseline metrics (hourly): %s", naive_metrics)

    plot_path = plot_forecast_comparison(
        y_true=hourly_test[TARGET_COLUMN],
        model_pred=sarima_pred,
        naive_pred=naive_pred,
        output_path=Path(output_dir) / "sarima_comparison.png",
        attribution=ATTRIBUTION,
        model_label="Model (SARIMA)",
        title="NL electricity load: SARIMA forecast vs actual (hourly, test period)",
    )

    return sarima_metrics, naive_metrics, plot_path
