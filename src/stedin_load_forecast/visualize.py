"""Plotting helpers for the two charts this demo produces.

Uses matplotlib's non-interactive "Agg" backend, set once at import time,
because this code runs headless in CI and in the CLI script — the default
backend tries to open an interactive window, which fails (or silently
does nothing useful) on a machine with no display.

Deliberately has no imports from the rest of this package (energy_charts,
stedin_open_data, ...): it only knows how to turn already-shaped
DataFrames/Series into images. That keeps it trivially reusable and
testable in isolation — a caller that wants a licensing attribution
printed on the chart (both data sources here are CC BY 4.0 and require
one) passes the text in via the `attribution` argument rather than this
module reaching into another module to fetch it itself.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def _add_attribution(fig: plt.Figure, attribution: str | None) -> None:
    """Small caption in the bottom-right corner, if the caller supplied one."""
    if attribution:
        fig.text(0.99, 0.01, attribution, ha="right", va="bottom", fontsize=7, color="gray")


def plot_forecast_comparison(
    y_true: pd.Series,
    model_pred: pd.Series,
    naive_pred: pd.Series,
    output_path: Path,
    attribution: str | None = None,
    model_label: str = "Model",
    title: str = "NL electricity load: forecast vs actual (test period)",
) -> Path:
    """Line chart comparing actual load to a model's and the naive baseline's predictions.

    All three lines share one axis (rather than, say, plotting error over
    time) because the point of this chart is to make the model's edge
    visually obvious at a glance: the naive baseline visibly lags behind
    sharp changes, the model tracks them. Generic enough to plot any
    single model against the baseline (the pipeline uses it once for
    GradientBoosting, once for SARIMA — model_label/title distinguish
    the two rather than this function assuming which model it's plotting).

    Args:
        y_true: Actual load values over the test period.
        model_pred: The trained model's predictions, same index as y_true.
        naive_pred: The naive baseline's predictions, same index as y_true.
        output_path: Where to save the PNG. Parent directories are
            created if missing.
        attribution: Optional data-source credit line, e.g.
            energy_charts.ATTRIBUTION, printed small in the corner.
        model_label: Legend label for model_pred, e.g. "Model (SARIMA)".
        title: Chart title.

    Returns:
        output_path, for convenient chaining.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true.index, y_true.values, label="Actual", linewidth=1.5)
    ax.plot(y_true.index, model_pred.values, label=model_label, linewidth=1)
    ax.plot(
        y_true.index,
        naive_pred.values,
        label="Naive baseline (yesterday)",
        linewidth=1,
        linestyle="--",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (MW)")
    ax.set_title(title)
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    _add_attribution(fig, attribution)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved forecast comparison plot to %s", output_path)
    return output_path


def plot_top_woonplaatsen(
    df: pd.DataFrame,
    output_path: Path,
    attribution: str | None = None,
) -> Path:
    """Horizontal bar chart of average annual consumption (SJV) by woonplaats.

    Horizontal (barh), not vertical bars, because Dutch place names
    ("'s-Gravenhage") don't fit as rotated x-axis labels without either
    truncating them or making the chart very tall — a horizontal layout
    keeps every label fully readable at a normal chart width.

    Args:
        df: Must have "WOONPLAATS" and "avg_sjv_kwh" columns, e.g. the
            output of stedin_open_data.top_woonplaatsen_by_connections().
        output_path: Where to save the PNG.
        attribution: Optional data-source credit line, e.g.
            stedin_open_data.ATTRIBUTION.

    Returns:
        output_path, for convenient chaining.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ascending order so barh (which draws bottom-to-top) ends up showing
    # the largest bar at the top of the chart, matching reading order.
    ordered = df.sort_values("avg_sjv_kwh", ascending=True)

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(ordered["WOONPLAATS"], ordered["avg_sjv_kwh"])
    ax.set_xlabel("Average standard annual consumption (kWh)")
    ax.set_title("Stedin open data: avg. electricity consumption, largest woonplaatsen")
    fig.tight_layout()
    _add_attribution(fig, attribution)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved top-woonplaatsen chart to %s", output_path)
    return output_path
