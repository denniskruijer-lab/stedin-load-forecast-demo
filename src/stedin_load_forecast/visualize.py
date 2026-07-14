"""Plotting helpers. Uses the non-interactive Agg backend so this works
headless (CI, scripts) without a display.
"""

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

logger = logging.getLogger(__name__)


def plot_forecast_comparison(
    y_true: pd.Series,
    model_pred: pd.Series,
    naive_pred: pd.Series,
    output_path: Path,
) -> Path:
    """Plot actual vs. model vs. naive-baseline predictions over the test period."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true.index, y_true.values, label="Actual", linewidth=1.5)
    ax.plot(y_true.index, model_pred.values, label="Model (GradientBoosting)", linewidth=1)
    ax.plot(
        y_true.index,
        naive_pred.values,
        label="Naive baseline (yesterday)",
        linewidth=1,
        linestyle="--",
    )
    ax.set_xlabel("Time")
    ax.set_ylabel("Load (MW)")
    ax.set_title("NL electricity load: forecast vs actual (test period)")
    ax.legend()
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    logger.info("Saved forecast comparison plot to %s", output_path)
    return output_path
