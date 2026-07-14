from unittest.mock import patch

import numpy as np
import pandas as pd

from stedin_load_forecast.pipeline import run_pipeline


def _synthetic_load_series(n_days: int = 12, seed: int = 42) -> pd.DataFrame:
    """Deterministic daily-seasonal load series with small noise."""
    rng = np.random.default_rng(seed)
    periods = n_days * 96
    index = pd.date_range("2026-06-01", periods=periods, freq="15min", tz="UTC")

    hour_of_day = index.hour + index.minute / 60
    daily_pattern = 100 + 30 * np.sin((hour_of_day / 24) * 2 * np.pi - np.pi / 2)
    noise = rng.normal(scale=2.0, size=periods)

    return pd.DataFrame({"load_mw": daily_pattern + noise}, index=index)


def test_run_pipeline_end_to_end_produces_metrics_and_plot(tmp_path):
    with patch(
        "stedin_load_forecast.pipeline.fetch_load", return_value=_synthetic_load_series()
    ) as mock_fetch:
        result = run_pipeline(
            start="2026-06-01",
            end="2026-06-12",
            country="nl",
            output_dir=tmp_path,
        )

    mock_fetch.assert_called_once_with(start="2026-06-01", end="2026-06-12", country="nl")

    for metrics in (
        result.model_metrics,
        result.naive_metrics,
        result.sarima_metrics,
        result.sarima_naive_metrics,
    ):
        assert set(metrics) == {"mae", "rmse", "mape"}

    assert len(result.y_true) == len(result.model_pred) == len(result.naive_pred)
    assert result.plot_path.exists()
    assert result.plot_path.stat().st_size > 0
    assert result.sarima_plot_path.exists()
    assert result.sarima_plot_path.stat().st_size > 0
    assert result.sarima_plot_path != result.plot_path
