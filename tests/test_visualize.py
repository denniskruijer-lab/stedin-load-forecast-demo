import pandas as pd

from stedin_load_forecast.visualize import plot_forecast_comparison


def test_plot_forecast_comparison_writes_a_nonempty_image(tmp_path):
    index = pd.date_range("2026-07-01", periods=24, freq="15min", tz="UTC")
    y_true = pd.Series(range(24), index=index, dtype=float)
    model_pred = pd.Series(range(24), index=index, dtype=float) + 1
    naive_pred = pd.Series(range(24), index=index, dtype=float) - 1

    output_path = tmp_path / "plots" / "forecast.png"
    result_path = plot_forecast_comparison(y_true, model_pred, naive_pred, output_path)

    assert result_path == output_path
    assert output_path.exists()
    assert output_path.stat().st_size > 0
