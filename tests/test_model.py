import numpy as np
import pandas as pd

from stedin_load_forecast.features import build_features
from stedin_load_forecast.model import (
    evaluate,
    sarima_forecast,
    seasonal_naive_predict,
    split_last_n,
    time_train_test_split,
    train_regressor,
    train_sarima,
)


def test_time_train_test_split_is_chronological_not_shuffled():
    df = pd.DataFrame({"x": range(10)})

    train, test = time_train_test_split(df, test_fraction=0.3)

    assert len(train) == 7
    assert len(test) == 3
    assert train["x"].tolist() == [0, 1, 2, 3, 4, 5, 6]
    assert test["x"].tolist() == [7, 8, 9]


def test_split_last_n_holds_out_an_absolute_row_count():
    df = pd.DataFrame({"x": range(10)})

    train, test = split_last_n(df, n=3)

    assert len(train) == 7
    assert len(test) == 3
    assert test["x"].tolist() == [7, 8, 9]


def test_evaluate_computes_mae_rmse_mape():
    y_true = pd.Series([100.0, 200.0, 100.0])
    y_pred = pd.Series([110.0, 190.0, 100.0])

    metrics = evaluate(y_true, y_pred)

    assert metrics["mae"] == np.mean([10, 10, 0])
    assert metrics["rmse"] == np.sqrt(np.mean([100, 100, 0]))
    assert round(metrics["mape"], 4) == round(np.mean([10 / 100, 10 / 200, 0]) * 100, 4)


def _synthetic_load_series(n_days: int = 12, seed: int = 42) -> pd.DataFrame:
    """Deterministic daily-seasonal load series with small noise."""
    rng = np.random.default_rng(seed)
    periods = n_days * 96  # 96 steps/day at 15-minute resolution
    index = pd.date_range("2026-06-01", periods=periods, freq="15min", tz="UTC")

    hour_of_day = index.hour + index.minute / 60
    daily_pattern = 100 + 30 * np.sin((hour_of_day / 24) * 2 * np.pi - np.pi / 2)
    noise = rng.normal(scale=2.0, size=periods)

    return pd.DataFrame({"load_mw": daily_pattern + noise}, index=index)


def test_regressor_beats_seasonal_naive_baseline_on_synthetic_seasonal_data():
    raw = _synthetic_load_series()
    featured = build_features(raw, target_col="load_mw", lags=(1, 4, 96))
    train, test = time_train_test_split(featured, test_fraction=0.2)

    feature_cols = ["hour", "day_of_week", "is_weekend", "lag_1", "lag_4", "lag_96"]
    model = train_regressor(train, feature_cols=feature_cols, target_col="load_mw")
    model_pred = model.predict(test[feature_cols])
    model_metrics = evaluate(test["load_mw"], model_pred)

    naive_pred = seasonal_naive_predict(test, season_lag_col="lag_96")
    naive_metrics = evaluate(test["load_mw"], naive_pred)

    assert model_metrics["mae"] < naive_metrics["mae"]
    assert model_metrics["mae"] < 5.0  # sanity bound given noise scale=2.0


def _synthetic_hourly_load_series(n_days: int = 10, seed: int = 42) -> pd.Series:
    """Deterministic daily-seasonal hourly load series with small noise.

    Kept small (n_days=10 -> 240 points) so SARIMA fitting in tests stays
    fast — see model.py's module docstring for why SARIMA in this
    codebase always runs on hourly, not 15-minute, data in the first
    place (751s vs. 3.9s fit time, measured empirically).
    """
    rng = np.random.default_rng(seed)
    periods = n_days * 24
    index = pd.date_range("2026-06-01", periods=periods, freq="h", tz="UTC")

    daily_pattern = 100 + 30 * np.sin((index.hour / 24) * 2 * np.pi - np.pi / 2)
    noise = rng.normal(scale=2.0, size=periods)

    return pd.Series(daily_pattern + noise, index=index, name="load_mw")


def test_train_sarima_fits_and_sarima_forecast_matches_requested_index():
    series = _synthetic_hourly_load_series()
    train, test = series.iloc[:-24], series.iloc[-24:]

    fitted = train_sarima(train, order=(1, 0, 0), seasonal_order=(1, 0, 0, 24))
    forecast = sarima_forecast(fitted, steps=len(test), index=test.index)

    assert len(forecast) == 24
    assert forecast.index.equals(test.index)


def test_sarima_beats_seasonal_naive_baseline_on_synthetic_seasonal_data():
    series = _synthetic_hourly_load_series(n_days=15).to_frame("load_mw")
    featured = build_features(series, target_col="load_mw", lags=(24,))
    train, test = time_train_test_split(featured, test_fraction=0.2)

    fitted = train_sarima(train["load_mw"], order=(1, 0, 0), seasonal_order=(1, 0, 0, 24))
    sarima_pred = sarima_forecast(fitted, steps=len(test), index=test.index)
    sarima_metrics = evaluate(test["load_mw"], sarima_pred)

    naive_pred = seasonal_naive_predict(test, season_lag_col="lag_24")
    naive_metrics = evaluate(test["load_mw"], naive_pred)

    assert sarima_metrics["mae"] < naive_metrics["mae"]
