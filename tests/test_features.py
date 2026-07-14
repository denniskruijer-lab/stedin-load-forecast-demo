import pandas as pd

from stedin_load_forecast.features import add_calendar_features, add_lag_features, build_features


def _sample_series(n=10, freq="15min"):
    index = pd.date_range("2026-07-06T00:00:00Z", periods=n, freq=freq)  # a Monday
    return pd.DataFrame({"load_mw": range(n)}, index=index)


def test_add_calendar_features_derives_hour_day_of_week_and_weekend():
    df = _sample_series(n=4)

    out = add_calendar_features(df)

    assert list(out["hour"]) == [0, 0, 0, 0]
    assert list(out["day_of_week"]) == [0, 0, 0, 0]  # Monday
    assert list(out["is_weekend"]) == [0, 0, 0, 0]


def test_add_calendar_features_flags_saturday_as_weekend():
    index = pd.date_range("2026-07-11T00:00:00Z", periods=2, freq="15min")  # a Saturday
    df = pd.DataFrame({"load_mw": [1, 2]}, index=index)

    out = add_calendar_features(df)

    assert list(out["is_weekend"]) == [1, 1]


def test_add_lag_features_shifts_target_and_drops_incomplete_rows():
    df = _sample_series(n=10)

    out = add_lag_features(df, target_col="load_mw", lags=(1, 2))

    assert len(out) == 8  # first 2 rows dropped (no lag_2 history)
    assert out["lag_1"].iloc[0] == out["load_mw"].iloc[0] - 1
    assert out["lag_2"].iloc[0] == out["load_mw"].iloc[0] - 2


def test_build_features_combines_calendar_and_lag_features():
    df = _sample_series(n=10)

    out = build_features(df, target_col="load_mw", lags=(1, 2))

    for col in ("hour", "day_of_week", "is_weekend", "lag_1", "lag_2"):
        assert col in out.columns
    assert len(out) == 8
