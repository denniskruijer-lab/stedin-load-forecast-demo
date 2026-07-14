"""Feature engineering for the load forecasting model.

Two families of features, both standard for short-term load forecasting:
calendar features (load has strong daily/weekly seasonality) and lag
features (recent and same-time-last-period values are strong predictors).
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Default lags, expressed in number of rows at the source's ~15-minute
# resolution: 1 step back, 1 hour back, 1 day back, 1 week back.
DEFAULT_LAGS = (1, 4, 96, 672)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour-of-day, day-of-week and is_weekend columns from the index."""
    out = df.copy()
    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add lagged copies of target_col. Rows without full lag history are dropped."""
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out.dropna()


def build_features(
    df: pd.DataFrame,
    target_col: str = "load_mw",
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Build the full feature set (calendar + lags) for training/inference."""
    logger.info("Building features from %d rows (lags=%s)", len(df), lags)
    out = add_calendar_features(df)
    out = add_lag_features(out, target_col=target_col, lags=lags)
    logger.info("%d rows remain after dropping incomplete lag history", len(out))
    return out
