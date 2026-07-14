"""Feature engineering for the load forecasting model.

Electricity load is dominated by two things: what time of day/week it is,
and what the load was doing recently. So we build exactly two families of
features:

- **Calendar features** (`add_calendar_features`): hour, day-of-week,
  weekend flag. Cheap, interpretable, and they capture almost all of the
  daily/weekly seasonality without needing something heavier like Fourier
  terms — appropriate for a demo of this size.
- **Lag features** (`add_lag_features`): the target's own recent history.
  A tree model can't "see" a trend on its own the way an ARIMA model can;
  giving it explicit lagged values lets it learn that relationship instead.

`CALENDAR_FEATURE_COLUMNS` and `lag_feature_columns()` exist so that the
*names* of the engineered columns are defined in exactly one place. The
model module (which needs to know which columns to feed the regressor)
imports these instead of hardcoding a parallel list — if the lag set ever
changes here, every consumer picks it up automatically instead of silently
drifting out of sync.
"""

import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Expressed in rows at the source's ~15-minute resolution: 15 minutes ago,
# 1 hour ago, 1 day ago, 1 week ago. Chosen to cover the three timescales
# that matter for load: short-term momentum (lag_1, lag_4), the previous
# day's shape (lag_96, since load repeats a similar daily curve), and the
# previous week's shape at the same weekday (lag_672, since weekday vs.
# weekend load differs a lot and lag_96 alone can't tell them apart).
DEFAULT_LAGS = (1, 4, 96, 672)

CALENDAR_FEATURE_COLUMNS = ["hour", "day_of_week", "is_weekend"]


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive hour-of-day, day-of-week, and a weekend flag from the index.

    Args:
        df: DataFrame with a DatetimeIndex.

    Returns:
        A copy of df with three extra columns: "hour" (0-23),
        "day_of_week" (0=Monday .. 6=Sunday), and "is_weekend" (0/1).
    """
    out = df.copy()
    out["hour"] = out.index.hour
    out["day_of_week"] = out.index.dayofweek
    out["is_weekend"] = out["day_of_week"].isin([5, 6]).astype(int)
    return out


def lag_feature_columns(lags: tuple[int, ...] = DEFAULT_LAGS) -> list[str]:
    """Column names add_lag_features() will produce for the given lags.

    Exists so callers (e.g. the model module, which needs the exact list
    of feature columns to train on) never have to hardcode "lag_{n}"
    strings themselves and risk them falling out of sync with the lags
    actually used to build the training data.
    """
    return [f"lag_{lag}" for lag in lags]


def feature_columns(lags: tuple[int, ...] = DEFAULT_LAGS) -> list[str]:
    """Full list of engineered feature column names (calendar + lags).

    This is the single source of truth for "what columns does the model
    train on" — build_features() produces exactly these columns (plus the
    original data), so `feature_columns(lags)` always matches its output.
    """
    return CALENDAR_FEATURE_COLUMNS + lag_feature_columns(lags)


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Add lagged copies of target_col as new columns.

    The first `max(lags)` rows won't have a full lag history (there's no
    "672 steps ago" for the 672nd row) and are dropped rather than
    imputed — training on a made-up value would teach the model a
    relationship that doesn't exist in the real data.

    Args:
        df: Input DataFrame, chronologically ordered.
        target_col: Column to lag.
        lags: How many rows back to shift, e.g. (1, 96) for "1 step ago"
            and "96 steps ago".

    Returns:
        A copy of df with one "lag_{n}" column per entry in `lags`, and
        rows without complete lag history removed.
    """
    out = df.copy()
    for lag in lags:
        out[f"lag_{lag}"] = out[target_col].shift(lag)
    return out.dropna()


def build_features(
    df: pd.DataFrame,
    target_col: str = "load_mw",
    lags: tuple[int, ...] = DEFAULT_LAGS,
) -> pd.DataFrame:
    """Build the full feature set (calendar + lags) for training or inference.

    This is the one function both the training pipeline and any future
    inference/serving path should call — computing features in a single
    place, rather than re-implementing the logic at each call site, is
    what keeps train-time and predict-time features from silently drifting
    apart.

    Args:
        df: Raw input DataFrame with a DatetimeIndex and target_col.
        target_col: Column to build lag features from.
        lags: Passed through to add_lag_features().

    Returns:
        df with calendar and lag columns added, and incomplete-history
        rows dropped. Column names match feature_columns(lags).
    """
    logger.info("Building features from %d rows (lags=%s)", len(df), lags)
    out = add_calendar_features(df)
    out = add_lag_features(out, target_col=target_col, lags=lags)
    logger.info("%d rows remain after dropping incomplete lag history", len(out))
    return out
