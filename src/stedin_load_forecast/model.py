"""Forecasting model: a seasonal-naive baseline and a gradient-boosted
regressor trained on calendar + lag features, plus shared evaluation.

**Why a baseline at all?** Any model complex enough to need a training
step should have to prove it's worth that complexity. seasonal_naive_predict
is the cheapest plausible forecaster (no fitting, no library, just "reuse
yesterday's value") — if train_regressor() can't beat it by a wide margin,
the extra machinery isn't earning its keep. It also gives a stakeholder
conversation a concrete anchor: "8.7% MAPE" means little on its own, but
"a third of the naive baseline's error" is immediately legible.

**Why GradientBoostingRegressor?** It handles the non-linear interactions
between calendar features and lag features (e.g. "yesterday's value
matters more on weekdays than weekends") without hand-built feature
crosses, needs no feature scaling (unlike linear models or neural nets),
and is a well-understood, production-standard choice that's easy to
explain to a non-ML stakeholder — all of which matter more here than
squeezing out the last percent of accuracy with a heavier time-series-
specific model (ARIMA, Prophet, a neural forecaster). See the README's
"Approach" section for the fuller trade-off discussion.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


def time_train_test_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a chronologically-ordered DataFrame into train/test by time, not at random.

    Random shuffling would let the model train on rows adjacent in time to
    its test rows — since lag features make neighbouring rows highly
    correlated, that leaks near-future information into training and
    makes the reported accuracy look better than it would in real
    deployment, where you only ever have the past to predict the future.

    Args:
        df: Chronologically ordered DataFrame (oldest row first).
        test_fraction: Fraction of rows (from the end) held out as the
            test set.

    Returns:
        (train, test) — the earlier (1 - test_fraction) of rows, then the
        later test_fraction of rows.
    """
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def seasonal_naive_predict(df: pd.DataFrame, season_lag_col: str = "lag_96") -> pd.Series:
    """Baseline forecast: predict the same value as the same time one season ago.

    "Season" here means one full daily cycle, not literally winter/summer.
    Default season_lag_col="lag_96" assumes the ~15-minute-resolution data
    this pipeline works with, where 96 steps = 24 hours, i.e. lag_96 is
    "the load at this exact time yesterday" — a strong baseline given how
    repetitive daily load curves are.

    Args:
        df: A features DataFrame that already has season_lag_col (i.e.
            produced by features.build_features with a matching lag).
        season_lag_col: Which lag column to use as the prediction.

    Returns:
        The lag column itself, treated as the forecast.
    """
    return df[season_lag_col]


def train_regressor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    """Fit a GradientBoostingRegressor on the given features and target.

    Args:
        train_df: Training data — must contain all of feature_cols and
            target_col (this is exactly what features.build_features()
            followed by time_train_test_split() produces).
        feature_cols: Column names to train on. In this pipeline these
            come from features.feature_columns(), so they always match
            what build_features() actually produced.
        target_col: Column to predict.
        random_state: Fixed for reproducibility — same data in, same
            model out, every run. Scikit-learn's default (None) would use
            unseeded randomness in the boosting process, making two runs
            on identical data produce silently different models.

    Returns:
        The fitted model. Call .predict(df[feature_cols]) to forecast.
    """
    logger.info(
        "Training GradientBoostingRegressor on %d rows, %d features",
        len(train_df),
        len(feature_cols),
    )
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Compute MAE, RMSE, and MAPE for a set of predictions.

    All three are reported together because each answers a different
    question: MAE is the average error in the target's own units (MW —
    intuitive for a stakeholder conversation), RMSE penalizes large
    individual misses more heavily than MAE does (relevant if big
    surprises matter more than being consistently a little off, e.g. for
    capacity planning), and MAPE is scale-free (a percentage), which
    makes it comparable across periods where load itself is much higher
    or lower (e.g. comparing forecast quality at night vs. daytime peak).

    Args:
        y_true: Actual values.
        y_pred: Predicted values, same length and order as y_true.

    Returns:
        {"mae": ..., "rmse": ..., "mape": ...} (MAPE as a percentage,
        e.g. 8.7 means 8.7%).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / y_true))) * 100

    metrics = {"mae": mae, "rmse": rmse, "mape": mape}
    logger.info("Evaluation: %s", metrics)
    return metrics
