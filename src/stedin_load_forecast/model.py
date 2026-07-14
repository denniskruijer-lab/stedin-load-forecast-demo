"""Forecasting model: a seasonal-naive baseline and a gradient-boosted
regressor trained on calendar + lag features, plus shared evaluation.
"""

import logging

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

logger = logging.getLogger(__name__)


def time_train_test_split(
    df: pd.DataFrame, test_fraction: float = 0.2
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological split — no shuffling, since this is time series data."""
    split_idx = int(len(df) * (1 - test_fraction))
    return df.iloc[:split_idx], df.iloc[split_idx:]


def seasonal_naive_predict(df: pd.DataFrame, season_lag_col: str = "lag_96") -> pd.Series:
    """Baseline: predict the same value as the same time one season ago.

    Default season_lag_col="lag_96" assumes ~15-minute resolution data,
    i.e. lag_96 = same time yesterday.
    """
    return df[season_lag_col]


def train_regressor(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    random_state: int = 42,
) -> GradientBoostingRegressor:
    logger.info("Training GradientBoostingRegressor on %d rows, %d features",
                len(train_df), len(feature_cols))
    model = GradientBoostingRegressor(random_state=random_state)
    model.fit(train_df[feature_cols], train_df[target_col])
    return model


def evaluate(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    """Mean absolute error, root mean squared error, mean absolute percentage error."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    errors = y_true - y_pred
    mae = float(np.mean(np.abs(errors)))
    rmse = float(np.sqrt(np.mean(errors**2)))
    mape = float(np.mean(np.abs(errors / y_true))) * 100

    metrics = {"mae": mae, "rmse": rmse, "mape": mape}
    logger.info("Evaluation: %s", metrics)
    return metrics
