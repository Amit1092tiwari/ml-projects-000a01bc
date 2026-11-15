"""Feature engineering utilities for time series traffic forecasting."""

from __future__ import annotations

import numpy as np
import pandas as pd


def create_calendar_features(df: pd.DataFrame, datetime_col: str = "ds") -> pd.DataFrame:
    """Extract calendar, seasonal, and cyclical features from datetime column.

    Args:
        df: Input DataFrame containing datetime_col.
        datetime_col: Column name containing datetime objects.

    Returns:
        DataFrame enriched with calendar features.
    """
    features = df.copy()
    dt = pd.to_datetime(features[datetime_col])

    features["dayofweek"] = dt.dt.dayofweek
    features["day"] = dt.dt.day
    features["month"] = dt.dt.month
    features["year"] = dt.dt.year
    features["quarter"] = dt.dt.quarter
    features["is_weekend"] = (dt.dt.dayofweek >= 5).astype(int)

    # Cyclical encodings for smooth periodic transitions
    features["dow_sin"] = np.sin(2 * np.pi * features["dayofweek"] / 7.0)
    features["dow_cos"] = np.cos(2 * np.pi * features["dayofweek"] / 7.0)
    features["month_sin"] = np.sin(2 * np.pi * (features["month"] - 1) / 12.0)
    features["month_cos"] = np.cos(2 * np.pi * (features["month"] - 1) / 12.0)

    return features


def generate_future_horizon(last_date: pd.Timestamp, periods: int = 30, freq: str = "D") -> pd.DataFrame:
    """Generate an empty future DataFrame covering the forecasting horizon.

    Args:
        last_date: The latest timestamp in historical training data.
        periods: Number of future periods to forecast.
        freq: Frequency string (e.g., 'D' for daily).

    Returns:
        DataFrame with column 'ds' extending into the future.
    """
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=periods, freq=freq)
    future_df = pd.DataFrame({"ds": future_dates})
    return create_calendar_features(future_df, datetime_col="ds")
