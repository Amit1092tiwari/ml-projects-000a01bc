"""Unit tests for feature engineering."""

import pandas as pd
from src.traffic_forecast.features import create_calendar_features, generate_future_horizon


def test_create_calendar_features():
    df = pd.DataFrame({"ds": pd.to_datetime(["2022-01-01", "2022-01-02", "2022-06-15"])})
    features = create_calendar_features(df, datetime_col="ds")

    expected_cols = ["dayofweek", "day", "month", "year", "quarter", "is_weekend", "dow_sin", "dow_cos", "month_sin", "month_cos"]
    for col in expected_cols:
        assert col in features.columns

    # 2022-01-01 is a Saturday (is_weekend = 1, dayofweek = 5)
    assert features.loc[0, "is_weekend"] == 1
    assert features.loc[0, "dayofweek"] == 5

    # 2022-06-15 is month 6, quarter 2
    assert features.loc[2, "month"] == 6
    assert features.loc[2, "quarter"] == 2


def test_generate_future_horizon():
    last_date = pd.Timestamp("2022-01-01")
    horizon = generate_future_horizon(last_date, periods=14, freq="D")

    assert len(horizon) == 14
    assert horizon.loc[0, "ds"] == pd.Timestamp("2022-01-02")
    assert horizon.iloc[-1]["ds"] == pd.Timestamp("2022-01-15")
    assert "dow_sin" in horizon.columns
