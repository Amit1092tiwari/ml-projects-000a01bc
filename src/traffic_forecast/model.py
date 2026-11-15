"""Time-series modeling and evaluation module for traffic forecast."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, ClassVar

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)

from src.traffic_forecast.features import create_calendar_features

logger = logging.getLogger(__name__)

# Check optional Prophet availability
try:
    from prophet import Prophet

    HAS_PROPHET = True
except ImportError:
    try:
        from fbprophet import Prophet

        HAS_PROPHET = True
    except ImportError:
        HAS_PROPHET = False


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    """Calculate standard regression and time-series evaluation metrics."""
    mae = float(mean_absolute_error(y_true, y_pred))
    mse = float(mean_squared_error(y_true, y_pred))
    rmse = float(np.sqrt(mse))
    # Handle zero division safely for MAPE
    non_zero = y_true != 0
    if np.any(non_zero):
        mape = float(mean_absolute_percentage_error(y_true[non_zero], y_pred[non_zero])) * 100.0
    else:
        mape = 0.0

    return {
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
        "mape_percent": round(mape, 2),
    }


class TrafficForecaster:
    """Production wrapper for traffic forecasting with fallback support."""

    FEATURE_COLS: ClassVar[list[str]] = [
        "dayofweek",
        "day",
        "month",
        "year",
        "quarter",
        "is_weekend",
        "dow_sin",
        "dow_cos",
        "month_sin",
        "month_cos",
    ]

    def __init__(self, use_prophet: bool = True):
        self.use_prophet = use_prophet and HAS_PROPHET
        self.model: Any | None = None
        self.backend = "prophet" if self.use_prophet else "gradient_boosting"
        logger.info("Initialized TrafficForecaster with backend: %s", self.backend)

    def fit(self, train_df: pd.DataFrame) -> TrafficForecaster:
        """Train the model on historical traffic data.

        Args:
            train_df: DataFrame with 'ds' (datetime) and 'y' (target count).
        """
        df = train_df.copy()
        if self.use_prophet:
            prophet_model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=True,
                daily_seasonality=False,
                seasonality_prior_scale=0.9,
            )
            prophet_model.fit(df[["ds", "y"]])
            self.model = prophet_model
        else:
            features = create_calendar_features(df, datetime_col="ds")
            X = features[self.FEATURE_COLS]
            y = features["y"]
            gb = GradientBoostingRegressor(
                n_estimators=150,
                learning_rate=0.05,
                max_depth=4,
                random_state=42,
            )
            gb.fit(X, y)
            self.model = gb

        return self

    def predict(self, future_df: pd.DataFrame) -> pd.DataFrame:
        """Generate forecast predictions.

        Args:
            future_df: DataFrame containing 'ds' column.

        Returns:
            DataFrame containing 'ds', 'yhat', and confidence intervals.
        """
        if self.model is None:
            raise RuntimeError("Model is not fitted yet. Call fit() before predict().")

        df = future_df.copy()
        if self.use_prophet:
            forecast = self.model.predict(df[["ds"]])
            output = pd.DataFrame({
                "ds": forecast["ds"],
                "yhat": np.maximum(0, forecast["yhat"]),  # Traffic cannot be negative
                "yhat_lower": np.maximum(0, forecast["yhat_lower"]),
                "yhat_upper": np.maximum(0, forecast["yhat_upper"]),
            })
        else:
            features = create_calendar_features(df, datetime_col="ds")
            X = features[self.FEATURE_COLS]
            preds = np.maximum(0, self.model.predict(X))
            # Heuristic standard deviation interval for tree model
            std_err = np.std(preds) * 0.15
            output = pd.DataFrame({
                "ds": features["ds"],
                "yhat": preds,
                "yhat_lower": np.maximum(0, preds - 1.96 * std_err),
                "yhat_upper": preds + 1.96 * std_err,
            })

        return output

    def save(self, filepath: str | Path) -> None:
        """Serialize model artifact to disk."""
        path = Path(filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"backend": self.backend, "model": self.model}, path)
        logger.info("Saved model artifact to %s", path)

    @classmethod
    def load(cls, filepath: str | Path) -> TrafficForecaster:
        """Load serialized model artifact from disk."""
        data = joblib.load(filepath)
        forecaster = cls(use_prophet=(data["backend"] == "prophet"))
        forecaster.model = data["model"]
        forecaster.backend = data["backend"]
        return forecaster
