"""Unit tests for forecasting model and metric calculations."""

import numpy as np
import pandas as pd

from src.traffic_forecast.model import TrafficForecaster, calculate_metrics


def test_calculate_metrics():
    y_true = np.array([100.0, 200.0, 300.0])
    y_pred = np.array([110.0, 190.0, 300.0])

    metrics = calculate_metrics(y_true, y_pred)
    assert metrics["mae"] == round((10.0 + 10.0 + 0.0) / 3.0, 2)
    assert metrics["rmse"] > 0
    assert "mape_percent" in metrics


def test_traffic_forecaster_fit_predict(tmp_path):
    dates = pd.date_range("2022-01-01", periods=60, freq="D")
    y = np.linspace(100, 200, 60) + np.random.normal(0, 5, 60)
    train_df = pd.DataFrame({"ds": dates, "y": y})

    # Test GradientBoosting fallback explicitly
    forecaster = TrafficForecaster(use_prophet=False)
    forecaster.fit(train_df)

    future_dates = pd.date_range("2022-03-02", periods=10, freq="D")
    future_df = pd.DataFrame({"ds": future_dates})
    preds = forecaster.predict(future_df)

    assert len(preds) == 10
    assert "yhat" in preds.columns
    assert (preds["yhat"] >= 0).all()  # Non-negative constraint

    # Test model serialization
    model_file = tmp_path / "model.joblib"
    forecaster.save(model_file)
    assert model_file.is_file()

    loaded = TrafficForecaster.load(model_file)
    loaded_preds = loaded.predict(future_df)
    np.testing.assert_allclose(preds["yhat"].values, loaded_preds["yhat"].values)
