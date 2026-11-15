"""Production Batch Inference Pipeline for Daily Traffic Forecasting."""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict
import pandas as pd

from src.traffic_forecast.data_loader import load_traffic_data
from src.traffic_forecast.features import generate_future_horizon
from src.traffic_forecast.model import TrafficForecaster, calculate_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger(__name__)


def check_data_drift(historical_series: pd.Series, predicted_series: pd.Series) -> Dict[str, Any]:
    """Check for statistical distribution drift between historical and forecast data."""
    hist_mean = float(historical_series.mean())
    hist_std = float(historical_series.std())
    pred_mean = float(predicted_series.mean())

    z_score = abs(pred_mean - hist_mean) / (hist_std if hist_std > 0 else 1.0)
    drift_detected = bool(z_score > 3.0)

    if drift_detected:
        logger.warning("DATA DRIFT WARNING: Forecast mean (%.2f) diverges from historical mean (%.2f) by %.2f sigma!",
                       pred_mean, hist_mean, z_score)
    else:
        logger.info("Data drift check passed (z-score: %.2f)", z_score)

    return {
        "drift_detected": drift_detected,
        "historical_mean": round(hist_mean, 2),
        "forecast_mean": round(pred_mean, 2),
        "z_score": round(z_score, 2),
    }


def run_batch_forecast_pipeline(
    data_path: str,
    output_dir: str = "artifacts/traffic_forecast",
    forecast_days: int = 30,
    test_days: int = 30,
    use_prophet: bool = True,
) -> Dict[str, Any]:
    """
    Execute end-to-end daily batch traffic forecasting pipeline.

    Steps:
      1. Load & validate data partition.
      2. Temporal train-test split for holdout evaluation.
      3. Fit forecasting model on historical partition.
      4. Compute validation metrics (MAE, RMSE, MAPE).
      5. Generate future batch forecast.
      6. Run data drift / anomaly check.
      7. Persist forecast artifacts (CSV, JSON metrics, model checkpoint).

    Returns:
        Dictionary summarizing execution status, metrics, and artifact paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Data
    logger.info("STEP 1: Ingesting dataset from %s", data_path)
    df = load_traffic_data(data_path, resample_freq="D")

    if len(df) <= test_days:
        raise ValueError(f"Dataset length ({len(df)}) must exceed test_days ({test_days})")

    # 2. Temporal Train-Test Split (Chronological, no future leakage)
    logger.info("STEP 2: Executing temporal holdout split (last %d days)", test_days)
    train_df = df.iloc[:-test_days].copy()
    test_df = df.iloc[-test_days:].copy()

    # 3. Fit Model
    logger.info("STEP 3: Training model on %d historical daily points", len(train_df))
    forecaster = TrafficForecaster(use_prophet=use_prophet)
    forecaster.fit(train_df)

    # 4. Evaluate Holdout
    logger.info("STEP 4: Evaluating holdout predictions")
    val_preds = forecaster.predict(test_df)
    metrics = calculate_metrics(test_df["y"].values, val_preds["yhat"].values)
    logger.info("Validation Metrics: MAE=%.2f, RMSE=%.2f, MAPE=%.2f%%",
                metrics["mae"], metrics["rmse"], metrics["mape_percent"])

    # Refit on full dataset for future batch forecasting
    logger.info("Refitting on full historical dataset (%d days) for batch horizon", len(df))
    full_forecaster = TrafficForecaster(use_prophet=use_prophet)
    full_forecaster.fit(df)

    # 5. Generate Future Horizon
    last_date = df["ds"].max()
    logger.info("STEP 5: Generating %d-day batch forecast from %s", forecast_days, last_date.strftime("%Y-%m-%d"))
    future_horizon = generate_future_horizon(last_date, periods=forecast_days, freq="D")
    forecast_df = full_forecaster.predict(future_horizon)

    # 6. Data Drift Check
    logger.info("STEP 6: Checking for distribution drift")
    drift_report = check_data_drift(df["y"], forecast_df["yhat"])

    # 7. Persist Artifacts
    logger.info("STEP 7: Exporting artifacts to %s", out_dir)
    forecast_csv = out_dir / "daily_traffic_forecast.csv"
    forecast_df.to_csv(forecast_csv, index=False)

    summary = {
        "pipeline_status": "SUCCESS",
        "backend": full_forecaster.backend,
        "historical_records": len(df),
        "forecast_days": forecast_days,
        "evaluation_metrics": metrics,
        "drift_report": drift_report,
        "forecast_start": str(future_horizon["ds"].min()),
        "forecast_end": str(future_horizon["ds"].max()),
    }

    metrics_json = out_dir / "forecast_metrics.json"
    with open(metrics_json, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    model_path = out_dir / "model.joblib"
    full_forecaster.save(model_path)

    logger.info("Batch forecast pipeline completed successfully!")
    return summary


def main():
    parser = argparse.ArgumentParser(description="Run daily batch traffic forecast pipeline.")
    parser.add_argument("--data-path", type=str, default="Traffic Forecast - Time Series Analysis/Traffic data.csv",
                        help="Path to raw traffic dataset CSV")
    parser.add_argument("--output-dir", type=str, default="artifacts/traffic_forecast",
                        help="Directory to save forecasts and metrics")
    parser.add_argument("--forecast-days", type=int, default=30,
                        help="Number of days to forecast")
    parser.add_argument("--test-days", type=int, default=30,
                        help="Number of days for holdout evaluation")
    parser.add_argument("--no-prophet", action="store_true",
                        help="Force using scikit-learn GradientBoosting fallback instead of Prophet")

    args = parser.parse_args()
    run_batch_forecast_pipeline(
        data_path=args.data_path,
        output_dir=args.output_dir,
        forecast_days=args.forecast_days,
        test_days=args.test_days,
        use_prophet=not args.no_prophet,
    )


if __name__ == "__main__":
    main()
