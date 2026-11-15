"""Integration test for daily batch forecast pipeline."""

import pandas as pd

from src.traffic_forecast.pipeline import run_batch_forecast_pipeline


def test_batch_forecast_pipeline_synthetic(tmp_path):
    data_file = tmp_path / "traffic.csv"
    dates = pd.date_range("2022-01-01", periods=100, freq="D")
    df = pd.DataFrame({
        "ID": range(100),
        "Datetime": [d.strftime("%d-%m-%Y %H:%M") for d in dates],
        "Count": [10 + i * 0.5 for i in range(100)],
    })
    df.to_csv(data_file, index=False)

    out_dir = tmp_path / "artifacts"
    result = run_batch_forecast_pipeline(
        data_path=str(data_file),
        output_dir=str(out_dir),
        forecast_days=14,
        test_days=10,
        use_prophet=False,
    )

    assert result["pipeline_status"] == "SUCCESS"
    assert result["forecast_days"] == 14
    assert (out_dir / "daily_traffic_forecast.csv").is_file()
    assert (out_dir / "forecast_metrics.json").is_file()
    assert (out_dir / "model.joblib").is_file()

    # Verify forecast file shape
    forecast_df = pd.read_csv(out_dir / "daily_traffic_forecast.csv")
    assert len(forecast_df) == 14
    assert {"ds", "yhat", "yhat_lower", "yhat_upper"}.issubset(set(forecast_df.columns))
