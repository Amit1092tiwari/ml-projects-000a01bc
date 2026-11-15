"""Unit tests for traffic data loading and schema validation."""

import pandas as pd
import pytest

from src.traffic_forecast.data_loader import load_traffic_data, validate_schema


def test_validate_schema_valid():
    df = pd.DataFrame({"Datetime": ["01-01-2022 00:00"], "Count": [10]})
    # Should not raise
    validate_schema(df)


def test_validate_schema_missing_column():
    df = pd.DataFrame({"WrongColumn": [1, 2, 3]})
    with pytest.raises(ValueError, match="Input data is missing required columns"):
        validate_schema(df)


def test_load_traffic_data_synthetic(tmp_path):
    csv_file = tmp_path / "sample_traffic.csv"
    data = {
        "ID": [1, 2, 3, 4],
        "Datetime": [
            "01-01-2022 00:00",
            "01-01-2022 01:00",
            "02-01-2022 00:00",
            "02-01-2022 01:00",
        ],
        "Count": [10, 20, 30, 40],
    }
    pd.DataFrame(data).to_csv(csv_file, index=False)

    df = load_traffic_data(csv_file, resample_freq="D")

    assert set(df.columns) == {"ds", "y"}
    assert len(df) == 2
    # Day 1 sum should be 10 + 20 = 30
    assert df.loc[0, "y"] == 30
    # Day 2 sum should be 30 + 40 = 70
    assert df.loc[1, "y"] == 70


def test_load_traffic_data_nonexistent_file():
    with pytest.raises(FileNotFoundError):
        load_traffic_data("non_existent_file.csv")
