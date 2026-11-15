"""Data loading and schema validation module for traffic forecasting."""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"Datetime", "Count"}


def validate_schema(df: pd.DataFrame) -> None:
    """Validate that the input dataframe contains the required columns."""
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input data is missing required columns: {missing}. Found: {list(df.columns)}")


def load_traffic_data(
    file_path: str | Path,
    resample_freq: str = "D",
    datetime_col: str = "Datetime",
    target_col: str = "Count",
) -> pd.DataFrame:
    """Load raw traffic CSV data, validate schema, parse timestamps, and aggregate.

    Args:
        file_path: Path to the traffic CSV dataset.
        resample_freq: Frequency string for resampling (e.g. 'D' for daily sum, 'h' for hourly).
        datetime_col: Name of the timestamp column.
        target_col: Name of the traffic count column.

    Returns:
        DataFrame with standardized columns 'ds' (datetime) and 'y' (target count).
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Traffic dataset not found at: {path.resolve()}")

    logger.info("Loading traffic data from %s", path)
    df = pd.read_csv(path)
    validate_schema(df)

    # Clean nulls
    initial_rows = len(df)
    df = df.dropna(subset=[datetime_col, target_col]).copy()
    dropped_nulls = initial_rows - len(df)
    if dropped_nulls > 0:
        logger.warning("Dropped %d rows with null values", dropped_nulls)

    # Convert timestamps
    try:
        df[datetime_col] = pd.to_datetime(df[datetime_col], format="%d-%m-%Y %H:%M")
    except (ValueError, TypeError):
        # Fallback to general datetime parser
        df[datetime_col] = pd.to_datetime(df[datetime_col], errors="coerce")
        df = df.dropna(subset=[datetime_col])

    df = df.sort_values(datetime_col).reset_index(drop=True)
    df.set_index(datetime_col, inplace=True)

    # Aggregate to specified frequency (daily by default)
    if resample_freq:
        resampled = df[[target_col]].resample(resample_freq).sum()
    else:
        resampled = df[[target_col]].copy()

    result_df = pd.DataFrame({
        "ds": resampled.index,
        "y": resampled[target_col].values,
    }).reset_index(drop=True)

    logger.info("Successfully loaded and resampled %d time periods", len(result_df))
    return result_df
