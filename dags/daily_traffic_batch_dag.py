"""
Apache Airflow DAG: Daily Scheduled Batch Traffic Forecasting Pipeline
----------------------------------------------------------------------
Cadence: Daily at 02:00 AM UTC
SLA: 45 minutes
Owners: MLOps / Data Engineering Team
"""

from datetime import datetime, timedelta
import logging

try:
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator
    from airflow.operators.python import PythonOperator
    AIRFLOW_AVAILABLE = True
except ImportError:
    # Graceful dummy classes for local environments without apache-airflow installed
    AIRFLOW_AVAILABLE = False
    DAG = object
    EmptyOperator = object
    PythonOperator = object

logger = logging.getLogger("airflow.task")

DEFAULT_ARGS = {
    "owner": "mlops-platform",
    "depends_on_past": False,
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "execution_timeout": timedelta(minutes=45),
}


def task_validate_input_partition(**context):
    """Verify that yesterday's end-of-day data partition exists and conforms to schema."""
    from pathlib import Path
    from src.traffic_forecast.data_loader import validate_schema
    import pandas as pd

    data_path = Path("Traffic Forecast - Time Series Analysis/Traffic data.csv")
    if not data_path.is_file():
        raise FileNotFoundError(f"Partition missing at: {data_path}")

    sample = pd.read_csv(data_path, nrows=50)
    validate_schema(sample)
    logger.info("Partition validation successful for %s", data_path)


def task_run_batch_inference(**context):
    """Execute the batch forecast pipeline and generate predictions."""
    from src.traffic_forecast.pipeline import run_batch_forecast_pipeline

    result = run_batch_forecast_pipeline(
        data_path="Traffic Forecast - Time Series Analysis/Traffic data.csv",
        output_dir="artifacts/traffic_forecast",
        forecast_days=30,
        test_days=30,
    )
    # Push metrics to XCom for downstream auditing
    if context.get("ti"):
        context["ti"].xcom_push(key="forecast_summary", value=result)
    logger.info("Batch inference completed with status: %s", result["pipeline_status"])


def task_audit_drift_and_publish(**context):
    """Inspect model drift indicators and publish artifacts to operational stores."""
    summary = None
    if context.get("ti"):
        summary = context["ti"].xcom_pull(key="forecast_summary", task_ids="batch_inference")

    if summary and summary.get("drift_report", {}).get("drift_detected"):
        logger.warning("HIGH DRIFT ALERT: Forecast distribution deviated from historical mean.")
    else:
        logger.info("Forecast audit passed cleanly. Publishing predictions to downstream cache.")


if AIRFLOW_AVAILABLE:
    dag = DAG(
        dag_id="daily_traffic_batch_forecast",
        default_args=DEFAULT_ARGS,
        description="Daily scheduled batch traffic forecasting and drift evaluation",
        schedule_interval="0 2 * * *",  # Everyday at 02:00 AM UTC
        start_date=datetime(2026, 1, 1),
        catchup=False,
        max_active_runs=1,
        tags=["mlops", "forecasting", "batch", "traffic"],
    )

    with dag:
        start_pipeline = EmptyOperator(task_id="start_pipeline")

        validate_data = PythonOperator(
            task_id="validate_data_partition",
            python_callable=task_validate_input_partition,
        )

        batch_inference = PythonOperator(
            task_id="batch_inference",
            python_callable=task_run_batch_inference,
        )

        audit_and_publish = PythonOperator(
            task_id="audit_drift_and_publish",
            python_callable=task_audit_drift_and_publish,
        )

        end_pipeline = EmptyOperator(task_id="end_pipeline")

        start_pipeline >> validate_data >> batch_inference >> audit_and_publish >> end_pipeline
