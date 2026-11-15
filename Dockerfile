# Hardened, lightweight container image for Batch ML Inference
FROM python:3.10-slim

# Prevent Python from writing .pyc files and enable unbuffered logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app

WORKDIR /app

# Install minimal OS build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# Copy application modules and configs
COPY src/ ./src/
COPY dags/ ./dags/
COPY "Traffic Forecast - Time Series Analysis/" "./Traffic Forecast - Time Series Analysis/"

# Security: Create and switch to non-root service account
RUN useradd --create-home appuser && chown -R appuser:appuser /app
USER appuser

# Default execution: run the daily batch forecast pipeline
CMD ["python", "-m", "src.traffic_forecast.pipeline", \
     "--data-path", "Traffic Forecast - Time Series Analysis/Traffic data.csv", \
     "--output-dir", "artifacts/traffic_forecast", \
     "--forecast-days", "30"]
