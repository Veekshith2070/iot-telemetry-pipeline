# IoT Sensor Telemetry & Anomaly Detection Pipeline

A lightweight data engineering pipeline built with **Python, Pandas, and NumPy** to ingest, clean, and process multi-sensor IoT time-series logs.

## Features
- **Data Ingestion & Cleaning:** Ingests high-frequency device telemetry (temperature, vibration, pressure) and handles missing readings via forward-fill strategies.
- **Rolling Window Analytics:** Computes rolling averages and local standard deviation per `device_id`.
- **Anomaly Detection:** Implements statistical Z-score thresholding to flag mechanical faults and temperature spikes in real-time.

## Installation & Usage
```bash
pip install pandas numpy
python pipeline.py