import numpy as np
import pandas as pd
import datetime

def generate_sensor_telemetry(num_records=1000):
    """Generates synthetic IoT multi-sensor telemetry data."""
    np.random.seed(42)
    base_time = datetime.datetime.now()
    timestamps = [base_time + datetime.timedelta(seconds=10 * i) for i in range(num_records)]
    
    device_ids = np.random.choice(['DEV_101', 'DEV_102', 'DEV_103'], size=num_records)
    temperature = np.random.normal(loc=72.0, scale=4.0, size=num_records)
    vibration = np.random.normal(loc=1.5, scale=0.3, size=num_records)
    pressure = np.random.normal(loc=30.0, scale=2.0, size=num_records)
    
    # Inject synthetic anomalies (e.g. overheating / mechanical fault)
    temperature[120:125] += 30.0  # Temperature spike
    vibration[450:455] += 2.5     # Vibration anomaly
    
    df = pd.DataFrame({
        'timestamp': timestamps,
        'device_id': device_ids,
        'temperature': temperature,
        'vibration': vibration,
        'pressure': pressure
    })
    
    # Introduce some realistic missing values
    df.loc[10:12, 'temperature'] = np.nan
    return df

def clean_and_process_telemetry(df):
    """Cleans sensor logs and computes statistical rolling metrics."""
    # Handle missing sensor telemetry using forward-fill
    df = df.copy()
    df['temperature'] = df['temperature'].ffill()
    
    # Calculate rolling statistics per device
    df['temp_rolling_avg'] = df.groupby('device_id')['temperature'].transform(lambda x: x.rolling(window=5, min_periods=1).mean())
    df['temp_rolling_std'] = df.groupby('device_id')['temperature'].transform(lambda x: x.rolling(window=5, min_periods=1).std()).fillna(0)
    
    # Flag statistical anomalies (Z-score > 2.5)
    df['is_anomaly'] = (df['temperature'] - df['temp_rolling_avg']).abs() > (2.5 * (df['temp_rolling_std'] + 1e-5))
    return df

if __name__ == "__main__":
    print("Ingesting IoT telemetry streams...")
    raw_data = generate_sensor_telemetry()
    processed_data = clean_and_process_telemetry(raw_data)
    
    anomalies = processed_data[processed_data['is_anomaly']]
    print(f"Ingested {len(raw_data)} telemetry records.")
    print(f"Detected {len(anomalies)} anomalous operational windows:")
    print(anomalies[['timestamp', 'device_id', 'temperature', 'vibration']].head())
    
    # Save cleaned logs to CSV
    processed_data.to_csv("processed_telemetry.csv", index=False)
    print("Exported processed records to processed_telemetry.csv")