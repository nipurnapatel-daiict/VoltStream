import os
import pandas as pd

def load_indian_grid_data(filepath="data/hourlyLoadDataIndia.csv"):
    """
    Ingests raw Grid-India logs and isolates the Northern Region zone 
    to preserve distinct weather-driven seasonality spikes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found at {filepath}")
    
    print("[LOG] Ingesting raw Grid-India hourly data streams...")
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip()
    
    time_col = [col for col in df.columns if 'date' in col.lower() or 'time' in col.lower()][0]
    
    nr_candidates = [col for col in df.columns if 'nr' in col.lower() or 'northen' in col.lower()]
    
    if nr_candidates:
        target_load_col = nr_candidates[0]
        print(f"[LOG] Identified Time Marker: '{time_col}' | Targeted Regional Node: '{target_load_col}'")
    else:
        target_load_col = [col for col in df.columns if col != time_col][0]
        print(f"[WARNING] NR Column not explicitly found. Defaulting to: '{target_load_col}'")
    
    df = df[[time_col, target_load_col]].copy()
    df.columns = ["timestamp", "grid_load_MW"]
    
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").reset_index(drop=True)
    df["grid_load_MW"] = df["grid_load_MW"].ffill()
    
    print(f"[SUCCESS] Ingested {len(df)} regional zone hourly data records.")
    return df
