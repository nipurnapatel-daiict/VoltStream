import pandas as pd
import numpy as np

def engineer_grid_features(df):
    """
    Extracts time-series cyclic signatures and algorithmically builds a ground truth 
    for grid anomalies without using pre-manipulated inputs.
    """
    print("[LOG] Transforming raw logs into temporal and statistical features...")
    df = df.copy()
    
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    df["month"] = df["timestamp"].dt.month 
    
    hourly_stats = df.groupby("hour")["grid_load_MW"].agg(["mean", "std"]).reset_index()
    hourly_stats.columns = ["hour", "grid_mean", "grid_std"]
    
    df = df.merge(hourly_stats, on="hour", how="left")
    
    df["is_anomaly"] = (df["grid_load_MW"] > (df["mean"] if "mean" in df.columns else df["grid_mean"] + 1.5 * (df["std"] if "std" in df.columns else df["grid_std"]))).astype(int)
    
    cols_to_drop = [c for c in ["grid_mean", "grid_std", "mean", "std"] if c in df.columns]
    df = df.drop(columns=cols_to_drop)
    
    return df
