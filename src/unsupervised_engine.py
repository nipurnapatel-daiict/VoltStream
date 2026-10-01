from sklearn.ensemble import IsolationForest

def train_unsupervised_detector(df):
    """
    STABLE ADVANCED ML FEATURE:
    Trains an Unsupervised Isolation Forest model to detect structural anomalies
    without relying on any historical label columns.
    """
    print("\n[LOG] Initializing Unsupervised Isolation Forest Engine...")
    
    features = ["hour", "day_of_week", "is_weekend", "month", "grid_load_MW"]
    X = df[features]
    
    model = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    model.fit(X)
    
    print("[SIMPLE SUMMARY] Unsupervised Isolation Forest trained successfully on multi-dimensional patterns.")
    return model
