import pandas as pd

def simulate_live_packet_ingestion(supervised_clf, unsupervised_if, forecaster, feature_names, live_packet: dict):
    """
    SIMULATED PRODUCTION GATEWAY:
    Executes a Dual-Verification protocol using both Supervised and Unsupervised paths.
    """
    input_df = pd.DataFrame([live_packet])
    input_df["timestamp"] = pd.to_datetime(input_df["timestamp"])
    
    input_df["hour"] = input_df["timestamp"].dt.hour
    input_df["day_of_week"] = input_df["timestamp"].dt.dayofweek
    input_df["is_weekend"] = input_df["day_of_week"].apply(lambda x: 1 if x >= 5 else 0)
    input_df["month"] = input_df["timestamp"].dt.month
    
    forecast_features = ["hour", "day_of_week", "is_weekend", "month"]
    predicted_value = float(forecaster.predict(input_df[forecast_features])[0])
    
    X_live = input_df[feature_names]
    supervised_pred = supervised_clf.predict(X_live)[0]
    
    unsupervised_pred = unsupervised_if.predict(X_live)[0]
    unsupervised_flag = 1 if unsupervised_pred == -1 else 0
    
    print(f"\n[LIVE INGESTION] Node Timestamp: {live_packet['timestamp']}")
    print(f"Observed Load: {live_packet['grid_load_MW']} MW | Expected Base Forecast: {predicted_value:.2f} MW")
    print(f"Pipeline Decisions -> Supervised Flag: {supervised_pred} | Unsupervised Flag: {unsupervised_flag}")
    
    if supervised_pred == 1 and unsupervised_flag == 1:
        print("[CRITICAL ALERT] DUAL-VERIFIED ENERGY ANOMALY! Both systems confirm a major grid spike.")
    elif supervised_pred == 1 or unsupervised_flag == 1 or (live_packet['grid_load_MW'] > (predicted_value * 1.3)):
        print("[WARNING] SUSPICIOUS ANOMALY FLAGGED. Single-source deviation detected. Routing to secondary check.")
        
        reasons = []
        if supervised_pred == 1:
            reasons.append("Supervised Random Forest Classifier flagged historical spike pattern")
        if unsupervised_flag == 1:
            reasons.append("Unsupervised Isolation Forest detected multi-dimensional data shape outlier")
        if live_packet['grid_load_MW'] > (predicted_value * 1.3):
            reasons.append(f"Observed load exceeded the 30% baseline variance threshold (Limit: {predicted_value * 1.3:.2f} MW)")
            
        for reason in reasons:
            print(f"  └─> Trigger Source: {reason}")
    else:
        print("[SYSTEM OK] All core parameters match normal grid optimization baselines.")

