import os
import joblib
from src.app import simulate_live_packet_ingestion

def get_valid_int(prompt, min_val, max_val):
    """Helper function to ensure user input is within a valid range."""
    while True:
        try:
            val = int(input(prompt).strip())
            if min_val <= val <= max_val:
                return val
            print(f"Value must be between {min_val} and {max_val}.")
        except ValueError:
            print("Invalid input. Please enter a valid whole number.")

def main():
    print("="*60)
    print("       VOLTSTREAM: LIVE INTERACTIVE INFERENCE CORE       ")
    print("="*60)
    
    if not os.path.exists("models/anomaly_classifier.joblib"):
        print("[ERROR] Pre-compiled models missing. Please run 'python train.py' first.")
        return

    print("[LOG] Restoring pre-trained multi-engine matrices...")
    supervised_clf = joblib.load("models/anomaly_classifier.joblib")
    unsupervised_if = joblib.load("models/unsupervised_detector.joblib")
    forecaster = joblib.load("models/load_forecaster.joblib")
    feature_names = joblib.load("models/feature_names.joblib")
    print("[SUCCESS] Core grid engines online and operational.")

    while True:
        print("\n" + "-"*50)
        user_input = input("Enter Observed Regional Load in MW (or type 'exit'): ").strip()
        
        if user_input.lower() == 'exit':
            print("Shutting down VoltStream gateway. Goodbye!")
            break
            
        try:
            current_load = float(user_input)
            
            print("\nConfigure Telemetry Packet Timestamp Context:")
            hour = get_valid_int("  Enter Hour (0-23): ", 0, 23)
            month = get_valid_int("  Enter Month (1-12): ", 1, 12)
            day_of_week = get_valid_int("  Enter Day of Week (0=Monday, 6=Sunday): ", 0, 6)
            is_weekend = 1 if day_of_week >= 5 else 0
            
            simulated_timestamp = f"2026-{month:02d}-15 {hour:02d}:00:00"
            
            live_packet = {
                "timestamp": simulated_timestamp, 
                "grid_load_MW": current_load,
                "hour": hour,
                "day_of_week": day_of_week,
                "is_weekend": is_weekend,
                "month": month
            }
            
            simulate_live_packet_ingestion(
                supervised_clf, unsupervised_if, forecaster, feature_names, live_packet
            )
            
        except ValueError:
            print("Invalid entry. Please enter a valid numerical load metric.")

if __name__ == "__main__":
    main()
