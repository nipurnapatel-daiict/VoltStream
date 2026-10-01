import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import load_indian_grid_data
from src.feature_engineering import engineer_grid_features
from src.model_pipeline import train_anomaly_classifier
from src.load_forecaster import train_load_forecaster
from src.unsupervised_engine import train_unsupervised_detector  # New Import
from src.app import simulate_live_packet_ingestion

def main():
    print("="*60)
    print("   VOLTSTREAM: SYSTEM ENGINE CORE INITIALIZATION   ")
    print("="*60)
    
    try:
        raw_data = load_indian_grid_data()
        processed_data = engineer_grid_features(raw_data)
        
        # Supervised Classifier
        classifier_model, feature_names = train_anomaly_classifier(processed_data)
        
        # Unsupervised Isolation Forest 
        unsupervised_model = train_unsupervised_detector(processed_data)
        
        # Time-Series Regressor Forecaster
        forecaster_model = train_load_forecaster(processed_data)
        
        print("\n[LOG] Booting real-time simulated product stream integration...")
        
        # Scenario A: Normal early morning packet 
        normal_live_packet = {
            "timestamp": "2026-05-15 04:00:00",
            "grid_load_MW": 34000.0,
            "hour": 4,
            "day_of_week": 4,
            "is_weekend": 0,
            "month": 5
        }
        
        # Scenario B: Extreme demand spike packet 
        anomaly_live_packet = {
            "timestamp": "2026-05-15 04:00:00",
            "grid_load_MW": 85000.0,  # Massive spike
            "hour": 4,
            "day_of_week": 4,
            "is_weekend": 0,
            "month": 5
        }
        
        simulate_live_packet_ingestion(classifier_model, unsupervised_model, forecaster_model, feature_names, normal_live_packet)
        simulate_live_packet_ingestion(classifier_model, unsupervised_model, forecaster_model, feature_names, anomaly_live_packet)
        
        print("\n" + "="*60)
        print("   [SUCCESS] VOLTSTREAM CORE PROCESS RUN COMPLETE   ")
        print("="*60)

    except Exception as e:
        print(f"\n[CRITICAL SYSTEM ERROR] Pipeline aborted: {str(e)}")

if __name__ == "__main__":
    main()
