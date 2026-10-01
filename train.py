import os
import sys
import joblib

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.data_ingestion import load_indian_grid_data
from src.feature_engineering import engineer_grid_features
from src.model_pipeline import train_anomaly_classifier
from src.unsupervised_engine import train_unsupervised_detector
from src.load_forecaster import train_load_forecaster

def main():
    print("="*60)
    print("        VOLTSTREAM: SECURE ML TRAINING PIPELINE        ")
    print("="*60)
    
    os.makedirs("models", exist_ok=True)
    
    try:
        raw_data = load_indian_grid_data()
        processed_data = engineer_grid_features(raw_data)
        
        classifier_model, feature_names = train_anomaly_classifier(processed_data)
        
        unsupervised_model = train_unsupervised_detector(processed_data)
        
        forecaster_model = train_load_forecaster(processed_data)
        
        print("\n[LOG] Serializing trained ML infrastructure onto disk...")
        joblib.dump(classifier_model, "models/anomaly_classifier.joblib")
        joblib.dump(unsupervised_model, "models/unsupervised_detector.joblib")
        joblib.dump(forecaster_model, "models/load_forecaster.joblib")
        joblib.dump(feature_names, "models/feature_names.joblib")
        
        print("\n" + "="*60)
        print("[SUCCESS] ALL 3 ENGINES COMPILED & FROZEN TO STORAGE SUCCESSFULLY!")
        print("="*60)

    except Exception as e:
        print(f"\n[CRITICAL ERROR] Training pipeline aborted: {str(e)}")

if __name__ == "__main__":
    main()


"""
$ python train.py
============================================================
        VOLTSTREAM: SECURE ML TRAINING PIPELINE        
============================================================
[LOG] Ingesting raw Grid-India hourly data streams...
[LOG] Identified Time Marker: 'datetime' | Targeted Regional Node: 'Northen Region Hourly Demand'
[SUCCESS] Ingested 46728 regional zone hourly data records.
[LOG] Transforming raw logs into temporal and statistical features...
[LOG] Initializing Anomaly Classification Engine...

=============================================
      VOLTSTREAM: ANOMALY CLASSIFIER MATRIX       
=============================================
              precision    recall  f1-score   support

           0       1.00      1.00      1.00      8541
           1       0.99      0.98      0.98       805

    accuracy                           1.00      9346
   macro avg       0.99      0.99      0.99      9346
weighted avg       1.00      1.00      1.00      9346

=============================================

[LOG] Initializing Unsupervised Isolation Forest Engine...
💡 [SIMPLE SUMMARY] Unsupervised Isolation Forest trained successfully on multi-dimensional patterns.
[LOG] Initializing Time-Series Forecasting Engine...

=============================================
      VOLTSTREAM: LOAD FORECASTER REGRESSION      
=============================================
Mean Absolute Error: 5378.34 MW
=============================================

[LOG] Serializing trained ML infrastructure onto disk...

============================================================
🎉 [SUCCESS] ALL 3 ENGINES COMPILED & FROZEN TO STORAGE SUCCESSFULLY!
============================================================
(venv) 

"""