from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_anomaly_classifier(df):
    """
    Executes a classification pipeline using a Random Forest Classifier
    to isolate normal patterns from abnormal grid overloads.
    """
    print("[LOG] Initializing Anomaly Classification Engine...")
    
    features = ["hour", "day_of_week", "is_weekend", "month", "grid_load_MW"]
    X = df[features]
    y = df["is_anomaly"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    
    predictions = model.predict(X_test)
    
    print("\n" + "="*45)
    print("      VOLTSTREAM: ANOMALY CLASSIFIER MATRIX       ")
    print("="*45)
    print(classification_report(y_test, predictions))
    print("="*45)
    
    return model, features
