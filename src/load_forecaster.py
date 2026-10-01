from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


def train_load_forecaster(df):
    """
    STABLE ML REGRESSION FEATURE:
    Predicts the continuous exact 'grid_load_MW' consumption based on temporal features.
    """
    print("[LOG] Initializing Time-Series Forecasting Engine...")

    features = ["hour", "day_of_week", "is_weekend", "month"]
    X = df[features]
    y = df["grid_load_MW"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=True
    )

    model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    print("\n" + "=" * 45)
    print("      VOLTSTREAM: LOAD FORECASTER REGRESSION      ")
    print("=" * 45)
    print(f"Mean Absolute Error: {mae:.2f} MW")
    print("=" * 45)

    return model
