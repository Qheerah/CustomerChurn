import joblib
import pandas as pd


MODEL_PATH = "models/churn_model.pkl"
FEATURE_PATH = "models/feature_columns.pkl"


model = joblib.load(MODEL_PATH)
feature_columns = joblib.load(FEATURE_PATH)


def predict_churn(features: dict) -> dict:

    X = pd.DataFrame([features])

    # Ensure exactly the same feature order
    X = X[feature_columns]

    probability = model.predict_proba(X)[0, 1]

    prediction = int(probability >= 0.5)

    if probability >= 0.70:
        risk_level = "High"
    elif probability >= 0.40:
        risk_level = "Medium"
    else:
        risk_level = "Low"

    return {
        "churn_probability": float(probability),
        "prediction": prediction,
        "risk_level": risk_level
    }