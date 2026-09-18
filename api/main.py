from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException

from src.data_processing import clean_data
from src.features import create_customer_snapshot


# --------------------------------------------------
# PROJECT PATHS
# --------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = (
    BASE_DIR
    / "data"
    / "raw"
    / "Online Retail.xlsx"
)

MODEL_PATH = (
    BASE_DIR
    / "models"
    / "churn_model.pkl"
)

FEATURE_PATH = (
    BASE_DIR
    / "models"
    / "feature_columns.pkl"
)


# --------------------------------------------------
# LOAD AND CLEAN DATA
# --------------------------------------------------

df = pd.read_excel(DATA_PATH)

df = clean_data(df)

model = joblib.load(MODEL_PATH)

feature_columns = joblib.load(FEATURE_PATH)


# --------------------------------------------------
# FASTAPI APP
# --------------------------------------------------

app = FastAPI(
    title="Online Retail Churn Prediction API",
    description=(
        "Predicts the probability that a customer "
        "will become inactive during the next 90 days."
    ),
    version="1.0.0"
)


# --------------------------------------------------
# HOME / HEALTH CHECK
# --------------------------------------------------

@app.get("/")
def root():

    return {
        "message": "Customer Churn Prediction API is running"
    }


# --------------------------------------------------
# PREDICT CUSTOMER CHURN
# --------------------------------------------------

@app.get("/predict/{customer_id}")
def predict_customer(customer_id: int):

    # ----------------------------------------------
    # Check whether customer exists
    # ----------------------------------------------

    customer_data = df[
        df["CustomerID"] == float(customer_id)
    ].copy()

    if customer_data.empty:

        raise HTTPException(
            status_code=404,
            detail=f"Customer {customer_id} was not found."
        )

    # ----------------------------------------------
    # Use latest available transaction as
    # observation date
    # ----------------------------------------------

    observation_date = df["InvoiceDate"].max()

    # ----------------------------------------------
    # Create customer features
    #
    # include_target=False means we only create
    # prediction features, not the Churn label.
    # ----------------------------------------------

    customer_features = create_customer_snapshot(
        df=customer_data,
        observation_date=observation_date,
        prediction_days=90,
        include_target=False
    )

    if customer_features.empty:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Unable to create features "
                f"for customer {customer_id}."
            )
        )

    # ----------------------------------------------
    # Select exactly the features used by model
    # ----------------------------------------------

    missing_features = [
        feature
        for feature in feature_columns
        if feature not in customer_features.columns
    ]

    if missing_features:

        raise HTTPException(
            status_code=500,
            detail={
                "message": "Required model features are missing.",
                "missing_features": missing_features
            }
        )

    X = customer_features[
        feature_columns
    ].copy()

    # ----------------------------------------------
    # Handle missing values
    # ----------------------------------------------

    X["AvgDaysBetweenOrders"] = (
        X["AvgDaysBetweenOrders"]
        .fillna(X["Recency"])
    )

    X["StdDaysBetweenOrders"] = (
        X["StdDaysBetweenOrders"]
        .fillna(0)
    )

    X["SpendChange"] = (
        X["SpendChange"]
        .fillna(0)
    )

    # ----------------------------------------------
    # Model prediction
    # ----------------------------------------------

    probability = model.predict_proba(X)[0, 1]

    prediction = int(
        probability >= 0.5
    )

    # ----------------------------------------------
    # Risk level
    # ----------------------------------------------

    if probability >= 0.70:

        risk_level = "High"

    elif probability >= 0.40:

        risk_level = "Medium"

    else:

        risk_level = "Low"

    # ----------------------------------------------
    # Customer metrics
    # ----------------------------------------------

    row = customer_features.iloc[0]

    customer_summary = {

        "CustomerID": int(customer_id),

        "Recency": float(
            row["Recency"]
        ),

        "Frequency": int(
            row["Frequency"]
        ),

        "Monetary": float(
            row["Monetary"]
        ),

        "AvgOrderValue": float(
            row["AvgOrderValue"]
        ),

        "AvgBasketSize": float(
            row["AvgBasketSize"]
        ),

        "UniqueProducts": int(
            row["UniqueProducts"]
        ),

        "OrdersLast30Days": int(
            row["OrdersLast30Days"]
        ),

        "SpendLast30Days": float(
            row["SpendLast30Days"]
        ),

        "OrdersPrevious30Days": int(
            row["OrdersPrevious30Days"]
        ),

        "SpendPrevious30Days": float(
            row["SpendPrevious30Days"]
        ),

        "SpendChange": (
            None
            if pd.isna(row["SpendChange"])
            else float(row["SpendChange"])
        ),

        "AvgDaysBetweenOrders": float(
            row["AvgDaysBetweenOrders"]
        ),

        "StdDaysBetweenOrders": float(
            row["StdDaysBetweenOrders"]
        ),

        "ReturnCount": int(
            row["ReturnCount"]
        ),

        "ReturnedQuantity": float(
            row["ReturnedQuantity"]
        ),

        "ReturnedValue": float(
            row["ReturnedValue"]
        ),

        "ReturnRate": float(
            row["ReturnRate"]
        )
    }

    # ----------------------------------------------
    # API response
    # ----------------------------------------------

    return {

        "customer": customer_summary,

        "prediction": {

            "churn_probability": round(
                float(probability),
                4
            ),

            "churn_percentage": round(
                float(probability) * 100,
                2
            ),

            "churn_prediction": prediction,

            "risk_level": risk_level
        }
    }