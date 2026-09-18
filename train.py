import os
import joblib
import pandas as pd

from src.data_processing import (
    load_data,
    clean_data
)

from src.features import (
    create_customer_snapshot
)

from src.model import (
    split_temporally,
    train_model,
    evaluate_model
)



DATA_PATH = "data/raw/Online Retail.xlsx"

MODEL_DIR = "models"

MODEL_PATH = os.path.join(
    MODEL_DIR,
    "churn_model.pkl"
)

FEATURE_PATH = os.path.join(
    MODEL_DIR,
    "feature_columns.pkl"
)



os.makedirs(
    MODEL_DIR,
    exist_ok=True
)



# 1. LOAD DATA


print("=" * 50)
print("1. LOADING DATA")
print("=" * 50)

df = load_data(DATA_PATH)

print(
    f"Raw dataset shape: {df.shape}"
)


# 2. CLEAN DATA


print("\n" + "=" * 50)
print("2. CLEANING DATA")
print("=" * 50)

df = clean_data(df)

print(
    f"Cleaned dataset shape: {df.shape}"
)



# 3. CREATE OBSERVATION DATES


print("\n" + "=" * 50)
print("3. CREATING OBSERVATION DATES")
print("=" * 50)

min_date = df["InvoiceDate"].min()
max_date = df["InvoiceDate"].max()

prediction_window = pd.Timedelta(
    days=90
)

observation_dates = pd.date_range(
    start=min_date + prediction_window,
    end=max_date - prediction_window,
    freq="MS"
)

print(
    f"Number of observation dates: "
    f"{len(observation_dates)}"
)

print(
    f"First observation date: "
    f"{observation_dates.min()}"
)

print(
    f"Last observation date: "
    f"{observation_dates.max()}"
)


# 4. BUILD CUSTOMER SNAPSHOTS


print("\n" + "=" * 50)
print("4. BUILDING CUSTOMER SNAPSHOTS")
print("=" * 50)

snapshots = []

for observation_date in observation_dates:

    print(
        f"Creating snapshot: "
        f"{observation_date.date()}"
    )

    snapshot = create_customer_snapshot(
        df=df,
        observation_date=observation_date,
        prediction_days=90
    )

    snapshots.append(snapshot)


# Combine all snapshots
model_data = pd.concat(
    snapshots,
    ignore_index=True
)

print(
    f"\nModel dataset shape: "
    f"{model_data.shape}"
)



# 5. REMOVE CUSTOMERS WITH LITTLE HISTORY


print("\n" + "=" * 50)
print("5. FILTERING CUSTOMER HISTORY")
print("=" * 50)

before_filter = len(model_data)

model_data = model_data[
    model_data["Frequency"] >= 2
].copy()

after_filter = len(model_data)

print(
    f"Rows before filter: {before_filter}"
)

print(
    f"Rows after filter:  {after_filter}"
)



# 6. HANDLE MISSING VALUES


print("\n" + "=" * 50)
print("6. HANDLING MISSING VALUES")
print("=" * 50)

model_data["AvgDaysBetweenOrders"] = (
    model_data["AvgDaysBetweenOrders"]
    .fillna(model_data["Recency"])
)

model_data["StdDaysBetweenOrders"] = (
    model_data["StdDaysBetweenOrders"]
    .fillna(0)
)

# SpendChange can be NaN when there was
# no spend in the previous 30-day period.
model_data["SpendChange"] = (
    model_data["SpendChange"]
    .fillna(0)
)

print(
    "Missing values handled."
)



# 7. CHURN DISTRIBUTION


print("\n" + "=" * 50)
print("7. CHURN DISTRIBUTION")
print("=" * 50)

print(
    model_data["Churn"]
    .value_counts()
)

print("\nChurn proportions:")

print(
    model_data["Churn"]
    .value_counts(
        normalize=True
    )
    .round(4)
)



# 8. TEMPORAL TRAIN / TEST SPLIT


print("\n" + "=" * 50)
print("8. TEMPORAL TRAIN / TEST SPLIT")
print("=" * 50)

X_train, X_test, y_train, y_test = (
    split_temporally(model_data)
)

print(
    f"X_train shape: {X_train.shape}"
)

print(
    f"X_test shape:  {X_test.shape}"
)

print(
    f"y_train shape: {y_train.shape}"
)

print(
    f"y_test shape:  {y_test.shape}"
)



# 9. TRAIN LIGHTGBM MODEL


print("\n" + "=" * 50)
print("9. TRAINING LIGHTGBM")
print("=" * 50)

model = train_model(
    X_train,
    y_train
)

print(
    "Model training completed."
)



# 10. EVALUATE MODEL


print("\n" + "=" * 50)
print("10. MODEL EVALUATION")
print("=" * 50)

results = evaluate_model(
    model,
    X_test,
    y_test
)


# 11. SAVE MODEL


print("\n" + "=" * 50)
print("11. SAVING MODEL")
print("=" * 50)

joblib.dump(
    model,
    MODEL_PATH
)

joblib.dump(
    X_train.columns.tolist(),
    FEATURE_PATH
)

print(
    f"Model saved to: {MODEL_PATH}"
)

print(
    f"Features saved to: {FEATURE_PATH}"
)



# 12. FINAL SUMMARY


print("\n" + "=" * 50)
print("TRAINING COMPLETE")
print("=" * 50)

print(
    f"ROC-AUC: "
    f"{results['roc_auc']:.4f}"
)

print(
    f"PR-AUC:  "
    f"{results['pr_auc']:.4f}"
)