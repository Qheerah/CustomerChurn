import joblib
import pandas as pd

from lightgbm import LGBMClassifier

from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    classification_report,
    confusion_matrix
)


FEATURE_COLUMNS = [
    "Recency",
    "Frequency",
    "Monetary",
    "AvgOrderValue",
    "AvgBasketSize",
    "UniqueProducts",
    "OrdersLast30Days",
    "SpendLast30Days",
    "QuantityLast30Days",
    "OrdersPrevious30Days",
    "SpendPrevious30Days",
    "SpendChange",
    "AvgDaysBetweenOrders",
    "StdDaysBetweenOrders",
    "ReturnCount",
    "ReturnedQuantity",
    "ReturnedValue",
    "ReturnRate"
]


def split_temporally(
    model_data: pd.DataFrame
):

    model_data = model_data.sort_values(
        "ObservationDate"
    )

    dates = sorted(
        model_data["ObservationDate"].unique()
    )

    split_date = dates[
        int(len(dates) * 0.75)
    ]

    train_data = model_data[
        model_data["ObservationDate"]
        < split_date
    ].copy()

    test_data = model_data[
        model_data["ObservationDate"]
        >= split_date
    ].copy()

    X_train = train_data[
        FEATURE_COLUMNS
    ]

    y_train = train_data["Churn"]

    X_test = test_data[
        FEATURE_COLUMNS
    ]

    y_test = test_data["Churn"]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )


def train_model(
    X_train,
    y_train
):

    model = LGBMClassifier(
        objective="binary",
        learning_rate=0.02,
        n_estimators=300,
        num_leaves=31,
        min_child_samples=20,
        reg_alpha=0.1,
        reg_lambda=0.1,
        random_state=42,
        verbosity=-1
    )

    model.fit(
        X_train,
        y_train
    )

    return model


def evaluate_model(
    model,
    X_test,
    y_test
):

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    results = {
        "roc_auc": roc_auc_score(
            y_test,
            probabilities
        ),
        "pr_auc": average_precision_score(
            y_test,
            probabilities
        )
    }

    print(
        "ROC-AUC:",
        round(results["roc_auc"], 4)
    )

    print(
        "PR-AUC:",
        round(results["pr_auc"], 4)
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("\nConfusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    return results