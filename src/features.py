import pandas as pd
import numpy as np


def create_customer_snapshot(
    df: pd.DataFrame,
    observation_date: pd.Timestamp,
    prediction_days: int = 90
):

    observation_date = pd.Timestamp(
        observation_date
    )

    prediction_end = (
        observation_date
        + pd.Timedelta(days=prediction_days)
    )

    
    # Historical transactions
    
    history = df[
        df["InvoiceDate"] <= observation_date
    ].copy()

    sales_history = history[
        history["Quantity"] > 0
    ].copy()

    
    # Future transactions
    

    future = df[
        (df["InvoiceDate"] > observation_date)
        & (df["InvoiceDate"] <= prediction_end)
    ].copy()

    future_sales = future[
        future["Quantity"] > 0
    ].copy()

    
    # Base customer features
    

    features = (
        sales_history
        .groupby("CustomerID")
        .agg(
            Recency=(
                "InvoiceDate",
                lambda x:
                    (observation_date - x.max()).days
            ),
            Frequency=(
                "InvoiceNo",
                "nunique"
            ),
            Monetary=(
                "TransactionValue",
                "sum"
            ),
            AvgBasketSize=(
                "Quantity",
                "mean"
            ),
            UniqueProducts=(
                "StockCode",
                "nunique"
            )
        )
        .reset_index()
    )

    
    # Actual average order value
    

    order_values = (
        sales_history
        .groupby(
            ["CustomerID", "InvoiceNo"]
        )["TransactionValue"]
        .sum()
        .reset_index()
    )

    avg_order_value = (
        order_values
        .groupby("CustomerID")["TransactionValue"]
        .mean()
        .rename("AvgOrderValue")
        .reset_index()
    )

    features = features.merge(
        avg_order_value,
        on="CustomerID",
        how="left"
    )

    
    # Last 30 days
    

    last_30_start = (
        observation_date
        - pd.Timedelta(days=30)
    )

    recent_30 = sales_history[
        sales_history["InvoiceDate"]
        > last_30_start
    ]

    recent_features = (
        recent_30
        .groupby("CustomerID")
        .agg(
            OrdersLast30Days=(
                "InvoiceNo",
                "nunique"
            ),
            SpendLast30Days=(
                "TransactionValue",
                "sum"
            ),
            QuantityLast30Days=(
                "Quantity",
                "sum"
            )
        )
        .reset_index()
    )

    features = features.merge(
        recent_features,
        on="CustomerID",
        how="left"
    )

    
    # Previous 30 days
    
    previous_30_start = (
        observation_date
        - pd.Timedelta(days=60)
    )

    previous_30 = sales_history[
        (sales_history["InvoiceDate"]
         > previous_30_start)
        &
        (sales_history["InvoiceDate"]
         <= last_30_start)
    ]

    previous_features = (
        previous_30
        .groupby("CustomerID")
        .agg(
            OrdersPrevious30Days=(
                "InvoiceNo",
                "nunique"
            ),
            SpendPrevious30Days=(
                "TransactionValue",
                "sum"
            )
        )
        .reset_index()
    )

    features = features.merge(
        previous_features,
        on="CustomerID",
        how="left"
    )

   
    # Spend change
   

    features["SpendChange"] = np.where(
        features["SpendPrevious30Days"] > 0,
        (
            features["SpendLast30Days"]
            / features["SpendPrevious30Days"]
        ),
        np.nan
    )

   
    # Order intervals

    sorted_sales = sales_history.sort_values(
        ["CustomerID", "InvoiceDate"]
    ).copy()

    sorted_sales["PreviousOrderDate"] = (
        sorted_sales
        .groupby("CustomerID")["InvoiceDate"]
        .shift(1)
    )

    sorted_sales["DaysBetweenOrders"] = (
        sorted_sales["InvoiceDate"]
        - sorted_sales["PreviousOrderDate"]
    ).dt.days

    interval_features = (
        sorted_sales
        .groupby("CustomerID")["DaysBetweenOrders"]
        .agg(
            AvgDaysBetweenOrders="mean",
            StdDaysBetweenOrders="std"
        )
        .reset_index()
    )

    features = features.merge(
        interval_features,
        on="CustomerID",
        how="left"
    )

    
    # Return behaviour

    returns_history = history[
        history["Quantity"] < 0
    ].copy()

    return_features = (
        returns_history
        .groupby("CustomerID")
        .agg(
            ReturnCount=(
                "InvoiceNo",
                "nunique"
            ),
            ReturnedQuantity=(
                "Quantity",
                lambda x: x.abs().sum()
            ),
            ReturnedValue=(
                "TransactionValue",
                lambda x: x.abs().sum()
            )
        )
        .reset_index()
    )

    features = features.merge(
        return_features,
        on="CustomerID",
        how="left"
    )

    
    # Fill missing values
    

    zero_columns = [
        "OrdersLast30Days",
        "SpendLast30Days",
        "QuantityLast30Days",
        "OrdersPrevious30Days",
        "SpendPrevious30Days",
        "ReturnCount",
        "ReturnedQuantity",
        "ReturnedValue"
    ]

    features[zero_columns] = (
        features[zero_columns]
        .fillna(0)
    )

    features["AvgDaysBetweenOrders"] = (
        features["AvgDaysBetweenOrders"]
        .fillna(features["Recency"])
    )

    features["StdDaysBetweenOrders"] = (
        features["StdDaysBetweenOrders"]
        .fillna(0)
    )

   
    # Return rate
   
    purchase_quantity = (
        sales_history
        .groupby("CustomerID")["Quantity"]
        .sum()
        .rename("PurchasedQuantity")
        .reset_index()
    )

    features = features.merge(
        purchase_quantity,
        on="CustomerID",
        how="left"
    )

    features["ReturnRate"] = np.where(
        features["PurchasedQuantity"] > 0,
        (
            features["ReturnedQuantity"]
            / features["PurchasedQuantity"]
        ),
        0
    )

   
    # Churn target
    

    future_customers = set(
        future_sales["CustomerID"].unique()
    )

    features["Churn"] = (
        ~features["CustomerID"]
        .isin(future_customers)
    ).astype(int)

    features["ObservationDate"] = (
        observation_date
    )

    return features