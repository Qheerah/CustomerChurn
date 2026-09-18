import pandas as pd

path="../data/raw/Online Retail.xlsx"
def load_data(path):
    df = pd.read_excel(path)
    return df


def clean_data(df):
    df = df.copy()

    df["TransactionValue"] = (
        df["Quantity"] * df["UnitPrice"]
    )

    # Customer-level modelling requires CustomerID
    df = df.dropna(
        subset=["CustomerID"]
    )

    df["Description"] = (
        df["Description"]
        .fillna("No description")
    )

    return df