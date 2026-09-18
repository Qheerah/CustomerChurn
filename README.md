# Online Retail Customer Churn Prediction

A machine learning project for analyzing customer purchasing behaviour and predicting whether a customer is likely to become inactive during a subsequent 90-day period.

The project uses the **Online Retail** transaction dataset and combines exploratory data analysis, RFM customer segmentation, Pareto analysis, feature engineering, and a LightGBM classification model. A FastAPI service provides predictions, while a Streamlit application is used as the user-facing dashboard.

## Project Objectives

The project aims to:

- Analyze customer purchasing behaviour.
- Identify countries and products with high sales and return activity.
- Examine revenue trends over time.
- Measure customer revenue concentration using Pareto analysis.
- Segment customers using RFM analysis.
- Build a model that predicts 90-day customer inactivity risk.
- Expose the trained model through an API.
- Provide a simple dashboard where a user enters a Customer ID and receives a churn-risk result.

## Dataset

The project uses the Kaggle **Online Retail** dataset.

The original dataset contains transaction-level records with the following fields:

- `InvoiceNo`
- `StockCode`
- `Description`
- `Quantity`
- `InvoiceDate`
- `UnitPrice`
- `CustomerID`
- `Country`

The dataset contains both purchases and returns. Negative quantities represent returned items.

During preprocessing:

- Transactions without `CustomerID` are removed.
- `TransactionValue` is calculated as `Quantity × UnitPrice`.
- Positive quantities are treated as sales.
- Negative quantities are treated as returns.

The dataset used in the analysis covers **1 December 2010 to 9 December 2011**.



## Analysis

### Exploratory Data Analysis

The analysis examines:

- Sales revenue by country.
- Returned products.
- Revenue trends by quarter.
- Customer purchasing behaviour.
- Product purchasing patterns.
- Customer revenue concentration.

### RFM Analysis

Customers are segmented using:

- **Recency** — how recently the customer purchased.
- **Frequency** — number of distinct invoices.
- **Monetary** — total transaction value.

The resulting segments include:

- Champions
- Loyal High-value Customers
- Hibernating / Average
- Lost
- Potential Loyalists
- At Risk High Value
- At risk low-value

### Pareto Analysis

The project also examines how revenue is distributed across customers to determine the concentration of customer value.

## Churn Definition

For this project, churn is treated as a **90-day inactivity proxy**.

A customer is labelled as churned when they make **no purchase during the 90-day period following an observation date**.

Therefore, the model predicts:

> **The probability that a customer will become inactive during the next 90 days.**

This is a modelling definition of inactivity and should not be interpreted as permanent customer loss.

## Feature Engineering

Customer-level features are generated from transaction history.

The model uses behavioural, purchasing, value, and return-related features, including:

- Recency
- Frequency
- Monetary
- Average basket behaviour
- Average order value
- Unique products purchased
- Orders in the last 30 days
- Spend in the last 30 days
- Quantity in the last 30 days
- Orders in the previous 30 days
- Spend in the previous 30 days
- Spend change
- Average days between orders
- Standard deviation of days between orders
- Return count
- Returned quantity
- Returned value
- Purchased quantity
- Return rate

## Model

The main predictive model is **LightGBM**.

A recency-only baseline was also evaluated to provide a comparison with the full feature set.

### Validation Strategy

The model uses a **temporal train/test split** rather than randomly shuffling observations.

Training observations come from earlier customer observation periods, while the test observations come from later periods. This better reflects the intended real-world use case, where historical behaviour is used to predict future inactivity.

### Model Performance

Latest reported test-set results:

| Metric | Recency Baseline | LightGBM |
|---|---:|---:|
| ROC-AUC | 0.6472 | 0.7759 |
| PR-AUC | 0.4683 | 0.5925 |

For the LightGBM model:

- Precision: **0.5850**
- Recall: **0.5405**
- F1-score: **0.5619**
- Recall@10%: **0.2179**

Recall@10% is measured on customer-period observations, since the modelling dataset contains repeated observations for customers across time.

## Project Structure

```text
CustomerChurn/
│
├── api/
│   ├── __init__.py
│   └── main.py
│
├── data/
│   └── raw/
│       └── Online Retail.xlsx
│
├── models/
│   ├── churn_model.pkl
│   └── feature_columns.pkl
│
├── notebooks/
│   └── customer_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── data_processing.py
│   ├── features.py
│   ├── model.py
│   └── predict.py
│
├── streamlit/
│   └── app.py
│
├── train.py
├── report.pdf
└── README.md
```

## Installation

Create and activate a Python virtual environment:

```bash
python -m venv my_env
source my_env/bin/activate
```

On Windows:

```bash
my_env\\Scripts\\activate
```

Install the required packages:

```bash
pip install pandas openpyxl scikit-learn lightgbm joblib fastapi uvicorn streamlit
```

## Data Setup

Place the raw dataset here:

```text
data/raw/Online Retail.xlsx
```

The expected filename is exactly:

```text
Online Retail.xlsx
```

## Training the Model

Run the training pipeline from the **project root**:

```bash
python train.py
```

The training process performs the following steps:

```text
Load raw data
    ↓
Clean transactions
    ↓
Create customer observation snapshots
    ↓
Generate customer features
    ↓
Create 90-day inactivity target
    ↓
Apply temporal train/test split
    ↓
Train LightGBM
    ↓
Evaluate model
    ↓
Save trained model
```

The trained files are saved in:

```text
models/churn_model.pkl
models/feature_columns.pkl
```

## Running the FastAPI Service

Start FastAPI from the **project root**, not from inside the `api/` directory:

```bash
uvicorn api.main:app --reload
```

The API will normally be available at:

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

### Health Check

```http
GET /
```

Example response:

```json
{
  "message": "Customer Churn Prediction API is running"
}
```

### Customer Prediction

```http
GET /predict/{customer_id}
```

Example:

```text
http://127.0.0.1:8000/predict/17850
```

The endpoint returns:

- Customer information
- Behavioural metrics
- Churn probability
- Churn percentage
- Binary churn prediction
- Risk level

## Running the Streamlit Dashboard

Start Streamlit from the project root:

```bash
streamlit run streamlit/app.py
```

The dashboard is designed so that the user only needs to enter a **Customer ID**. The application sends the request to FastAPI, receives the prediction, and displays the customer's churn risk and supporting behavioural information.

The intended flow is:

```text
Streamlit
    ↓
Customer ID
    ↓
FastAPI
    ↓
Feature generation
    ↓
Saved LightGBM model
    ↓
Churn probability
    ↓
Streamlit dashboard
```

## API and Model Design

The predictive service does **not** train the model when a prediction is requested.

Training is handled by:

```text
train.py
```

The FastAPI application loads the previously trained model:

```text
models/churn_model.pkl
```

and the saved model feature order:

```text
models/feature_columns.pkl
```

This separation keeps training and prediction independent.

## Important Notes

### Feature consistency

The same preprocessing and feature-engineering logic should be used during training and prediction. Changes to the feature definitions may require retraining the model and regenerating the saved model files.

### Observation date

During API inference, the latest available transaction date in the dataset is used as the observation date. Predictions are therefore based on the historical data available in the supplied dataset.

### Risk thresholds

The dashboard currently groups predicted probabilities into three display categories:

- **Low:** probability below 0.40
- **Medium:** probability from 0.40 to below 0.70
- **High:** probability of 0.70 or above

These categories are application-level risk labels and are separate from the model's underlying probability.

## Limitations

The project has several limitations:

- The dataset covers approximately one year, so there are only a limited number of valid observation periods for a full 90-day future window.
- The 90-day inactivity definition is a proxy for churn rather than a confirmed measure of permanent customer loss.
- Customer observations are repeated over time, and the 90-day observation windows overlap.
- The dataset contains transaction information but does not include demographic, marketing, customer-service, or external economic variables.
- Model predictions are probabilistic and should be used as decision-support information rather than certainty about future customer behaviour.

## Technologies

- Python
- Pandas
- Scikit-learn
- LightGBM
- Joblib
- FastAPI
- Uvicorn
- Streamlit
- Jupyter Notebook
- Excel dataset


**Customer Churn Prediction Project**

This project was developed as a practical machine learning application covering data analysis, feature engineering, customer segmentation, predictive modelling, and API/dashboard deployment.
