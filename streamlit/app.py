import requests
import streamlit as st


API_URL = (
    "http://127.0.0.1:8000"
)



st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title(
    "Customer Churn Prediction"
)

st.write(
    "Estimate the probability that a customer "
    "will become inactive during the next 90 days."
)


st.divider()


st.subheader(
    "Customer Lookup"
)

customer_id = st.number_input(
    "Enter Customer ID",
    min_value=1,
    step=1,
    value=17850
)


predict_button = st.button(
    "Predict Churn",
    type="primary"
)


if predict_button:

    with st.spinner(
        "Analyzing customer..."
    ):

        try:

            response = requests.get(
                f"{API_URL}/predict/{int(customer_id)}",
                timeout=15
            )

            response.raise_for_status()

            result = response.json()


        except requests.exceptions.HTTPError:

            if response.status_code == 404:

                st.error(
                    f"Customer {int(customer_id)} "
                    "was not found."
                )

            else:

                st.error(
                    f"API error: {response.text}"
                )

            st.stop()


        except requests.exceptions.ConnectionError:

            st.error(
                "Could not connect to FastAPI. "
                "Make sure the API is running."
            )

            st.stop()


        except requests.exceptions.RequestException as e:

            st.error(
                f"Request failed: {e}"
            )

            st.stop()


    customer = result["customer"]

    prediction = result["prediction"]

    probability = (
        prediction["churn_probability"]
    )

    percentage = (
        prediction["churn_percentage"]
    )

    risk_level = (
        prediction["risk_level"]
    )


    st.divider()

    st.subheader(
        f"Customer {int(customer['CustomerID'])}"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Churn Probability",
            f"{percentage:.1f}%"
        )


    with col2:

        st.metric(
            "Risk Level",
            risk_level
        )


    with col3:

        st.metric(
            "Recency",
            f"{customer['Recency']:.0f} days"
        )



    if risk_level == "High":

        st.error(
            "HIGH CHURN RISK — "
            "this customer should be prioritised "
            "for retention action."
        )

    elif risk_level == "Medium":

        st.warning(
            "MEDIUM CHURN RISK — "
            "this customer should be monitored "
            "and considered for targeted engagement."
        )

    else:

        st.success(
            "LOW CHURN RISK — "
            "this customer currently shows relatively "
            "low risk of becoming inactive."
        )



    st.subheader(
        "Customer Behaviour"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Frequency",
            f"{customer['Frequency']} orders"
        )


    with col2:

        st.metric(
            "Total Spend",
            f"£{customer['Monetary']:,.2f}"
        )


    with col3:

        st.metric(
            "Average Order",
            f"£{customer['AvgOrderValue']:,.2f}"
        )


    with col4:

        st.metric(
            "Unique Products",
            f"{customer['UniqueProducts']}"
        )


    st.subheader(
        "Recent Activity"
    )


    col1, col2, col3, col4 = st.columns(4)


    with col1:

        st.metric(
            "Orders - Last 30 Days",
            f"{customer['OrdersLast30Days']}"
        )


    with col2:

        st.metric(
            "Spend - Last 30 Days",
            f"£{customer['SpendLast30Days']:,.2f}"
        )


    with col3:

        st.metric(
            "Previous 30-Day Spend",
            f"£{customer['SpendPrevious30Days']:,.2f}"
        )


    with col4:

        if customer["SpendChange"] is not None:

            st.metric(
                "Spend Change",
                f"{customer['SpendChange']:.2f}"
            )

        else:

            st.metric(
                "Spend Change",
                "N/A"
            )


    st.subheader(
        "Purchase Pattern"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Avg. Days Between Orders",
            f"{customer['AvgDaysBetweenOrders']:.1f}"
        )


    with col2:

        st.metric(
            "Std. Days Between Orders",
            f"{customer['StdDaysBetweenOrders']:.1f}"
        )


    with col3:

        st.metric(
            "Average Basket Size",
            f"{customer['AvgBasketSize']:.1f}"
        )

    # RETURNS

    st.subheader(
        "Return Behaviour"
    )


    col1, col2, col3 = st.columns(3)


    with col1:

        st.metric(
            "Return Count",
            f"{customer['ReturnCount']}"
        )


    with col2:

        st.metric(
            "Returned Quantity",
            f"{customer['ReturnedQuantity']:.0f}"
        )


    with col3:

        st.metric(
            "Return Rate",
            f"{customer['ReturnRate']:.1%}"
        )


    # RECOMMENDATION

    st.subheader(
        "Recommended Action"
    )


    if risk_level == "High":

        st.write(
            "Prioritise this customer for a "
            "retention campaign. Consider a personalised "
            "offer, reminder, or direct engagement."
        )

    elif risk_level == "Medium":

        st.write(
            "Monitor this customer closely and consider "
            "a targeted re-engagement campaign."
        )

    else:

        st.write(
            "Continue normal engagement and monitor "
            "future changes in purchasing behaviour."
        )