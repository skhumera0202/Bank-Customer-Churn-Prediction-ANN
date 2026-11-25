
import streamlit as st
from prediction import predict_churn

# Page configuration
st.set_page_config(
    page_title="Bank Customer Churn Prediction",
    page_icon="🏦",
    layout="centered"
)

# Application title
st.title("🏦 Bank Customer Churn Prediction")

st.write(
    "Enter customer details below to predict "
    "whether the customer is likely to leave the bank."
)

st.divider()

# Customer information
st.subheader("Customer Information")

col1, col2 = st.columns(2)

with col1:
    credit_score = st.number_input(
        "Credit Score",
        min_value=300,
        max_value=900,
        value=650
    )

    geography = st.selectbox(
        "Geography",
        ["France", "Germany", "Spain"]
    )

    gender = st.selectbox(
        "Gender",
        ["Female", "Male"]
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=35
    )

    tenure = st.number_input(
        "Tenure (Years)",
        min_value=0,
        max_value=10,
        value=5
    )

with col2:
    balance = st.number_input(
        "Account Balance",
        min_value=0.0,
        value=60000.0,
        step=1000.0
    )

    num_of_products = st.selectbox(
        "Number of Products",
        [1, 2, 3, 4]
    )

    has_cr_card = st.selectbox(
        "Has Credit Card?",
        ["Yes", "No"]
    )

    is_active_member = st.selectbox(
        "Is Active Member?",
        ["Yes", "No"]
    )

    estimated_salary = st.number_input(
        "Estimated Salary",
        min_value=0.0,
        value=50000.0,
        step=1000.0
    )

st.divider()

# Prediction button
if st.button("Predict Customer Churn", type="primary"):

    # Convert Yes/No values into 1/0
    card_value = 1 if has_cr_card == "Yes" else 0
    active_value = 1 if is_active_member == "Yes" else 0

    # Call prediction function
    result, probability = predict_churn(
        credit_score=credit_score,
        geography=geography,
        gender=gender,
        age=age,
        tenure=tenure,
        balance=balance,
        num_of_products=num_of_products,
        has_cr_card=card_value,
        is_active_member=active_value,
        estimated_salary=estimated_salary
    )

    st.subheader("Prediction Result")

    st.metric(
        "Churn Probability",
        f"{probability:.2%}"
    )

    st.progress(probability)

    if probability >= 0.5:
        st.error(result)
    else:
        st.success(result)

    st.caption(
        "This is a model-generated estimate, not a guarantee "
        "of customer behaviour."
    )
