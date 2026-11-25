

import streamlit as st
from prediction import predict_churn


# ==================================================
# 1. PAGE CONFIGURATION
# ==================================================

st.set_page_config(
    page_title="Churn Intelligence | Bank Analytics",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ==================================================
# 2. CUSTOM CSS
# ==================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #151116;
    color: #F5EBDD;
}

/* Main content spacing */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #21191F;
    border-right: 1px solid #60404E;
}

/* Sidebar text */
[data-testid="stSidebar"] * {
    color: #F2E5D8 !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #F2D5B8 !important;
}

[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
    color: #CDB6AC !important;
}

[data-testid="stSidebar"] hr {
    border-color: #60404E !important;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #F2D5B8 !important;
}

/* Regular text */
.stMarkdown p,
.stMarkdown li {
    color: #F5EBDD;
}

/* Input labels */
[data-testid="stWidgetLabel"] p,
[data-testid="stWidgetLabel"] label,
.stTextInput label,
.stNumberInput label,
.stSelectbox label {
    color: #E8C9AD !important;
}

/* Input fields */
.stNumberInput input,
.stTextInput input {
    background-color: #F1F0F3 !important;
    color: #241B22 !important;
    border-radius: 8px;
}

/* Select boxes */
.stSelectbox div[data-baseweb="select"] > div {
    background-color: #F1F0F3;
    color: #241B22;
    border-radius: 8px;
}

/* Help text */
[data-testid="stTooltipIcon"] {
    color: #E8C9AD !important;
}

/* Hero section */
.hero {
    background: linear-gradient(120deg, #38232E, #21191F);
    padding: 35px;
    border-radius: 18px;
    border: 1px solid #684453;
    margin-bottom: 30px;
}

.hero h1 {
    color: #F4E5D1 !important;
    font-size: 36px;
    font-weight: 700;
    margin: 0 0 12px 0;
}

.hero p {
    color: #D6C0B5;
    font-size: 16px;
    margin: 0;
    line-height: 1.7;
}

/* Section headings */
.section-heading {
    color: #E8C9AD;
    font-size: 22px;
    font-weight: 600;
    margin-top: 15px;
    margin-bottom: 12px;
}

/* Form container */
[data-testid="stForm"] {
    background-color: #1C161D;
    border: 1px solid #49323D;
    border-radius: 15px;
    padding: 22px;
}

/* Form submit button */
div[data-testid="stFormSubmitButton"] button {
    background: linear-gradient(90deg, #815064, #A27665);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 12px 20px;
    font-weight: 600;
    min-height: 48px;
    transition: 0.3s;
}

div[data-testid="stFormSubmitButton"] button:hover {
    background: linear-gradient(90deg, #A27665, #BD947D);
    color: white;
    border: none;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #261D25;
    border: 1px solid #60404E;
    padding: 20px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #D6C0B5 !important;
}

div[data-testid="stMetricValue"] {
    color: #F2D5B8 !important;
}

/* Progress bar */
.stProgress > div > div > div > div {
    background-color: #B58A75;
}

/* Expander */
[data-testid="stExpander"] {
    background-color: #21191F;
    border: 1px solid #60404E;
    border-radius: 10px;
}

[data-testid="stExpander"] summary {
    color: #F2D5B8 !important;
}

/* Alerts */
[data-testid="stAlert"] {
    border-radius: 10px;
}

/* Dividers */
hr {
    border-color: #49323D;
}

/* Footer */
.footer {
    color: #BDA9AE;
    text-align: center;
    font-size: 13px;
    margin-top: 45px;
    padding: 18px;
    border-top: 1px solid #49323D;
}

</style>
""", unsafe_allow_html=True)


# ==================================================
# 3. SIDEBAR
# ==================================================

with st.sidebar:

    st.markdown("## 🏦 Churn Intelligence")

    st.caption("AI-Powered Banking Analytics")

    st.divider()

    st.markdown("### About the Project")

    st.write(
        "This application uses an Artificial Neural Network "
        "to estimate the likelihood of a bank customer leaving."
    )

    st.divider()

    st.markdown("### Model Information")

    st.write("**Algorithm:** Artificial Neural Network")

    st.write("**Framework:** TensorFlow / Keras")

    st.write("**Problem Type:** Binary Classification")

    st.write("**ROC-AUC:** 0.8580")

    st.divider()

    st.markdown("### Technologies")

    st.write("🐍 Python")

    st.write("🧠 TensorFlow")

    st.write("📊 Scikit-learn")

    st.write("🎨 Streamlit")

    st.divider()

    st.caption("Developed by Humera Shaikh")


# ==================================================
# 4. MAIN HERO HEADER
# ==================================================

st.html("""
<div class="hero">

    <h1>Customer Churn Intelligence</h1>

    <p>
        Understand customer behaviour using Artificial Intelligence.
        Enter customer details to estimate their churn probability.
    </p>

</div>
""")


# ==================================================
# 5. CUSTOMER PROFILE SECTION
# ==================================================

st.markdown(
    '<div class="section-heading">Customer Profile</div>',
    unsafe_allow_html=True
)

st.write(
    "Provide the customer's account and demographic information below."
)


# ==================================================
# 6. CUSTOMER INPUT FORM
# ==================================================

with st.form("customer_prediction_form"):

    col1, col2 = st.columns(2)

    # Personal and account details
    with col1:

        st.markdown("#### 👤 Personal & Account Details")

        credit_score = st.number_input(
            "Credit Score",
            min_value=300,
            max_value=900,
            value=650,
            help="Customer credit score."
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

    # Banking information
    with col2:

        st.markdown("#### 💳 Banking Information")

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
            "Active Member?",
            ["Yes", "No"]
        )

        estimated_salary = st.number_input(
            "Estimated Salary",
            min_value=0.0,
            value=50000.0,
            step=1000.0
        )

    st.divider()

    submitted = st.form_submit_button(
        "✨ Analyse Customer Churn",
        use_container_width=True
    )


# ==================================================
# 7. PREDICTION AND RESULTS
# ==================================================

if submitted:

    # Convert Yes/No values into numerical values
    card_value = 1 if has_cr_card == "Yes" else 0

    active_value = 1 if is_active_member == "Yes" else 0

    try:

        with st.spinner("Analysing customer information..."):

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

        # Ensure probability is between 0 and 1
        probability = float(probability)

        probability = max(0.0, min(1.0, probability))

        # Prediction section
        st.divider()

        st.markdown(
            '<div class="section-heading">Prediction Analysis</div>',
            unsafe_allow_html=True
        )

        # Display metrics
        metric_col1, metric_col2 = st.columns(2)

        with metric_col1:

            st.metric(
                label="Churn Probability",
                value=f"{probability:.2%}"
            )

        with metric_col2:

            classification = (
                "Likely to Leave"
                if probability >= 0.5
                else "Likely to Stay"
            )

            st.metric(
                label="Model Classification",
                value=classification
            )

        # Probability indicator
        st.markdown("#### Churn Probability Indicator")

        st.progress(probability)

        st.caption(
            f"The model estimates a {probability:.2%} "
            "probability of customer churn."
        )

        # Display prediction message
        if probability >= 0.5:

            st.error(
                "⚠️ Customer is classified as likely to leave."
            )

        else:

            st.success(
                "✓ Customer is classified as likely to stay."
            )

        # Prediction details
        with st.expander("📋 View Prediction Details"):

            st.write(f"**Model response:** {result}")

            st.write("**Decision threshold:** 50%")

            st.write(
                f"**Estimated probability:** {probability:.4f}"
            )

        # Disclaimer
        st.info(
            "This prediction is a model-generated estimate based "
            "on the supplied information. It does not guarantee "
            "actual customer behaviour."
        )

    except Exception as e:

        st.error(
            f"Prediction could not be completed: {e}"
        )


# ==================================================
# 8. FOOTER
# ==================================================

st.html("""
<div class="footer">

    Built with Python, TensorFlow and Streamlit

    <br><br>

    Bank Customer Churn Prediction | Humera Shaikh

</div>
""")
