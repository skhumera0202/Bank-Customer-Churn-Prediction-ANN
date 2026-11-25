

import joblib
import numpy as np
import tensorflow as tf
import pandas as pd

# ==================================================
# 1. Load the trained ANN model
# ==================================================

model = tf.keras.models.load_model(
    "ann_model.keras",
    compile=False
)

# ==================================================
# 2. Load saved preprocessing objects
# ==================================================

label_encoder = joblib.load("label_encoder.pkl")
one_hot_encoder = joblib.load("one_hot_encoder.pkl")
scaler = joblib.load("scaler.pkl")


# ==================================================
# 3. Create prediction function
# ==================================================

def predict_churn(
    credit_score,
    geography,
    gender,
    age,
    tenure,
    balance,
    num_of_products,
    has_cr_card,
    is_active_member,
    estimated_salary
):

    # Create a DataFrame for the new customer
    customer_data = pd.DataFrame([{
        "CreditScore": credit_score,
        "Gender": gender,
        "Age": age,
        "Tenure": tenure,
        "Balance": balance,
        "NumOfProducts": num_of_products,
        "HasCrCard": has_cr_card,
        "IsActiveMember": is_active_member,
        "EstimatedSalary": estimated_salary,
        "Geography": geography
    }])

    # Encode Gender
    customer_data["Gender"] = label_encoder.transform(
        customer_data["Gender"]
    )

    # Encode Geography
    geo_encoded = one_hot_encoder.transform(
        customer_data[["Geography"]]
    )

    geo_columns = one_hot_encoder.get_feature_names_out(
        ["Geography"]
    )

    geo_df = pd.DataFrame(
        geo_encoded,
        columns=geo_columns
    )

    # Remove original Geography column
    customer_data = customer_data.drop(
        "Geography",
        axis=1
    )

    # Combine encoded Geography features
    customer_data = pd.concat(
        [customer_data, geo_df],
        axis=1
    )

    # Ensure exact training feature order
    feature_order = [
        "CreditScore",
        "Gender",
        "Age",
        "Tenure",
        "Balance",
        "NumOfProducts",
        "HasCrCard",
        "IsActiveMember",
        "EstimatedSalary",
        "Geography_France",
        "Geography_Germany",
        "Geography_Spain"
    ]

    customer_data = customer_data[feature_order]

    # Scale customer data
    customer_scaled = scaler.transform(customer_data)

    # Make prediction
    probability = float(
        model.predict(customer_scaled, verbose=0)[0][0]
    )

    # Classification threshold
    threshold = 0.5

    if probability >= threshold:
        prediction = "Customer is likely to leave the bank."
    else:
        prediction = "Customer is likely to stay with the bank."

    return prediction, probability


# ==================================================
# 4. Test with one sample customer
# ==================================================

if __name__ == "__main__":

    result, probability = predict_churn(
        credit_score=650,
        geography="France",
        gender="Female",
        age=35,
        tenure=5,
        balance=60000,
        num_of_products=2,
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=50000
    )

    print("\nCustomer Churn Prediction:")
    print(result)
    print(f"Churn Probability: {probability:.2%}")
