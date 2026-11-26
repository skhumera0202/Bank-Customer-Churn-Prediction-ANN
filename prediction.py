
import numpy as np
import pandas as pd
import tensorflow as tf
import joblib


# ==========================================
# 1. LOAD MODEL AND PREPROCESSING OBJECTS
# ==========================================

model = tf.keras.models.load_model(
    "ann_model.h5",
    compile=False
)

scaler = joblib.load("scaler.pkl")
encoder = joblib.load("one_hot_encoder.pkl")


# ==========================================
# 2. CUSTOMER CHURN PREDICTION FUNCTION
# ==========================================

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

    # Convert gender into numeric format
    if isinstance(gender, str):
        gender = gender.strip().lower()

        if gender == "male":
            gender = 1
        elif gender == "female":
            gender = 0
        else:
            raise ValueError("Gender must be Male or Female.")

    # Create customer input dataframe
    input_data = pd.DataFrame([{
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

    # ======================================
    # 3. ENCODE GEOGRAPHY
    # ======================================

    encoded_data = encoder.transform(
        input_data[["Geography"]]
    )

    encoded_columns = encoder.get_feature_names_out(
        ["Geography"]
    )

    encoded_df = pd.DataFrame(
        encoded_data,
        columns=encoded_columns,
        index=input_data.index
    )

    # Remove original Geography column
    input_data = input_data.drop(
        columns=["Geography"]
    )

    # ======================================
    # 4. COMBINE AND ORDER FEATURES
    # ======================================

    final_input = pd.concat(
        [input_data, encoded_df],
        axis=1
    )

    final_input = final_input[
        scaler.feature_names_in_
    ]

    # ======================================
    # 5. SCALE INPUT
    # ======================================

    scaled_input = scaler.transform(final_input)

    # ======================================
    # 6. ANN PREDICTION
    # ======================================

    probability = float(
        model.predict(
            scaled_input,
            verbose=0
        )[0][0]
    )

    prediction = 1 if probability >= 0.5 else 0

    if prediction == 1:
        result = "Customer is likely to leave the bank."
    else:
        result = "Customer is likely to stay with the bank."

    return result, probability


# ==========================================
# 7. SAMPLE PREDICTION
# ==========================================

if __name__ == "__main__":

    result, probability = predict_churn(
        credit_score=600,
        geography="France",
        gender="Male",
        age=40,
        tenure=3,
        balance=60000,
        num_of_products=2,
        has_cr_card=1,
        is_active_member=1,
        estimated_salary=50000
    )

    print("\nCustomer Churn Prediction:")
    print(result)
    print(f"Churn Probability: {probability:.2%}")
