import pickle

import numpy as np
import pandas as pd
import streamlit as st
from keras.models import load_model

st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="centered")

MODEL_PATH = "saved_model/model.keras"
PREPROCESSOR_PATH = "saved_model/preprocessor.pkl"


@st.cache_resource
def load_artifacts():
    model = load_model(MODEL_PATH)
    with open(PREPROCESSOR_PATH, "rb") as f:
        preprocessor = pickle.load(f)
    return model, preprocessor


model, preprocessor = load_artifacts()

st.title("📉 Customer Churn Predictor")
st.write("Enter a customer's details to predict whether they are likely to churn.")

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        credit_score = st.number_input("Credit Score", min_value=300, max_value=900, value=650)
        geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
        gender = st.selectbox("Gender", ["Male", "Female"])
        age = st.number_input("Age", min_value=18, max_value=100, value=35)
        tenure = st.number_input("Tenure (years with bank)", min_value=0, max_value=10, value=3)

    with col2:
        balance = st.number_input("Account Balance", min_value=0.0, value=50000.0, step=1000.0)
        num_products = st.number_input("Number of Products", min_value=1, max_value=4, value=1)
        has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
        is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])
        estimated_salary = st.number_input("Estimated Salary", min_value=0.0, value=60000.0, step=1000.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    input_df = pd.DataFrame(
        [
            {
                "CreditScore": credit_score,
                "Geography": geography,
                "Gender": gender,
                "Age": age,
                "Tenure": tenure,
                "Balance": balance,
                "NumOfProducts": num_products,
                "HasCrCard": 1 if has_cr_card == "Yes" else 0,
                "IsActiveMember": 1 if is_active_member == "Yes" else 0,
                "EstimatedSalary": estimated_salary,
            }
        ]
    )

    X_transformed = preprocessor.transform(input_df)
    probability = float(model.predict(X_transformed, verbose=0)[0][0])
    prediction = "Churn" if probability > 0.5 else "Stay"

    st.subheader("Result")
    if prediction == "Churn":
        st.error(f"⚠️ Likely to churn — probability: {probability:.2%}")
    else:
        st.success(f"✅ Likely to stay — churn probability: {probability:.2%}")

    st.progress(min(max(probability, 0.0), 1.0))
