import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def load_model():
    return joblib.load("best_model.pkl")

best_model = load_model()

st.title("UPI Fraud Detection")

st.write("Fill in the transaction details to predict whether it's fraudulent or not.")

step = st.number_input("Step (time step of transaction)", min_value=0, format="%d")
amount = st.number_input("Transaction Amount", min_value=0.0)
oldbalanceOrg = st.number_input("Sender Balance Before Transaction", min_value=0.0)
newbalanceOrig = st.number_input("Sender Balance After Transaction", min_value=0.0)
oldbalanceDest = st.number_input("Receiver Balance Before Transaction", min_value=0.0)
newbalanceDest = st.number_input("Receiver Balance After Transaction", min_value=0.0)

transaction_type = st.selectbox(
    "Transaction Type",
    ["CASH_OUT", "DEBIT", "PAYMENT", "TRANSFER"]
)

type_CASH_OUT = 1 if transaction_type == "CASH_OUT" else 0
type_DEBIT = 1 if transaction_type == "DEBIT" else 0
type_PAYMENT = 1 if transaction_type == "PAYMENT" else 0
type_TRANSFER = 1 if transaction_type == "TRANSFER" else 0

if st.button("Predict Fraud"):
    input_features = np.array([[step, amount, oldbalanceOrg, newbalanceOrig, oldbalanceDest, newbalanceDest, type_CASH_OUT, type_DEBIT, type_PAYMENT, type_TRANSFER]])
    prediction = best_model.predict(input_features)[0]
    proba = best_model.predict_proba(input_features)[0]
    st.markdown("---")
    if prediction == 1:
        st.error(f"Prediction: **Fraudulent Transaction**")
    else:
        st.success(f"Prediction: **Not Fraudulent**")

    st.write(f"**Confidence:** {proba[1]*100:.2f}% Fraud, {proba[0]*100:.2f}% Not Fraud")
