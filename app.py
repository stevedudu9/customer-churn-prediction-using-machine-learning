"""Streamlit app for customer churn prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st


ROOT = Path(__file__).resolve().parent
MODEL_PATH = ROOT / "outputs" / "model" / "logistic_regression_pipeline.joblib"
METRICS_PATH = ROOT / "outputs" / "model_metrics.json"
REPORT_PATH = ROOT / "outputs" / "analysis_report.md"


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_metrics() -> dict:
    with METRICS_PATH.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def build_customer_input() -> pd.DataFrame:
    st.sidebar.header("Customer profile")

    gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.sidebar.selectbox("Senior citizen", ["No", "Yes"])
    partner = st.sidebar.selectbox("Has partner", ["No", "Yes"])
    dependents = st.sidebar.selectbox("Has dependents", ["No", "Yes"])
    tenure = st.sidebar.slider("Tenure in months", 0, 72, 12)

    st.sidebar.header("Services")
    phone_service = st.sidebar.selectbox("Phone service", ["No", "Yes"])
    multiple_lines = st.sidebar.selectbox(
        "Multiple lines", ["No", "Yes", "No phone service"]
    )
    internet_service = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    online_security = st.sidebar.selectbox(
        "Online security", ["No", "Yes", "No internet service"]
    )
    online_backup = st.sidebar.selectbox(
        "Online backup", ["No", "Yes", "No internet service"]
    )
    device_protection = st.sidebar.selectbox(
        "Device protection", ["No", "Yes", "No internet service"]
    )
    tech_support = st.sidebar.selectbox(
        "Tech support", ["No", "Yes", "No internet service"]
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV", ["No", "Yes", "No internet service"]
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming movies", ["No", "Yes", "No internet service"]
    )

    st.sidebar.header("Account and billing")
    contract = st.sidebar.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.sidebar.selectbox("Paperless billing", ["No", "Yes"])
    payment_method = st.sidebar.selectbox(
        "Payment method",
        [
            "Electronic check",
            "Mailed check",
            "Bank transfer (automatic)",
            "Credit card (automatic)",
        ],
    )
    monthly_charges = st.sidebar.number_input("Monthly charges", 0.0, 200.0, 75.0, 1.0)
    total_charges = st.sidebar.number_input(
        "Total charges", 0.0, 10000.0, float(monthly_charges * max(tenure, 1)), 10.0
    )

    return pd.DataFrame(
        [
            {
                "gender": gender,
                "SeniorCitizen": senior_citizen,
                "Partner": partner,
                "Dependents": dependents,
                "tenure": tenure,
                "PhoneService": phone_service,
                "MultipleLines": multiple_lines,
                "InternetService": internet_service,
                "OnlineSecurity": online_security,
                "OnlineBackup": online_backup,
                "DeviceProtection": device_protection,
                "TechSupport": tech_support,
                "StreamingTV": streaming_tv,
                "StreamingMovies": streaming_movies,
                "Contract": contract,
                "PaperlessBilling": paperless_billing,
                "PaymentMethod": payment_method,
                "MonthlyCharges": monthly_charges,
                "TotalCharges": total_charges,
            }
        ]
    )


def risk_label(probability: float) -> tuple[str, str]:
    if probability >= 0.70:
        return "High churn risk", "red"
    if probability >= 0.40:
        return "Medium churn risk", "orange"
    return "Low churn risk", "green"


def main() -> None:
    st.set_page_config(page_title="Customer Churn Predictor", page_icon="📉", layout="wide")
    st.title("Customer Churn Prediction")
    st.write(
        "This app uses the trained Logistic Regression pipeline to estimate whether a "
        "telecom customer is likely to churn."
    )

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python src/churn_analysis.py` first.")
        st.stop()

    model = load_model()
    metrics = load_metrics()
    customer = build_customer_input()

    probability = float(model.predict_proba(customer)[0, 1])
    prediction = "Yes" if probability >= 0.5 else "No"
    label, color = risk_label(probability)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted churn", prediction)
    col2.metric("Churn probability", f"{probability:.1%}")
    col3.metric("Risk level", label)
    col4.metric("Model ROC-AUC", f"{metrics['model']['roc_auc']:.1%}")

    st.progress(probability)
    st.markdown(f":{color}[{label}]")

    st.subheader("Recommended retention action")
    if probability >= 0.70:
        st.write(
            "Prioritize this customer for retention outreach, review service quality, "
            "and consider a targeted loyalty or contract offer."
        )
    elif probability >= 0.40:
        st.write(
            "Monitor this customer and consider proactive support, onboarding help, "
            "or a value-focused offer."
        )
    else:
        st.write("This customer appears lower risk. Continue normal engagement and service quality monitoring.")

    st.subheader("Input customer profile")
    st.dataframe(customer, use_container_width=True)

    with st.expander("Model performance from the test set"):
        st.json(metrics["model"])

    if REPORT_PATH.exists():
        with st.expander("Project report summary"):
            st.markdown(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
