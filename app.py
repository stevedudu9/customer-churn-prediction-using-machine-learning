"""Streamlit app for customer churn prediction."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import streamlit as st
from src.churn_analysis import DATA_PATH, load_and_clean_data


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


@st.cache_data
def load_analysis_data() -> pd.DataFrame:
    return load_and_clean_data(DATA_PATH)[0]


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
        "Multiple lines", ["No", "Yes"] if phone_service == "Yes" else ["No phone service"]
    )
    internet_service = st.sidebar.selectbox("Internet service", ["DSL", "Fiber optic", "No"])
    internet_options = ["No", "Yes"] if internet_service != "No" else ["No internet service"]
    online_security = st.sidebar.selectbox(
        "Online security", internet_options
    )
    online_backup = st.sidebar.selectbox(
        "Online backup", internet_options
    )
    device_protection = st.sidebar.selectbox(
        "Device protection", internet_options
    )
    tech_support = st.sidebar.selectbox(
        "Tech support", internet_options
    )
    streaming_tv = st.sidebar.selectbox(
        "Streaming TV", internet_options
    )
    streaming_movies = st.sidebar.selectbox(
        "Streaming movies", internet_options
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
        "Total charges", 0.0, 15000.0, float(monthly_charges * tenure), 10.0
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
    # Long encoded feature names must wrap inside report tables on narrow screens.
    st.html("""<style>
    @media (max-width: 640px) {
        [data-testid="stMarkdownContainer"] th,
        [data-testid="stMarkdownContainer"] td { overflow-wrap: anywhere; }
    }
    </style>""")
    st.title("Customer Churn Prediction")
    st.write(
        "Which customer patterns are associated with churn, and where should retention "
        "efforts be investigated? This public IBM Telco sample combines descriptive "
        "analysis with an interpretable logistic-regression baseline."
    )
    st.caption("Portfolio demonstration · Retrospective customer snapshot · Recommendations require testing")

    if not MODEL_PATH.exists():
        st.error("Model file not found. Run `python src/churn_analysis.py` first.")
        st.stop()

    if not METRICS_PATH.exists() or not DATA_PATH.exists():
        st.error("Required data or metrics are missing. Run `python src/churn_analysis.py` first.")
        st.stop()
    try:
        model = load_model()
        metrics = load_metrics()
        analysis = load_analysis_data()
    except (OSError, ValueError, KeyError, ImportError, AttributeError) as exc:
        st.error(f"Project assets could not be loaded ({type(exc).__name__}). Install the pinned requirements and rerun the analysis.")
        st.stop()

    st.subheader("Customer patterns in the sample")
    overall = analysis["ChurnFlag"].mean()
    month = analysis.loc[analysis["Contract"] == "Month-to-month", "ChurnFlag"].mean()
    churn_tenure = analysis.loc[analysis["ChurnFlag"] == 1, "tenure"].mean()
    stay_tenure = analysis.loc[analysis["ChurnFlag"] == 0, "tenure"].mean()
    a, b, c = st.columns(3)
    a.metric("Overall sample churn", f"{overall:.1%}")
    b.metric("Month-to-month churn", f"{month:.1%}")
    c.metric("Mean tenure: churned / stayed", f"{churn_tenure:.1f} / {stay_tenure:.1f} months")
    contract = analysis.groupby("Contract")["ChurnFlag"].agg(["sum", "count", "mean"])
    contract_chart = contract[["mean"]].mul(100).rename(columns={"mean": "Observed churn (%)"})
    st.bar_chart(contract_chart, x_label="Contract type", y_label="Observed churn (%)")
    st.caption("Full-sample descriptive comparisons; contract and tenure associations do not prove causation. Tenure groups are not acquisition cohorts or a churn hazard curve.")
    st.write("Business interpretation: investigate onboarding and support needs in month-to-month and shorter-tenure segments, then test a defined retention action against a control group.")
    with st.expander("Segment counts and definitions"):
        st.dataframe(contract.rename(columns={"sum": "Churned", "count": "Customers", "mean": "Churn proportion"}))
        st.write("Churn = Yes is encoded as 1. Rates are labelled churners divided by all records in each segment, not annualized churn. Tenure is measured in months at the recorded snapshot.")

    st.subheader("Explore an illustrative customer score")
    customer = build_customer_input()
    if int(customer.iloc[0]["tenure"]) == 0 and float(customer.iloc[0]["TotalCharges"]) != 0:
        st.warning("A zero-tenure account should have zero total charges in this demonstration. Correct the billing inputs to score it.")
        st.stop()

    probability = float(model.predict_proba(customer)[0, 1])
    prediction = "Yes" if probability >= 0.5 else "No"
    label, color = risk_label(probability)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Predicted churn", prediction)
    col2.metric("Model score", f"{probability:.1%}")
    col3.metric("Risk level", label)
    col4.metric("Test ROC-AUC", f"{metrics['model']['roc_auc']:.3f}")
    st.caption("The score is the model's estimated class probability, without demonstrated calibration or a future churn horizon. Prediction uses 0.5; low/medium/high bands use 0.4/0.7 for illustration only. ROC-AUC is ranking performance, not accuracy.")

    st.progress(probability)
    st.markdown(f":{color}[{label}]")

    st.subheader("Retention hypothesis to consider")
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
    st.dataframe(customer, width="stretch")

    with st.expander("Model performance from the test set"):
        st.write("80/20 stratified row split; random state 42. Learned imputation, scaling, and encoding fit on training records only. This does not establish prospective performance or intervention impact.")
        st.json(metrics["model"])

    st.info("Limitations: public demonstration data; one retrospective random split; no calibration, longitudinal cohorts, experiments, or measured retention uplift. Validate feature timing and performance on current company data before operational use.")

    if REPORT_PATH.exists():
        with st.expander("Project report summary"):
            st.markdown(REPORT_PATH.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
