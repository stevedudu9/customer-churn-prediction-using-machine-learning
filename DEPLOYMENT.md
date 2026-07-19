# Deployment Guide

This project can be deployed as an interactive Streamlit web app.

## Local app run

From the project folder:

```bash
python -m pip install -r requirements.txt
python src/churn_analysis.py
streamlit run app.py
```

The analysis script creates the trained model under `outputs/model/`. The Streamlit app loads that saved model and predicts churn risk from user-entered customer details.

## Deploy on Streamlit Community Cloud

1. Push this project folder to GitHub.
2. Go to Streamlit Community Cloud.
3. Choose **New app**.
4. Select the GitHub repository.
5. Set the main file path to:

```text
app.py
```

6. Deploy the app.

## Deployment checklist

- `app.py` exists in the project root.
- `requirements.txt` includes Streamlit and the machine learning libraries.
- `outputs/model/logistic_regression_pipeline.joblib` exists.
- `outputs/model_metrics.json` exists.
- The app can run locally before deploying.

## Suggested app description

An interactive customer churn prediction app built with Python, Pandas, Scikit-learn, and Streamlit. The app predicts churn probability for telecom customers and provides retention recommendations based on model output.
