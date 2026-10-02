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

An interactive customer churn prediction app built with Python, Pandas, Scikit-learn, and Streamlit. The app explores descriptive churn patterns and an illustrative retrospective model score. Recommendations require testing; this is not a validated production prediction service.

## Audit qualifications (2026-10-02)

The test set is held out from fitting, not a future temporal holdout. Full-sample exploratory analysis includes test rows; 13 test profiles match training profiles across distinct IDs. ROC-AUC is ranking, not accuracy. Probabilities and 0.4/0.7 app risk bands are not validated calibration or business thresholds. Coefficients are regularized associations on log-odds, not causal effects; multi-category dummy encoding and correlated charges limit isolated odds interpretations. No cohort/funnel analysis, experiments, retention uplift, or customer lifetime value were measured. The current README and generated analysis report contain the detailed qualification. Local changes have not been published or deployed.
