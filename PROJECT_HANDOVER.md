# Project Handover: Customer Churn Prediction

## What this project is

This is an end-to-end machine-learning project that predicts whether a telecommunications customer is likely to leave the company ("churn"). It uses the IBM Telco Customer Churn dataset, which contains customer demographics, account information, subscribed services, monthly charges, total charges, tenure, and the churn outcome.

The business purpose is to identify customers at higher risk of leaving so the company can offer timely, useful retention support.

## What is included

- Data cleaning and preparation.
- Exploratory data analysis (EDA) and churn visualizations.
- A Logistic Regression churn-prediction model.
- Model evaluation using accuracy, precision, recall, ROC-AUC, and a confusion matrix.
- Customer-level churn probabilities and risk scores.
- Business recommendations for retention.
- An interactive Streamlit prediction app.
- Documentation for deployment, a demo video, a blog post, and interview practice.

## What we completed in this chat

1. Extracted and organized the IBM Telco Customer Churn dataset in `data/`.
2. Built the complete Python analysis and modelling workflow in `src/churn_analysis.py`.
3. Cleaned the data by converting `TotalCharges` to a numeric value and handling blank or missing values.
4. Created churn charts for distribution, contract type, internet service, payment method, tenure, numeric distributions, model evaluation, and the most influential features.
5. Built a reproducible preprocessing-and-model pipeline that:
   - removes the non-predictive `customerID` column;
   - uses an 80/20 stratified train/test split;
   - median-imputes and scales numeric features;
   - mode-imputes and one-hot encodes categorical features;
   - trains Logistic Regression with a fixed random seed.
6. Saved the trained pipeline, evaluation metrics, feature coefficients, cleaned dataset, test predictions, charts, and written analysis report under `outputs/`.
7. Added the required README charts so a lecturer or recruiter can see key findings immediately without running the project.
8. Added a realistic **Model Limitations** section to the README.
9. Built `app.py`, an interactive Streamlit app where a user enters a customer profile and receives a prediction, probability, risk level, and retention suggestion.
10. Added deployment instructions and presentation materials in `DEPLOYMENT.md` and `docs/`.
11. Re-ran and audited the analysis, including syntax checks, model loading, saved-prediction checks, probability-range checks, and a sample app-style prediction.
12. Kept a complete copy of the project together with the original dataset folder.
13. Created and published the public GitHub repository:
    - https://github.com/stevedudu9/customer-churn-prediction-using-machine-learning
14. Confirmed that the Streamlit deployment is reachable:
    - https://customer-churn-prediction-using-machine-learning-8u58yeufi66lk.streamlit.app/

## Latest model output

The dataset contains 7,043 records; the model was fitted on 5,634 training records and evaluated on 1,409 held-out-from-fitting test records.

| Measurement | Result |
|---|---:|
| Churn rate in dataset | 26.5% |
| Training records | 5,634 |
| Test records | 1,409 |
| Accuracy | 80.6% |
| Precision | 65.7% |
| Recall | 55.9% |
| ROC-AUC | 0.842 (ranking, not accuracy) |
| Confusion matrix | TN 926, FP 109, FN 165, TP 209 |

The ROC-AUC of 0.842 shows the model distinguishes reasonably well between customers who churn and customers who stay. The recall of 55.9% means it identifies over half of the customers who actually churn at the default 0.5 threshold. This is a solid baseline model, but not a claim of perfection; threshold tuning and comparison with other models can improve the retention-campaign trade-off.

## Main churn insights

The results show that churn is especially connected to contract type, tenure, payment method, internet-service choices, and monthly charges. In general, month-to-month customers and newer customers show higher churn risk, while longer-term contracts are associated with lower churn risk.

## Important files

| File or folder | Purpose |
|---|---|
| `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` | Original dataset used in the project |
| `src/churn_analysis.py` | Full cleaning, analysis, training, and evaluation workflow |
| `app.py` | Streamlit prediction application |
| `outputs/analysis_report.md` | Written findings and business recommendations |
| `outputs/model/logistic_regression_pipeline.joblib` | Saved trained model pipeline |
| `outputs/model_metrics.json` | Saved performance metrics |
| `outputs/test_customer_predictions.csv` | Model predictions for the held-out test set |
| `outputs/figures/` | All generated charts |
| `README.md` | Public project overview with embedded visualizations |
| `DEPLOYMENT.md` | Streamlit deployment guide |
| `docs/` | Demo script, blog post, and interview practice material |

## How to run it again

```bash
python -m pip install -r requirements.txt
python src/churn_analysis.py
streamlit run app.py
```

Running the analysis script regenerates the outputs. Then the Streamlit command opens the interactive prediction application.

## Model limitations and next improvements

Logistic Regression is a good, interpretable baseline, but it assumes additive linear relationships between encoded predictors and churn log-odds. The default 0.5 threshold was not optimized for business costs and misses 165 of 374 test churners.

Potential next steps are threshold tuning based on the cost of retention offers, class-imbalance techniques, more feature engineering, and comparison against models such as Random Forest or Gradient Boosting. Any model should also be validated on recent real company data before business use.

## Audit qualifications (2026-10-02)

The test set is held out from fitting, not a future temporal holdout. Full-sample exploratory analysis includes test rows; 13 test profiles match training profiles across distinct IDs. ROC-AUC is ranking, not accuracy. Probabilities and 0.4/0.7 app risk bands are not validated calibration or business thresholds. Coefficients are regularized associations on log-odds, not causal effects; multi-category dummy encoding and correlated charges limit isolated odds interpretations. No cohort/funnel analysis, experiments, retention uplift, or customer lifetime value were measured. The current README and generated analysis report contain the detailed qualification. Local changes have not been published or deployed.
