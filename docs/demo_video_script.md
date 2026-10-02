# Short Demo Video Script

## Video length

Aim for 60 to 90 seconds.

## Scene 1: Project introduction

**Show:** README title and visualizations.

**Say:**  
"This is my Customer Churn Prediction project using the IBM Telco Customer Churn dataset. The goal is to understand churn associations and suggest retention priorities to test; this is a retrospective sample-data demonstration."

## Scene 2: Dataset and workflow

**Show:** Project workflow or repository structure.

**Say:**  
"The project includes data cleaning, exploratory analysis, visualizations, encoding and scaling, model training, evaluation, and business recommendations."

## Scene 3: Key insights

**Show:** Churn distribution and contract churn chart.

**Say:**  
"The analysis shows that churn is especially high among month-to-month customers. The project also compares churn across internet service, payment method, tenure, and monthly charges."

## Scene 4: Model performance

**Show:** Metrics or analysis report.

**Say:**  
"I used Logistic Regression because it is interpretable and suitable for binary classification. The model achieved about 80.6% accuracy and ROC-AUC 0.842 on the test set."

## Scene 5: App demo

**Show:** Streamlit app.

**Say:**  
"I also created a simple app where a user can enter a customer profile and receive an illustrative model score, predicted label, risk band, and retention hypothesis; the score is not demonstrated to be calibrated."

## Scene 6: Limitations and improvements

**Show:** Model limitations section.

**Say:**  
"The model is not perfect. It uses a default 0.5 threshold and misses some churn customers. Future improvements could include threshold tuning, imbalance handling, feature engineering, and comparing models like Random Forest or Gradient Boosting."

## Scene 7: Closing

**Show:** README or app result screen.

**Say:**  
"This project demonstrates end-to-end machine learning, business analysis, visualization, model evaluation, and a local interactive demonstration. It does not establish production or intervention impact."

## Audit qualifications (2026-10-02)

The test set is held out from fitting, not a future temporal holdout. Full-sample exploratory analysis includes test rows; 13 test profiles match training profiles across distinct IDs. ROC-AUC is ranking, not accuracy. Probabilities and 0.4/0.7 app risk bands are not validated calibration or business thresholds. Coefficients are regularized associations on log-odds, not causal effects; multi-category dummy encoding and correlated charges limit isolated odds interpretations. No cohort/funnel analysis, experiments, retention uplift, or customer lifetime value were measured. The current README and generated analysis report contain the detailed qualification. Local changes have not been published or deployed.
