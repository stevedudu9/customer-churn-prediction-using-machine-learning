# Predicting Customer Churn with Logistic Regression

Customer churn prediction is a practical machine learning problem where the goal is to identify customers who are likely to stop using a company's service. For a subscription-based business, reducing churn can protect revenue and lower the cost of acquiring replacement customers.

This project uses the IBM Telco Customer Churn dataset, which contains demographic information, account details, service subscriptions, billing information, and a churn label.

## Problem framing

The task is a binary classification problem:

- `Churn = Yes`: the customer left.
- `Churn = No`: the customer stayed.

The model classifies a labelled demonstration snapshot. A prospective horizon and feature measurement times are not established, and calibration has not been demonstrated.

## Data preparation

The dataset required several cleaning and preprocessing steps:

1. `TotalCharges` was converted from text to numeric.
2. Blank `TotalCharges` values were filled with `0.0`.
3. Duplicate rows were checked.
4. `SeniorCitizen` was converted from `0/1` into `No/Yes` for easier interpretation.
5. The target variable was encoded as `ChurnFlag`, where churned customers are `1` and retained customers are `0`.

The model excluded `customerID` because it is an identifier, not a meaningful predictor.

## Exploratory analysis

The analysis found several churn patterns:

- Month-to-month contracts had much higher churn than longer-term contracts.
- Fiber optic internet customers showed higher churn.
- Electronic check customers had higher churn than other payment-method groups.
- Customers who churned had shorter average tenure.
- Churned customers had higher average monthly charges.

These associations suggest hypotheses to investigate and test, not proven business interventions.

## Model choice

Logistic Regression was selected because it is:

- Suitable for binary classification.
- Easy to explain.
- Fast to train.
- Useful for identifying factors associated with churn.

Logistic Regression estimates the log-odds of churn based on the input features. Positive coefficients increase the predicted churn probability, while negative coefficients reduce it.

## Preprocessing pipeline

The project uses a Scikit-learn pipeline:

- Numeric features are median-imputed and standardized.
- Categorical features are mode-imputed and one-hot encoded.
- The model is trained only after preprocessing is fitted on the training data.

This prevents learned preprocessing from using test-set statistics. Full-data EDA includes test rows, and prospective outcome-time leakage cannot be ruled out without timestamped feature definitions.

## Evaluation metrics

The model was evaluated using:

- Accuracy: overall percentage of correct predictions.
- Precision: of customers predicted to churn, how many actually churned.
- Recall: of customers who actually churned, how many the model found.
- ROC-AUC: how well the model separates churners from non-churners across thresholds.
- Confusion matrix: detailed count of correct and incorrect predictions.

The final model achieved:

- Accuracy: 80.6%
- Precision: 65.7%
- Recall: 55.9%
- ROC-AUC: 0.842 (ranking, not accuracy)

ROC-AUC 0.842 describes retrospective ranking on this split. At 0.5, recall is 55.9%; 165 of 374 labelled test churners are missed.

## Business recommendations

The model can support customer retention by helping the company:

1. Prioritize high-risk customers for outreach.
2. Focus retention campaigns on month-to-month customers.
3. Review service quality for high-risk internet service groups.
4. Encourage longer-term contracts using loyalty offers.
5. Track whether retention campaigns actually reduce churn.

## Limitations

The model uses Logistic Regression, which assumes a mostly linear relationship between features and churn log-odds. It also uses the default 0.5 decision threshold, which may not be the best threshold for a business retention campaign.

Future improvements could include:

- Threshold tuning.
- Class imbalance techniques.
- Additional customer behavior features.
- Comparing Random Forest, Gradient Boosting, or XGBoost.
- Measuring campaign return on investment.

## Conclusion

This project demonstrates a complete machine learning workflow from data cleaning and visualization to model training, evaluation, interpretation, and deployment. More importantly, it connects model results to business decisions, which is the key value of churn prediction.

## Audit qualifications (2026-10-02)

The test set is held out from fitting, not a future temporal holdout. Full-sample exploratory analysis includes test rows; 13 test profiles match training profiles across distinct IDs. ROC-AUC is ranking, not accuracy. Probabilities and 0.4/0.7 app risk bands are not validated calibration or business thresholds. Coefficients are regularized associations on log-odds, not causal effects; multi-category dummy encoding and correlated charges limit isolated odds interpretations. No cohort/funnel analysis, experiments, retention uplift, or customer lifetime value were measured. The current README and generated analysis report contain the detailed qualification. Local changes have not been published or deployed.
