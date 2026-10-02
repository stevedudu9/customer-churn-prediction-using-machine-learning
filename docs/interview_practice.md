# Interview Practice Notes

Use these answers to explain the project clearly in interviews.

## 1. What problem does this project solve?

This project predicts whether a telecom customer is likely to churn. The potential use is prioritization, but this snapshot does not demonstrate early warning or successful interventions. Timestamped prospective validation is needed.

## 2. Why did you choose Logistic Regression?

I chose Logistic Regression because it is a strong baseline model for binary classification. It is also interpretable, which is useful for business analytics because I can explain which features increase or decrease churn risk.

## 3. How did you avoid data leakage?

I split the data into training and test sets before fitting preprocessing steps. Numeric scaling, imputation, and categorical encoding are inside a Scikit-learn pipeline, so they are fitted only on the training data and then applied to the test data.

## 4. What metrics did you use?

I used accuracy, precision, recall, ROC-AUC, and the confusion matrix. For churn prediction, recall is especially important because missing high-risk customers can mean losing revenue.

## 5. What were the final results?

The model achieved about 80.6% accuracy, 65.7% precision, 55.9% recall, and ROC-AUC 0.842 on the test set.

## 6. What does ROC-AUC mean here?

ROC-AUC measures how well the model separates churned customers from retained customers across different probability thresholds. A score of 0.842 estimates the chance that a randomly chosen labelled churner receives a higher score than a labelled non-churner, with ties receiving half credit. It is not accuracy or evidence of calibration.

## 7. What did you learn from the analysis?

Month-to-month contracts, shorter tenure, higher monthly charges, fiber optic internet, and electronic check payments were associated with higher churn risk.

## 8. What are the model limitations?

The model uses Logistic Regression, so it may not capture complex nonlinear relationships. It also uses a default 0.5 threshold, which may not be ideal for a retention campaign. Recall could be improved because the model still misses some churn customers.

## 9. How would you improve the project?

I would tune the classification threshold, try class imbalance techniques, engineer more customer behavior features, and compare models such as Random Forest and Gradient Boosting.

## 10. How would the business use this model?

After prospective validation, the business could explore score-based prioritization. Customer lifetime value and treatment responsiveness are not estimated in this project. The company should also track whether outreach improves retention compared with a control group.

## 11. How would you explain false positives and false negatives?

A false positive is a customer predicted to churn who actually stays. This may waste retention resources. A false negative is a customer predicted to stay who actually churns. This is often more costly because the company loses the customer without taking action.

## 12. What part of the project are you most proud of?

I am proud that the project does not stop at model accuracy. It includes visual insights, model interpretation, business recommendations, a deployment-ready app, and a discussion of limitations.

## Audit qualifications (2026-10-02)

The test set is held out from fitting, not a future temporal holdout. Full-sample exploratory analysis includes test rows; 13 test profiles match training profiles across distinct IDs. ROC-AUC is ranking, not accuracy. Probabilities and 0.4/0.7 app risk bands are not validated calibration or business thresholds. Coefficients are regularized associations on log-odds, not causal effects; multi-category dummy encoding and correlated charges limit isolated odds interpretations. No cohort/funnel analysis, experiments, retention uplift, or customer lifetime value were measured. The current README and generated analysis report contain the detailed qualification. Local changes have not been published or deployed.
