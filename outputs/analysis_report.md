# Customer Churn Analysis Report

## Executive summary

The public IBM Telco sample contains **7,043 customer records**. **1,869 are labelled churned**, giving an overall sample churn rate of **26.5%**. The logistic regression model achieved **80.6% accuracy**, **65.7% precision**, **55.9% recall**, and **ROC-AUC 0.842** on a stratified 20% test set held out from fitting (random state 42; 5,634 training / 1,409 test records). This is a retrospective classification benchmark, not a validated forecast of next-month churn.

## Main findings

- Month-to-month customers have the highest contract churn rate at **42.7%**, compared with **2.8%** for the lowest-risk contract category.
- **Fiber optic** customers show the highest internet-service churn rate at **41.9%**.
- **Electronic check** has the highest payment-method churn rate at **45.3%**.
- Churned customers have an average tenure of **18.0 months**, versus **37.6 months** for retained customers.
- Churned customers pay **USD 74.44** per month on average, versus **USD 61.27** for retained customers.

## Model evaluation

| Metric | Score |
|---|---:|
| Accuracy | 0.806 |
| Precision (churn) | 0.657 |
| Recall (churn) | 0.559 |
| F1 (churn) | 0.604 |
| ROC-AUC | 0.842 |

Precision, recall, F1, and the confusion matrix use the default 0.5 decision threshold. Predicting every test customer as staying would achieve 73.5% accuracy but zero churn recall. ROC-AUC measures ranking, not accuracy or calibrated individual risk.

Confusion matrix (rows are actual classes; columns are predicted classes):

| | Predicted stay | Predicted churn |
|---|---:|---:|
| Actual stay | 926 | 109 |
| Actual churn | 165 | 209 |

## Strong model indicators

Positive coefficients increase model log-odds with other encoded inputs fixed; negative coefficients reduce them. These are regularized conditional associations, not causal effects or a general feature-importance ranking. Numeric coefficients use standardized units. Multi-category variables retain all categories, so individual coefficients are not conventional odds ratios against an omitted reference category. Correlated tenure, monthly charges, and total charges make isolated coefficient interpretations fragile.

| feature                                           |   coefficient |
|:--------------------------------------------------|--------------:|
| categorical__InternetService_Fiber optic          |         0.594 |
| categorical__Contract_Month-to-month              |         0.529 |
| numeric__TotalCharges                             |         0.520 |
| categorical__PaperlessBilling_Yes                 |         0.370 |
| categorical__PaymentMethod_Electronic check       |         0.162 |
| categorical__StreamingTV_Yes                      |         0.157 |
| categorical__StreamingMovies_Yes                  |         0.157 |
| categorical__SeniorCitizen_Yes                    |         0.146 |
| numeric__tenure                                   |        -1.244 |
| categorical__Contract_Two year                    |        -0.815 |
| categorical__InternetService_DSL                  |        -0.701 |
| numeric__MonthlyCharges                           |        -0.600 |
| categorical__OnlineSecurity_No internet service   |        -0.349 |
| categorical__StreamingTV_No internet service      |        -0.349 |
| categorical__DeviceProtection_No internet service |        -0.349 |
| categorical__InternetService_No                   |        -0.349 |

## Business recommendations

1. **Investigate shorter-tenure and month-to-month segments.** Consider testing onboarding support and proactive check-ins. The descriptive tenure comparison is cross-sectional, not a measured first-year churn hazard or cohort retention curve.
2. **Encourage longer commitments.** Test loyalty discounts or service credits that make one- and two-year contracts attractive without eroding margin.
3. **Review high-risk internet experiences.** Investigate service quality, pricing, and support journeys for the internet category with the highest churn.
4. **Explore support and security needs.** Consider testing relevant support offers; service associations do not establish that adding a bundle prevents churn.
5. **Explore score-based outreach.** The score adds a multivariable ranking to segment summaries. Before operational use, define a future churn horizon, validate prospectively, check calibration, choose thresholds using offer costs and capacity, and obtain customer-value data. Lifetime value and treatment benefit are not measured here.
6. **Test intervention results.** Randomly assign eligible customers to a defined retention action or control group before outreach. Predefine the follow-up horizon, churn/retention outcome, sample size, intention-to-treat comparison, margin/cost guardrails, and uncertainty intervals. Incremental retention and campaign ROI have not been measured in this project.

## Limitations

- The dataset is a public demonstration snapshot, not this student's employer/customer data. It does not include dated interactions, complaints, service outages, or retention offers.
- Feature measurement time relative to churn is not established here; a retrospective row split cannot prove prospective availability or exclude real-world outcome-time leakage.
- Full-data descriptive analysis includes test rows. The test set is held out from fitting, not an independently untouched future/temporal validation set. No hyperparameter search is implemented.
- Distinct customer IDs can have identical predictor profiles. The audited split has 13 test rows matching training profiles; grouped/temporal validation is a future robustness check.
- Scores are not demonstrated to be calibrated probabilities. The app's 0.4/0.7 risk bands are illustrative, not optimized campaign thresholds.
- Associations and retention suggestions do not establish causation or proven retention uplift.
- Model performance should be revalidated on current company data before deployment.
- Logistic regression assumes additive linear effects on log-odds, not on churn probability, and may miss nonlinear relationships.
