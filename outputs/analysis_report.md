# Customer Churn Analysis Report

## Executive summary

The dataset contains **7,043 customers**. **1,869 customers churned**, giving an overall churn rate of **26.5%**. The logistic regression model achieved **80.6% accuracy**, **65.7% precision**, **55.9% recall**, and **84.2% ROC-AUC** on an unseen 20% test set.

## Main findings

- Month-to-month customers have the highest contract churn rate at **42.7%**, compared with **2.8%** for the lowest-risk contract category.
- **Fiber optic** customers show the highest internet-service churn rate at **41.9%**.
- **Electronic check** has the highest payment-method churn rate at **45.3%**.
- Churned customers have an average tenure of **18.0 months**, versus **37.6 months** for retained customers.
- Churned customers pay **$74.44** per month on average, versus **$61.27** for retained customers.

## Model evaluation

| Metric | Score |
|---|---:|
| Accuracy | 0.806 |
| Precision (churn) | 0.657 |
| Recall (churn) | 0.559 |
| ROC-AUC | 0.842 |

Confusion matrix (rows are actual classes; columns are predicted classes):

| | Predicted stay | Predicted churn |
|---|---:|---:|
| Actual stay | 926 | 109 |
| Actual churn | 165 | 209 |

## Strong model indicators

Positive coefficients increase predicted churn probability; negative coefficients reduce it. Coefficients describe associations in this model and should not be interpreted as proof of causation.

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

1. **Prioritize new month-to-month customers.** Trigger onboarding support and proactive check-ins during the first year, when churn risk is highest.
2. **Encourage longer commitments.** Test loyalty discounts or service credits that make one- and two-year contracts attractive without eroding margin.
3. **Review high-risk internet experiences.** Investigate service quality, pricing, and support journeys for the internet category with the highest churn.
4. **Promote support and security services.** Offer relevant technical support and online-security bundles, especially to high-risk internet customers.
5. **Use probability-based outreach.** Rank active customers by predicted churn probability and focus retention resources on customers with both high risk and high lifetime value.
6. **Track intervention results.** Run controlled experiments and monitor recall, precision, retention lift, and campaign return on investment over time.

## Limitations

- The dataset is a historical snapshot and does not include interaction history, complaints, service outages, or retention offers.
- Model performance should be revalidated on current company data before deployment.
- Logistic regression is interpretable but may not capture every nonlinear customer behavior pattern.
