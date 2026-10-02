# Customer Churn Prediction / Retention Analytics

Which customer patterns are associated with churn, and where should a business investigate retention opportunities? This student project combines descriptive customer analysis and interpretable logistic regression on a public IBM Telco sample. It recommends priorities to test; it does not demonstrate causal effects or production impact.

## Dataset and key findings

The checked-in `data/WA_Fn-UseC_-Telco-Customer-Churn.csv` contains 7,043 customer records and 21 original columns. [IBM's original example](https://github.com/IBM/customer-churn-prediction) identifies this filename as its sample dataset. It is demonstration data, not customer data collected by this student. The snapshot has no explicit future prediction horizon or dated product events.

| Comparison | Calculation | Result |
|---|---|---:|
| Overall sample churn | 1,869 / 7,043 | 26.5% |
| Month-to-month churn | 1,655 / 3,875 | 42.7% |
| One-year contract churn | 166 / 1,473 | 11.3% |
| Two-year contract churn | 48 / 1,695 | 2.8% |
| Mean tenure, churned | 33,603 / 1,869 months | 18.0 months |
| Mean tenure, stayed | 194,387 / 5,174 months | 37.6 months |

![Observed churn by contract](outputs/figures/02_churn_by_contract.png)

These are full-sample associations. Contract choice can reflect customer differences, and churn can truncate tenure. These figures do not establish that longer contracts prevent churn. Tenure segments are not acquisition cohorts, a funnel, or a survival analysis.

## Analytical approach and model evaluation

1. Convert `TotalCharges` to numeric. All 11 blank values have zero tenure and are assigned zero total charges under that explicit assumption. Missing charges outside zero-tenure records require review.
2. Check exact duplicates, unique IDs, and target encoding (`Yes=1`, `No=0`). Map `SeniorCitizen` to categorical Yes/No.
3. Exclude `customerID`, `Churn`, and `ChurnFlag`. Use the remaining 19 predictors.
4. Split rows 80/20 with `stratify=y`, `random_state=42`: 5,634 training / 1,409 test records. Test counts: 374 churned / 1,035 stayed.
5. Fit training-only preprocessing: numeric median imputation and standardization; categorical mode imputation and one-hot encoding (`drop='if_binary'`, unknowns ignored). Output: 40 encoded features.
6. Fit regularized logistic regression: lbfgs, C=1, max_iter=2000, no class weighting or resampling. No hyperparameter or threshold search is implemented.

| Metric | Test result |
|---|---:|
| ROC-AUC | **0.8419695678 → 0.842** |
| Accuracy | 0.8055358410 → 80.6% |
| Precision, churn | 0.6572327044 → 65.7% |
| Recall, churn | 0.5588235294 → 55.9% |
| F1, churn | 0.6040462428 → 0.604 |

Threshold-based metrics use 0.5. Confusion matrix: TN 926, FP 109, FN 165, TP 209. Always predicting stay gives 73.5% accuracy and zero churn recall. **ROC-AUC 0.842 is not 84.2% accuracy**; it measures ranking. Calibration has not been established.

Test rows are excluded from fitting. Full-data EDA includes them, so this is not independently untouched temporal validation. Customer IDs are distinct; 13 test records share identical predictor profiles with training records. Excluding these profiles as a diagnostic gives AUC 0.8428343700 on 1,396 rows; it does not replace the benchmark or create a new independent holdout. Measurement time relative to churn is unknown, so prospective outcome-time leakage cannot be ruled out from the CSV alone.

Coefficients are regularized conditional associations on log-odds. Numeric coefficients use standardized units. Multi-category variables retain all categories, so individual coefficients are not conventional odds ratios against an omitted reference. Correlated tenure/charges and redundant service variables limit isolated interpretations. Coefficients are not causal effects or absolute feature-importance rankings.

## Business implications

- Investigate onboarding/support needs in month-to-month and shorter-tenure segments.
- Explore service/pricing hypotheses without diagnosing service faults from churn labels alone.
- Use multivariable ranking only after prospective validation, calibration checks, and cost/capacity-based threshold selection.
- Test a defined action with randomized treatment/control assignment, a fixed horizon, intention-to-treat outcomes, uncertainty intervals, and cost/margin guardrails.

No intervention, incremental retention, revenue uplift, customer lifetime value, or campaign ROI was measured. Product-analytics relevance includes segmentation, retention questions, tenure, and KPI interpretation. Cohort/funnel analysis, behavioral events, and A/B testing are future extensions.

## Local run and validation

Tested with Python 3.12.14. Direct dependencies are pinned; the saved model uses scikit-learn 1.9.0. Cross-version model loading is [unsupported](https://scikit-learn.org/stable/model_persistence.html). Only load the trusted project artifact; rerun training when rebuilding it.

```bash
python -m venv .venv
# Activate .venv using your platform's command.
python -m pip install -r requirements-dev.txt
python src/churn_analysis.py
python -m pytest -q
python -m streamlit run app.py
```

Training regenerates outputs. Direct pins are not a full transitive lock or cross-platform bit-for-bit guarantee.

## Structure and technologies

```text
app.py                 Single-page Streamlit application
src/churn_analysis.py  Cleaning, plots, pipeline, evaluation, report
data/                 Original sample CSV
outputs/               Cleaned CSV, predictions, metrics, coefficients, charts, model
tests/                Reproduction, split-boundary, input and app tests
docs/                 Technical and presentation materials
```

Python, pandas, NumPy, scikit-learn, Matplotlib, Streamlit, joblib. No notebooks, SQL analysis, Power BI reports, production integration, or completed experiments are included.

## App, demo and limitations

The app shows customer findings first, then an illustrative scorer, segment counts, model metrics, and report. Risk bands 0.4/0.7 are illustrative, not optimized business thresholds. Service inputs enforce basic consistency. Scores do not establish a future churn probability.

[Existing demo](https://customer-churn-prediction-using-machine-learning-8u58yeufi66lk.streamlit.app/) · [Repository](https://github.com/stevedudu9/customer-churn-prediction-using-machine-learning) · [Portfolio](https://stevedudu9.github.io)

The public demo reflects the reviewed project version but may sleep when inactive. Revalidate on current business data before operational use, including feature timing, fairness, calibration, and temporal/grouped robustness. Demographic predictors require governance review before targeting.

## License

Project code: [MIT](LICENSE). Retain sample-data source attribution; the code license does not establish separate dataset rights.
