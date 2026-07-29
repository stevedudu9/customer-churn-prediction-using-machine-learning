# Customer Churn Prediction

[![Live demo](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://customer-churn-prediction-using-machine-learning-8u58yeufi66lk.streamlit.app/) [![Repository](https://img.shields.io/badge/GitHub-Repository-181717?logo=github)](https://github.com/stevedudu9/customer-churn-prediction-using-machine-learning) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

End-to-end customer-churn analysis using the IBM Telco dataset. The project cleans data, explores churn patterns, trains an interpretable logistic-regression model, evaluates performance, and translates findings into retention recommendations.

## Live demo

[Open the prediction app](https://customer-churn-prediction-using-machine-learning-8u58yeufi66lk.streamlit.app/)

## Screenshots

![Churn distribution](outputs/figures/01_churn_distribution.png)

## Features

- Exploratory analysis of churn drivers
- Reproducible preprocessing and stratified evaluation
- Interpretable logistic-regression risk scoring
- Streamlit prediction interface and retention recommendations

## Technology stack

Python, Streamlit, pandas, scikit-learn, Matplotlib, seaborn, and logistic regression.

## Installation

```bash
git clone https://github.com/stevedudu9/customer-churn-prediction-using-machine-learning.git
cd customer-churn-prediction-using-machine-learning
python -m pip install -r requirements.txt
```

## Running locally

```bash
python src/churn_analysis.py
streamlit run app.py
```

## Project structure

```text
app.py                 Streamlit prediction application
src/churn_analysis.py  Analysis and modelling workflow
data/                  IBM Telco source data
outputs/               Metrics, figures, reports, and model artefacts
docs/                  Supporting materials
```

## Methodology

The workflow cleans `TotalCharges`, explores churn by customer and service characteristics, performs a stratified train/test split, encodes categorical inputs, and evaluates logistic regression with accuracy, precision, recall, ROC-AUC, and a confusion matrix.

## Validation and testing

Run the analysis script before the app so the reusable model pipeline and generated outputs are refreshed. Review the saved metrics, ROC curve, and confusion matrix.

## Limitations

The model reflects the IBM Telco dataset and a linear classification approach. Performance and recommendations should be revalidated for any new business context.

## Future improvements

- Add automated model-regression checks.
- Compare calibrated and non-linear baseline models.

## License

Distributed under the [MIT License](LICENSE).

---

Built by [Steve Dudu](https://github.com/stevedudu9) · [Portfolio](https://github.com/stevedudu9/data-portfolio)
