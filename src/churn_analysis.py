"""End-to-end IBM Telco customer churn analysis and prediction workflow."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    roc_auc_score,
    RocCurveDisplay,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
OUTPUT_DIR = ROOT / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = OUTPUT_DIR / "model"
RANDOM_STATE = 42


def prepare_directories() -> None:
    FIGURE_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)


def load_and_clean_data(path: Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    original_shape = df.shape
    duplicate_count = int(df.duplicated().sum())

    df.columns = df.columns.str.strip()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_total_charges = int(df["TotalCharges"].isna().sum())
    df["TotalCharges"] = df["TotalCharges"].fillna(0.0)
    df = df.drop_duplicates().copy()
    df["SeniorCitizen"] = df["SeniorCitizen"].map({0: "No", 1: "Yes"})
    df["ChurnFlag"] = df["Churn"].map({"No": 0, "Yes": 1})

    quality = {
        "original_rows": int(original_shape[0]),
        "original_columns": int(original_shape[1]),
        "duplicate_rows_removed": duplicate_count,
        "blank_total_charges_filled_with_zero": missing_total_charges,
        "clean_rows": int(len(df)),
    }
    return df, quality


def save_current_figure(filename: str) -> None:
    plt.tight_layout()
    plt.savefig(FIGURE_DIR / filename, dpi=180, bbox_inches="tight")
    plt.close()


def create_visualizations(df: pd.DataFrame) -> None:
    colors = ["#4472C4", "#ED7D31"]

    churn_counts = df["Churn"].value_counts().reindex(["No", "Yes"])
    ax = churn_counts.plot(kind="bar", color=colors, rot=0, figsize=(7, 4.5))
    ax.set(title="Customer Churn Distribution", xlabel="Churn", ylabel="Customers")
    for container in ax.containers:
        ax.bar_label(container, fmt="%d")
    save_current_figure("01_churn_distribution.png")

    contract_rate = (
        df.groupby("Contract", observed=False)["ChurnFlag"].mean().mul(100).sort_values()
    )
    ax = contract_rate.plot(kind="barh", color="#ED7D31", figsize=(7, 4.5))
    ax.set(title="Churn Rate by Contract Type", xlabel="Churn rate (%)", ylabel="")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    save_current_figure("02_churn_by_contract.png")

    internet_rate = (
        df.groupby("InternetService", observed=False)["ChurnFlag"].mean().mul(100).sort_values()
    )
    ax = internet_rate.plot(kind="barh", color="#5B9BD5", figsize=(7, 4.5))
    ax.set(title="Churn Rate by Internet Service", xlabel="Churn rate (%)", ylabel="")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    save_current_figure("03_churn_by_internet_service.png")

    payment_rate = (
        df.groupby("PaymentMethod", observed=False)["ChurnFlag"].mean().mul(100).sort_values()
    )
    ax = payment_rate.plot(kind="barh", color="#70AD47", figsize=(8, 5))
    ax.set(title="Churn Rate by Payment Method", xlabel="Churn rate (%)", ylabel="")
    ax.bar_label(ax.containers[0], fmt="%.1f%%", padding=3)
    save_current_figure("04_churn_by_payment_method.png")

    tenure_bins = [-1, 12, 24, 48, 60, np.inf]
    tenure_labels = ["0-12", "13-24", "25-48", "49-60", "61-72"]
    tenure_group = pd.cut(df["tenure"], bins=tenure_bins, labels=tenure_labels)
    tenure_rate = df.assign(TenureGroup=tenure_group).groupby(
        "TenureGroup", observed=False
    )["ChurnFlag"].mean().mul(100)
    ax = tenure_rate.plot(kind="bar", color="#A5A5A5", rot=0, figsize=(7, 4.5))
    ax.set(title="Churn Rate by Tenure", xlabel="Tenure (months)", ylabel="Churn rate (%)")
    ax.bar_label(ax.containers[0], fmt="%.1f%%")
    save_current_figure("05_churn_by_tenure.png")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for label, color in zip(["No", "Yes"], colors):
        subset = df.loc[df["Churn"] == label]
        axes[0].hist(subset["MonthlyCharges"], bins=25, alpha=0.55, label=label, color=color)
        axes[1].hist(subset["tenure"], bins=24, alpha=0.55, label=label, color=color)
    axes[0].set(title="Monthly Charges by Churn", xlabel="Monthly charges", ylabel="Customers")
    axes[1].set(title="Tenure by Churn", xlabel="Tenure (months)", ylabel="Customers")
    for ax in axes:
        ax.legend(title="Churn")
    save_current_figure("06_numeric_distributions.png")


def build_model(df: pd.DataFrame) -> tuple[Pipeline, dict, pd.DataFrame, np.ndarray, np.ndarray]:
    X = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
    y = df["ChurnFlag"]

    numeric_features = X.select_dtypes(include=np.number).columns.tolist()
    categorical_features = X.select_dtypes(exclude=np.number).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="if_binary")),
        ]
    )
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=2000, random_state=RANDOM_STATE)),
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    cm = confusion_matrix(y_test, predictions)
    metrics = {
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions)),
        "recall": float(recall_score(y_test, predictions)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "confusion_matrix": cm.tolist(),
        "classification_report": classification_report(
            y_test, predictions, target_names=["Stayed", "Churned"], output_dict=True
        ),
    }

    prediction_output = X_test.copy()
    prediction_output.insert(0, "customerID", df.loc[X_test.index, "customerID"])
    prediction_output["actual_churn"] = y_test.map({0: "No", 1: "Yes"})
    prediction_output["predicted_churn"] = pd.Series(predictions, index=X_test.index).map(
        {0: "No", 1: "Yes"}
    )
    prediction_output["churn_probability"] = probabilities

    return model, metrics, prediction_output, y_test.to_numpy(), probabilities


def create_model_visualizations(
    model: Pipeline, metrics: dict, y_test: np.ndarray, probabilities: np.ndarray
) -> pd.DataFrame:
    cm = np.asarray(metrics["confusion_matrix"])
    disp = ConfusionMatrixDisplay(cm, display_labels=["Stayed", "Churned"])
    disp.plot(cmap="Blues", colorbar=False)
    plt.title("Logistic Regression Confusion Matrix")
    save_current_figure("07_confusion_matrix.png")

    RocCurveDisplay.from_predictions(y_test, probabilities)
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title("Receiver Operating Characteristic Curve")
    save_current_figure("08_roc_curve.png")

    preprocessor = model.named_steps["preprocessor"]
    feature_names = preprocessor.get_feature_names_out()
    coefficients = model.named_steps["classifier"].coef_[0]
    importance = pd.DataFrame(
        {"feature": feature_names, "coefficient": coefficients}
    ).sort_values("coefficient")
    importance.to_csv(OUTPUT_DIR / "feature_coefficients.csv", index=False)

    selected = pd.concat([importance.head(10), importance.tail(10)])
    colors = np.where(selected["coefficient"] > 0, "#ED7D31", "#4472C4")
    ax = selected.plot(
        kind="barh", x="feature", y="coefficient", color=colors, legend=False, figsize=(9, 8)
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set(title="Strongest Logistic Regression Churn Drivers", xlabel="Coefficient", ylabel="")
    save_current_figure("09_feature_coefficients.png")
    return importance


def business_summary(df: pd.DataFrame, metrics: dict, importance: pd.DataFrame) -> str:
    overall_rate = df["ChurnFlag"].mean() * 100
    contract_rates = df.groupby("Contract")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
    internet_rates = df.groupby("InternetService")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
    payment_rates = df.groupby("PaymentMethod")["ChurnFlag"].mean().mul(100).sort_values(ascending=False)
    churned = df[df["ChurnFlag"] == 1]
    retained = df[df["ChurnFlag"] == 0]

    positive = importance.tail(8).sort_values("coefficient", ascending=False)
    negative = importance.head(8)
    feature_table = pd.concat([positive, negative]).to_markdown(index=False, floatfmt=".3f")

    return f"""# Customer Churn Analysis Report

## Executive summary

The dataset contains **{len(df):,} customers**. **{int(df['ChurnFlag'].sum()):,} customers churned**, giving an overall churn rate of **{overall_rate:.1f}%**. The logistic regression model achieved **{metrics['accuracy']:.1%} accuracy**, **{metrics['precision']:.1%} precision**, **{metrics['recall']:.1%} recall**, and **{metrics['roc_auc']:.1%} ROC-AUC** on an unseen 20% test set.

## Main findings

- Month-to-month customers have the highest contract churn rate at **{contract_rates.iloc[0]:.1f}%**, compared with **{contract_rates.iloc[-1]:.1f}%** for the lowest-risk contract category.
- **{internet_rates.index[0]}** customers show the highest internet-service churn rate at **{internet_rates.iloc[0]:.1f}%**.
- **{payment_rates.index[0]}** has the highest payment-method churn rate at **{payment_rates.iloc[0]:.1f}%**.
- Churned customers have an average tenure of **{churned['tenure'].mean():.1f} months**, versus **{retained['tenure'].mean():.1f} months** for retained customers.
- Churned customers pay **${churned['MonthlyCharges'].mean():.2f}** per month on average, versus **${retained['MonthlyCharges'].mean():.2f}** for retained customers.

## Model evaluation

| Metric | Score |
|---|---:|
| Accuracy | {metrics['accuracy']:.3f} |
| Precision (churn) | {metrics['precision']:.3f} |
| Recall (churn) | {metrics['recall']:.3f} |
| ROC-AUC | {metrics['roc_auc']:.3f} |

Confusion matrix (rows are actual classes; columns are predicted classes):

| | Predicted stay | Predicted churn |
|---|---:|---:|
| Actual stay | {metrics['confusion_matrix'][0][0]} | {metrics['confusion_matrix'][0][1]} |
| Actual churn | {metrics['confusion_matrix'][1][0]} | {metrics['confusion_matrix'][1][1]} |

## Strong model indicators

Positive coefficients increase predicted churn probability; negative coefficients reduce it. Coefficients describe associations in this model and should not be interpreted as proof of causation.

{feature_table}

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
"""


def main() -> None:
    prepare_directories()
    df, quality = load_and_clean_data(DATA_PATH)
    df.to_csv(OUTPUT_DIR / "cleaned_telco_churn.csv", index=False)
    create_visualizations(df)

    model, metrics, predictions, y_test, probabilities = build_model(df)
    importance = create_model_visualizations(model, metrics, y_test, probabilities)

    joblib.dump(model, MODEL_DIR / "logistic_regression_pipeline.joblib")
    predictions.sort_values("churn_probability", ascending=False).to_csv(
        OUTPUT_DIR / "test_customer_predictions.csv", index=False
    )
    with (OUTPUT_DIR / "model_metrics.json").open("w", encoding="utf-8") as handle:
        json.dump({"data_quality": quality, "model": metrics}, handle, indent=2)
    (OUTPUT_DIR / "analysis_report.md").write_text(
        business_summary(df, metrics, importance), encoding="utf-8"
    )

    print("Analysis complete")
    print(f"Rows: {len(df):,} | Churn rate: {df['ChurnFlag'].mean():.1%}")
    print(
        "Accuracy: {accuracy:.3f} | Precision: {precision:.3f} | "
        "Recall: {recall:.3f} | ROC-AUC: {roc_auc:.3f}".format(**metrics)
    )
    print(f"Outputs: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
