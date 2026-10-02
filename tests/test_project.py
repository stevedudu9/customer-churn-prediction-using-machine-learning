"""Regression, train-only preprocessing, invalid-data and app input checks."""
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.churn_analysis import DATA_PATH, RANDOM_STATE, build_model, load_and_clean_data


@pytest.fixture(scope="module")
def analysis():
    df, quality = load_and_clean_data(DATA_PATH)
    return df, quality, build_model(df)


def test_descriptive_and_model_benchmark(analysis):
    df, quality, (_, metrics, _, _, _) = analysis
    assert quality["blank_total_charges_filled_with_zero"] == 11
    assert len(df) == 7043 and int(df.ChurnFlag.sum()) == 1869
    month = df[df.Contract.eq("Month-to-month")]
    assert len(month) == 3875 and int(month.ChurnFlag.sum()) == 1655
    assert df.loc[df.ChurnFlag.eq(1), "tenure"].mean() == pytest.approx(17.979133226324237)
    assert df.loc[df.ChurnFlag.eq(0), "tenure"].mean() == pytest.approx(37.56996521066873)
    assert metrics["roc_auc"] == pytest.approx(0.8419695678007698, abs=1e-9)
    assert metrics["confusion_matrix"] == [[926,109],[165,209]]
    assert metrics["f1"] == pytest.approx(0.6040462427745664)


def test_learned_preprocessing_uses_training_rows(analysis):
    df, _, (model, _, predictions, _, _) = analysis
    X = df.drop(columns=["customerID", "Churn", "ChurnFlag"])
    Xt, Xv, _, _ = train_test_split(X, df.ChurnFlag, test_size=.2, stratify=df.ChurnFlag, random_state=RANDOM_STATE)
    assert set(Xt.index).isdisjoint(Xv.index)
    assert set(predictions.customerID) == set(df.loc[Xv.index, "customerID"])
    pre = model.named_steps["preprocessor"]
    numeric = ["tenure","MonthlyCharges","TotalCharges"]
    fitted = pre.named_transformers_["numeric"]
    np.testing.assert_allclose(fitted.named_steps["imputer"].statistics_, Xt[numeric].median())
    np.testing.assert_allclose(fitted.named_steps["scaler"].mean_, Xt[numeric].mean())
    assert not np.allclose(fitted.named_steps["scaler"].mean_, X[numeric].mean())


@pytest.mark.parametrize("fault", ["invalid_target", "missing_charges", "duplicate_id", "missing_column"])
def test_bad_data_fails_explicitly(tmp_path, fault):
    df = pd.read_csv(DATA_PATH)
    if fault == "invalid_target":
        df.loc[0,"Churn"] = "Unknown"
    elif fault == "missing_charges":
        df.loc[0,"TotalCharges"] = "bad"
        df.loc[0,"tenure"] = 12
    elif fault == "duplicate_id":
        df.loc[1,"customerID"] = df.loc[0,"customerID"]
    else:
        df = df.drop(columns="Churn")
    path = tmp_path / "bad.csv"
    df.to_csv(path,index=False)
    with pytest.raises(ValueError):
        load_and_clean_data(path)


def test_app_service_consistency_and_scoring():
    at = AppTest.from_file(str(ROOT/"app.py")).run(timeout=30)
    assert not at.exception
    assert any(m.label == "Month-to-month churn" and m.value == "42.7%" for m in at.metric)
    assert any(m.label == "Test ROC-AUC" and m.value == "0.842" for m in at.metric)
    boxes = {b.label:b for b in at.sidebar.selectbox}
    assert boxes["Multiple lines"].value == "No phone service"
    boxes["Internet service"].set_value("No")
    at.run(timeout=30)
    assert not at.exception
    boxes = {b.label:b for b in at.sidebar.selectbox}
    for label in ["Online security","Online backup","Device protection","Tech support","Streaming TV","Streaming movies"]:
        assert boxes[label].value == "No internet service"
    boxes["Phone service"].set_value("Yes")
    boxes["Internet service"].set_value("Fiber optic")
    at.run(timeout=30)
    assert not at.exception
    boxes = {b.label:b for b in at.sidebar.selectbox}
    assert boxes["Multiple lines"].options == ["No","Yes"]
