"""Behavioral contracts protecting data, leakage and ranking metrics."""

import importlib

import numpy as np
import pandas as pd
import pytest


def core():
    assert importlib.util.find_spec("churn") is not None, "churn package is not implemented"
    return importlib.import_module("churn.data")


def sample(n=100):
    return pd.DataFrame(
        {
            "customerID": [f"customer-{i}" for i in range(n)],
            "tenure": np.arange(n) % 73,
            "MonthlyCharges": np.full(n, 50.0),
            "TotalCharges": np.arange(n).astype(float) * 50,
            "Contract": ["Month-to-month", "One year"] * (n // 2),
            "InternetService": ["DSL", "Fiber optic"] * (n // 2),
            "PaymentMethod": ["Electronic check"] * n,
            "PaperlessBilling": ["Yes"] * n,
            "OnlineSecurity": ["No"] * n,
            "TechSupport": ["No"] * n,
            "Churn": ["No", "Yes"] * (n // 2),
        }
    )


def test_duplicate_customers_are_rejected():
    frame = sample()
    frame.loc[1, "customerID"] = frame.loc[0, "customerID"]
    with pytest.raises(ValueError, match="unique"):
        core().validate(frame)


@pytest.mark.parametrize(
    "column,value",
    [("Churn", "Maybe"), ("tenure", -1), ("MonthlyCharges", float("inf")), ("Contract", "Mystery")],
)
def test_invalid_schema_values_are_rejected(column, value):
    frame = sample()
    frame.loc[0, column] = value
    with pytest.raises(ValueError):
        core().validate(frame)


def test_blank_total_charges_only_allowed_for_new_customers():
    frame = sample().astype({"TotalCharges": object})
    frame.loc[0, "TotalCharges"] = " "
    assert pd.isna(core().validate(frame).loc[0, "TotalCharges"])
    frame.loc[1, "TotalCharges"] = " "
    with pytest.raises(ValueError):
        core().validate(frame)


def test_split_is_repeatable_disjoint_and_complete():
    data = core()
    first = data.split_data(data.validate(sample()), 42)
    second = data.split_data(data.validate(sample()), 42)
    sets = [set(part.customerID) for part in first.values()]
    assert [len(part) for part in first.values()] == [60, 20, 20]
    assert not (sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
    assert len(set.union(*sets)) == 100
    for name in first:
        pd.testing.assert_frame_equal(first[name], second[name])


def test_ranking_metrics_use_capacity_and_handle_ties():
    core()
    metrics = importlib.import_module("churn.metrics")
    actual = metrics.evaluate(np.array([1, 0, 1, 0, 0]), np.array([0.9, 0.8, 0.7, 0.2, 0.1]), 0.4)
    assert actual["precision_at_capacity"] == 0.5
    assert actual["recall_at_capacity"] == 0.5
    assert actual["lift_at_capacity"] == 1.25
    assert actual["selected_count"] == 2


def test_preprocessor_fits_training_data_only_and_excludes_id_target():
    data = core()
    model = importlib.import_module("churn.model")
    train = data.validate(sample())
    pipeline = model.build_model({"algorithm": "logistic", "C": 1.0}, 42)
    pipeline.fit(train, train.Churn.eq("Yes").astype(int))
    before = (
        pipeline.named_steps["preprocess"]
        .named_transformers_["numeric"]
        .named_steps["imputer"]
        .statistics_.copy()
    )
    other = train.iloc[:2].copy()
    other["MonthlyCharges"] = 1000000
    prediction = pipeline.predict_proba(other)[:, 1]
    np.testing.assert_array_equal(
        before,
        pipeline.named_steps["preprocess"].named_transformers_["numeric"].named_steps["imputer"].statistics_,
    )
    names = pipeline.named_steps["preprocess"].get_feature_names_out()
    assert not any("customerID" in name or "Churn" in name for name in names)
    assert np.isfinite(prediction).all()
