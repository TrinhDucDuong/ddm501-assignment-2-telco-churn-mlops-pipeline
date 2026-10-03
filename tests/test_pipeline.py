"""Real training and HTTP boundary tests on small deterministic data."""

import importlib
import json

from fastapi.testclient import TestClient
import joblib
import numpy as np
import pytest

from test_core import sample


def pipeline_module():
    assert importlib.util.find_spec("churn.pipeline") is not None, "pipeline is not implemented"
    return importlib.import_module("churn.pipeline")


def test_training_persists_predictions_and_disjoint_split_ids(tmp_path):
    module = pipeline_module()
    source = tmp_path / "input.csv"
    sample().to_csv(source, index=False)
    config = {"data_path": str(source), "output_dir": str(tmp_path / "out"), "seed": 42, "capacity": 0.2}
    result = module.run(config)
    assert result["rows"] == 100
    assert len(result["data_sha256"]) == 64
    model = joblib.load(tmp_path / "out/model.joblib")
    assert np.isfinite(model.predict_proba(sample())).all()
    metadata = json.loads((tmp_path / "out/summary.json").read_text())
    assert metadata["test"]["average_precision"] > metadata["baseline_test"]["average_precision"]
    ids = json.loads((tmp_path / "out/split_ids.json").read_text())
    assert not set(ids["train"]) & set(ids["test"])


def test_api_predicts_and_rejects_corrupt_payload(tmp_path):
    pipeline_module()
    assert importlib.util.find_spec("churn.api") is not None, "API is not implemented"
    from churn.api import create_app
    from churn.model import build_model

    frame = sample()
    model = build_model({"algorithm": "logistic", "C": 1.0}, 42)
    model.fit(frame, frame.Churn.eq("Yes").astype(int))
    artifact = tmp_path / "model.joblib"
    joblib.dump(model, artifact)
    with TestClient(create_app(artifact)) as client:
        assert client.get("/health").status_code == 200
        records = frame.drop(columns="Churn").iloc[:2].to_dict(orient="records")
        response = client.post("/predict", json={"records": records})
        assert response.status_code == 200
        predictions = response.json()["predictions"]
        assert len(predictions) == 2
        assert all(0 <= item["churn_probability"] <= 1 for item in predictions)
        records[0]["MonthlyCharges"] = -10
        assert client.post("/predict", json={"records": records}).status_code == 422
        assert client.post("/predict", json={"records": []}).status_code == 422
        assert client.post("/predict", json={"records": [{"customerID": "missing"}]}).status_code == 422


def test_config_rejects_invalid_capacity(tmp_path):
    pipeline_module()
    from churn.config import load_config

    file = tmp_path / "config.yaml"
    file.write_text("seed: 42\ncapacity: 0\ndata_path: input.csv\noutput_dir: out\n")
    with pytest.raises(ValueError):
        load_config(file)
