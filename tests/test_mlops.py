"""Selection and release invariants, checked independently of training internals."""

import importlib
import json

import pytest


def mlops():
    assert importlib.util.find_spec("churn.mlops") is not None, "MLOps module is not implemented"
    return importlib.import_module("churn.mlops")


def test_winner_is_validation_only_with_speed_tiebreak():
    records = [
        {"name": "a", "average_precision": 0.65, "fit_seconds": 1, "test_ap": 0.99},
        {"name": "b", "average_precision": 0.7, "fit_seconds": 3, "test_ap": 0.3},
        {"name": "c", "average_precision": 0.7, "fit_seconds": 2, "test_ap": 0.2},
    ]
    assert mlops().select_winner(records)["name"] == "c"


def test_failed_gate_preserves_existing_champion(tmp_path):
    module = mlops()
    champion = tmp_path / "champion.json"
    champion.write_text('{"version": "previous"}')
    summary = {
        "validation": {"average_precision": 0.4, "lift_at_capacity": 1.1},
        "baseline_validation_ap": 0.3,
    }
    (tmp_path / "summary.json").write_text(json.dumps(summary))
    model = tmp_path / "model.joblib"
    model.write_bytes(b"previous model bytes")
    with pytest.raises(ValueError, match="gate"):
        module.deploy(
            {
                "output_dir": str(tmp_path),
                "gates": {"min_average_precision": 0.6, "min_lift": 2.0, "min_ap_gain": 0.05},
            }
        )
    assert json.loads(champion.read_text()) == {"version": "previous"}
    assert model.read_bytes() == b"previous model bytes"


def test_gate_rejects_nonfinite_values():
    summary = {
        "validation": {"average_precision": float("nan"), "lift_at_capacity": 3.0},
        "baseline_validation_ap": 0.3,
    }
    with pytest.raises(ValueError):
        mlops().check_gate(summary, {"min_average_precision": 0.6, "min_lift": 2.0, "min_ap_gain": 0.05})


def test_gate_accepts_qualified_candidate():
    summary = {
        "validation": {"average_precision": 0.7, "lift_at_capacity": 2.5},
        "baseline_validation_ap": 0.26,
    }
    mlops().check_gate(summary, {"min_average_precision": 0.6, "min_lift": 2.0, "min_ap_gain": 0.05})


@pytest.mark.parametrize("reverse_matrix", [False, True])
def test_pipeline_registers_model_and_retry_reuses_version(tmp_path, monkeypatch, reverse_matrix):
    from test_core import sample

    module = mlops()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    source = tmp_path / "input.csv"
    sample().to_csv(source, index=False)
    out = tmp_path / "out"
    config = {
        "data_path": str(source),
        "output_dir": str(out),
        "seed": 42,
        "capacity": 0.2,
        "experiment_name": "integration-test",
        "model_name": "integration-churn",
        "gates": {"min_average_precision": 0.6, "min_lift": 1.5, "min_ap_gain": 0.05},
        "experiments": [
            {"name": "prior", "algorithm": "dummy"},
            {"name": "logistic", "algorithm": "logistic", "C": 1.0},
        ],
    }
    if reverse_matrix:
        config["experiments"].reverse()
    result = module.run_mlops(config)
    assert result["winner"] == "logistic"
    first = json.loads((out / "champion.json").read_text())
    module.deploy(config)
    second = json.loads((out / "champion.json").read_text())
    assert first["version"] == second["version"]
    import pandas as pd

    scores = pd.read_csv(out / "scored_customers.csv")
    assert len(scores) == 20
    assert scores.contact.sum() == 4
    from mlflow import MlflowClient

    client = MlflowClient()
    assert client.get_model_version_by_alias("integration-churn", "candidate").run_id == result["run_id"]


def test_changed_source_bytes_fail_before_training(tmp_path):
    module = mlops()
    source = tmp_path / "input.csv"
    source.write_text("altered,data\n")
    (tmp_path / "source.json").write_text('{"sha256": "wrong"}')
    with pytest.raises(ValueError, match="checksum"):
        module.ingest({"data_path": str(source), "output_dir": str(tmp_path / "out")})


def test_moved_tracking_store_requires_fresh_output_directory(tmp_path, monkeypatch):
    import shutil

    module = mlops()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    original = tmp_path / "original"
    relocated = tmp_path / "relocated"
    config = {"output_dir": str(original), "experiment_name": "relocation-test"}
    module.configure_tracking(config)
    relocated.mkdir()
    shutil.copyfile(original / "mlflow.db", relocated / "mlflow.db")
    config["output_dir"] = str(relocated)
    with pytest.raises(ValueError, match="CHURN_OUTPUT_DIR"):
        module.configure_tracking(config)


@pytest.mark.parametrize(
    "specs",
    [
        [{"name": "lr", "algorithm": "logistic"}],
        [{"name": "a", "algorithm": "dummy"}, {"name": "b", "algorithm": "dummy"}],
        [{"name": "a", "algorithm": "dummy"}, {"name": "a", "algorithm": "logistic"}],
    ],
)
def test_ambiguous_experiment_matrix_fails_before_training(tmp_path, specs):
    with pytest.raises(ValueError):
        mlops().train_experiments({"output_dir": str(tmp_path), "experiments": specs})
