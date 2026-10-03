"""Tracked experiments, validation-only model choice and guarded local deployment."""

import hashlib
import json
import math
import os
from pathlib import Path
import platform
import shutil
from time import perf_counter
from typing import Any

import joblib
import mlflow
from mlflow import MlflowClient
from mlflow.models import infer_signature
import pandas as pd
import sklearn

from churn.data import FEATURES, sha256, split_data, validate
from churn.metrics import evaluate
from churn.model import build_model
from churn.pipeline import latency_ms, write_json


def select_winner(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick maximum validation AP; break exact ties by fit time, then name."""
    if not records:
        raise ValueError("No experiment results")
    return sorted(records, key=lambda row: (-row["average_precision"], row["fit_seconds"], row["name"]))[0]


def check_gate(summary: dict[str, Any], gates: dict[str, float]) -> None:
    """Reject unqualified candidates before changing any deployment pointer."""
    metrics = summary["validation"]
    ap, lift = metrics["average_precision"], metrics["lift_at_capacity"]
    gain = ap - summary["baseline_validation_ap"]
    if not all(math.isfinite(value) for value in [ap, lift, gain]):
        raise ValueError("Quality gate: nonfinite metrics")
    if ap < gates["min_average_precision"] or lift < gates["min_lift"] or gain < gates["min_ap_gain"]:
        raise ValueError("Quality gate rejected candidate; champion unchanged")


def configure_tracking(config: dict[str, Any]) -> str:
    """Create a local SQLite registry, overridable with MLFLOW_TRACKING_URI."""
    out = Path(config["output_dir"]).resolve()
    out.mkdir(parents=True, exist_ok=True)
    override = os.environ.get("MLFLOW_TRACKING_URI")
    uri = override or "sqlite:///" + (out / "mlflow.db").as_posix()
    mlflow.set_tracking_uri(uri)
    # MLflow also changes os.environ; implicit local configuration must not become
    # an explicit override for a later call with a different output directory.
    if override is None:
        os.environ.pop("MLFLOW_TRACKING_URI", None)
    client = MlflowClient()
    name = config["experiment_name"]
    experiment = client.get_experiment_by_name(name)
    if experiment is None:
        client.create_experiment(name, artifact_location=(out / "mlruns").as_uri())
    elif override is None and experiment.artifact_location != (out / "mlruns").as_uri():
        raise ValueError(
            "This MLflow store belongs to a different project location. "
            "Set CHURN_OUTPUT_DIR to a new empty directory (for example artifacts-fresh), "
            "then rerun; keep the old directory only as historical evidence."
        )
    mlflow.set_experiment(name)
    return uri


def ingest(config: dict[str, Any]) -> str:
    """Copy the immutable input snapshot and verify its published local checksum."""
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    source = Path(config["data_path"])
    digest = sha256(source)
    manifest = source.with_name("source.json")
    if manifest.exists() and json.loads(manifest.read_text())["sha256"] != digest:
        raise ValueError("Data checksum mismatch")
    shutil.copyfile(source, out / "raw.csv")
    write_json(out / "ingestion.json", {"sha256": digest, "bytes": source.stat().st_size})
    return str(out / "raw.csv")


def validate_stage(config: dict[str, Any]) -> str:
    """Persist a validated snapshot; bad data fails the stage before training."""
    out = Path(config["output_dir"])
    frame = validate(pd.read_csv(out / "raw.csv"))
    frame.to_csv(out / "validated.csv", index=False)
    write_json(
        out / "validation.json",
        {
            "rows": len(frame),
            "missing_total_charges": int(frame.TotalCharges.isna().sum()),
            "churn_count": int(frame.Churn.eq("Yes").sum()),
        },
    )
    return str(out / "validated.csv")


def prepare(config: dict[str, Any]) -> str:
    """Materialize disjoint partitions; fitted preprocessing stays inside each trial."""
    out = Path(config["output_dir"])
    parts = split_data(validate(pd.read_csv(out / "validated.csv")), config["seed"])
    for name, part in parts.items():
        part.to_csv(out / f"{name}.csv", index=False)
    write_json(out / "split_ids.json", {name: part.customerID.tolist() for name, part in parts.items()})
    return str(out / "split_ids.json")


def source_hash() -> str:
    """Fingerprint all Python modules, independent of a Git checkout being present."""
    digest = hashlib.sha256()
    for file in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(file.name.encode())
        digest.update(file.read_bytes())
    return digest.hexdigest()


def train_experiments(config: dict[str, Any]) -> str:
    """Run the declared matrix on one split and log actual parameters and artifacts."""
    out = Path(config["output_dir"])
    specs = config["experiments"]
    baselines = [spec for spec in specs if spec.get("algorithm") == "dummy"]
    if len(baselines) != 1:
        raise ValueError("Experiment matrix must contain exactly one dummy baseline")
    if len({spec["name"] for spec in specs}) != len(specs):
        raise ValueError("Experiment names must be unique")
    configure_tracking(config)
    train, val = (pd.read_csv(out / f"{name}.csv") for name in ["train", "validation"])
    records = []
    data_hash = json.loads((out / "ingestion.json").read_text())["sha256"]
    for spec in config["experiments"]:
        with mlflow.start_run(run_name=spec["name"]) as active:
            model = build_model(spec, config["seed"])
            start = perf_counter()
            model.fit(train, train.Churn.eq("Yes").astype(int))
            fit_seconds = perf_counter() - start
            metrics = evaluate(
                val.Churn.eq("Yes").astype(int), model.predict_proba(val)[:, 1], config["capacity"]
            )
            metrics.update(fit_seconds=fit_seconds, p95_model_latency_ms=latency_ms(model, val))
            mlflow.log_params(
                {
                    **spec,
                    "seed": config["seed"],
                    "capacity": config["capacity"],
                    "features": ",".join(FEATURES),
                }
            )
            mlflow.set_tags(
                {
                    "data_sha256": data_hash,
                    "source_sha256": source_hash(),
                    "split": "60/20/20",
                    "metric_partition": "validation",
                }
            )
            mlflow.log_metrics(metrics)
            mlflow.log_artifact(str(out / "split_ids.json"))
            example = train[FEATURES].head(3)
            mlflow.sklearn.log_model(
                model,
                "model",
                signature=infer_signature(example, model.predict(example)),
                input_example=example,
                pip_requirements=["scikit-learn==1.7.2", "pandas==2.3.3", "numpy==2.2.6"],
            )
            joblib.dump(model, out / f"{spec['name']}.joblib")
            records.append({"name": spec["name"], "run_id": active.info.run_id, **metrics})
    pd.DataFrame(records).to_csv(out / "experiments.csv", index=False)
    write_json(out / "winner.json", select_winner(records))
    return str(out / "experiments.csv")


def evaluate_winner(config: dict[str, Any]) -> str:
    """Evaluate the preselected model on holdout; do not reselect using test scores."""
    out = Path(config["output_dir"])
    winner = json.loads((out / "winner.json").read_text())
    model = joblib.load(out / f"{winner['name']}.joblib")
    test = pd.read_csv(out / "test.csv")
    probability = model.predict_proba(test)[:, 1]
    metrics = evaluate(test.Churn.eq("Yes").astype(int), probability, config["capacity"])
    trials = pd.read_csv(out / "experiments.csv")
    validation = {key: value for key, value in winner.items() if key not in {"name", "run_id"}}
    baseline_names = [spec["name"] for spec in config["experiments"] if spec["algorithm"] == "dummy"]
    if len(baseline_names) != 1:
        raise ValueError("Experiment matrix must contain exactly one dummy baseline")
    baseline = trials.loc[trials.name.eq(baseline_names[0])]
    if len(baseline) != 1:
        raise ValueError("Baseline result is missing or duplicated")
    summary = {
        "winner": winner["name"],
        "run_id": winner["run_id"],
        "validation": validation,
        "test": metrics,
        "baseline_validation_ap": float(baseline.iloc[0].average_precision),
        "data_sha256": json.loads((out / "ingestion.json").read_text())["sha256"],
        "source_sha256": source_hash(),
        "seed": config["seed"],
        "capacity": config["capacity"],
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
        "rows": json.loads((out / "validation.json").read_text())["rows"],
        "split_sizes": {
            name: len(pd.read_csv(out / f"{name}.csv")) for name in ["train", "validation", "test"]
        },
    }
    write_json(out / "summary.json", summary)
    test[["customerID", "Churn"]].assign(churn_probability=probability).to_csv(
        out / "test_predictions.csv", index=False
    )
    configure_tracking(config)
    with mlflow.start_run(run_id=winner["run_id"]):
        mlflow.log_metrics({"test_" + key: value for key, value in metrics.items()})
        mlflow.log_artifact(str(out / "summary.json"))
    return str(out / "summary.json")


def deploy(config: dict[str, Any]) -> str:
    """Register a candidate and atomically publish the local artifact after the gate."""
    out = Path(config["output_dir"])
    summary = json.loads((out / "summary.json").read_text())
    check_gate(summary, config["gates"])
    configure_tracking(config)
    client = MlflowClient()
    name, run_id = config["model_name"], summary["run_id"]
    # Reuse the same run's version on Airflow retries instead of duplicating releases.
    versions = client.search_model_versions(f"name='{name}'")
    version = next((item for item in versions if item.run_id == run_id), None)
    if version is None:
        version = mlflow.register_model(f"runs:/{run_id}/model", name)
    staged = out / "model.joblib.tmp"
    shutil.copyfile(out / f"{summary['winner']}.joblib", staged)
    joblib.load(staged)
    staged.replace(out / "model.joblib")
    client.set_registered_model_alias(name, "candidate", version.version)
    write_json(
        out / "champion.json",
        {
            "name": name,
            "version": version.version,
            "run_id": run_id,
            "data_sha256": summary["data_sha256"],
            "artifact_sha256": sha256(out / "model.joblib"),
            "scope": "local demonstration only; production approval and A/B test required",
        },
    )
    return str(out / "champion.json")


def score(config: dict[str, Any]) -> str:
    """Score the holdout as a simulated incoming batch; publish a capped contact list."""
    out = Path(config["output_dir"])
    model = joblib.load(out / "model.joblib")
    frame = validate(pd.read_csv(out / "test.csv"), labeled=False)
    result = frame[["customerID"]].assign(churn_probability=model.predict_proba(frame)[:, 1])
    result = result.sort_values("churn_probability", ascending=False, kind="stable")
    result["contact"] = [index < math.ceil(len(result) * config["capacity"]) for index in range(len(result))]
    result.to_csv(out / "scored_customers.csv", index=False)
    return str(out / "scored_customers.csv")


STAGES = {
    "ingest": ingest,
    "validate": validate_stage,
    "prepare": prepare,
    "train": train_experiments,
    "evaluate": evaluate_winner,
    "deploy": deploy,
    "score": score,
}


def run_mlops(config: dict[str, Any]) -> dict[str, Any]:
    """Execute the same persisted stages used by the orchestration DAG."""
    for stage in STAGES.values():
        stage(config)
    return json.loads((Path(config["output_dir"]) / "summary.json").read_text())
