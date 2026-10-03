"""Measured baseline run, preserving split provenance and scored records."""

import json
import platform
from pathlib import Path
from time import perf_counter
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn

from churn.data import sha256, split_data, validate
from churn.metrics import evaluate
from churn.model import build_model


def write_json(path: Path, value: Any) -> None:
    """Write readable, finite JSON to a temporary file before replacing its target."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def latency_ms(model: Any, frame: pd.DataFrame, repeats: int = 50) -> float:
    """Measure warm single-row model latency; excludes HTTP/network overhead."""
    one = frame.iloc[:1]
    model.predict_proba(one)
    times = []
    for _ in range(repeats):
        start = perf_counter()
        model.predict_proba(one)
        times.append((perf_counter() - start) * 1000)
    return float(np.percentile(times, 95))


def run(config: dict[str, Any]) -> dict[str, Any]:
    """Train a prior baseline and logistic model; persist actual holdout results."""
    out = Path(config["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    data = validate(pd.read_csv(config["data_path"]))
    parts = split_data(data, config["seed"])
    write_json(out / "split_ids.json", {name: part.customerID.tolist() for name, part in parts.items()})
    train, validation, test = (parts[name] for name in ["train", "validation", "test"])
    y_train = train.Churn.eq("Yes").astype(int)
    model = build_model({"algorithm": "logistic", "C": 1.0}, config["seed"])
    dummy = build_model({"algorithm": "dummy"}, config["seed"])
    start = perf_counter()
    model.fit(train, y_train)
    fit_seconds = perf_counter() - start
    dummy.fit(train, y_train)
    test_probability = model.predict_proba(test)[:, 1]
    summary = {
        "rows": len(data),
        "data_sha256": sha256(config["data_path"]),
        "seed": config["seed"],
        "capacity": config["capacity"],
        "split_sizes": {name: len(part) for name, part in parts.items()},
        "missing_total_charges": int(data.TotalCharges.isna().sum()),
        "churn_count": int(data.Churn.eq("Yes").sum()),
        "validation": evaluate(
            validation.Churn.eq("Yes").astype(int), model.predict_proba(validation)[:, 1], config["capacity"]
        ),
        "test": evaluate(test.Churn.eq("Yes").astype(int), test_probability, config["capacity"]),
        "baseline_test": evaluate(
            test.Churn.eq("Yes").astype(int), dummy.predict_proba(test)[:, 1], config["capacity"]
        ),
        "fit_seconds": fit_seconds,
        "p95_model_latency_ms": latency_ms(model, test),
        "python": platform.python_version(),
        "sklearn": sklearn.__version__,
    }
    joblib.dump(model, out / "model.joblib")
    test[["customerID", "Churn"]].assign(churn_probability=test_probability).to_csv(
        out / "test_predictions.csv", index=False
    )
    write_json(out / "summary.json", summary)
    return summary
