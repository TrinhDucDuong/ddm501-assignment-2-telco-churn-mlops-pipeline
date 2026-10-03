"""Inspect actual Airflow metadata and persisted outputs after the Docker DAG run."""

import json
import hashlib
from pathlib import Path

from airflow.models import DagBag, DagRun, TaskInstance
from airflow.utils.session import create_session
import pandas as pd


def main() -> None:
    """Fail on import errors, non-successful tasks, or missing scored output."""
    bag = DagBag("/opt/airflow/dags", include_examples=False)
    assert not bag.import_errors, bag.import_errors
    dag = bag.get_dag("telco_churn_training")
    assert len(dag.tasks) == 7
    expected = ["ingest", "validate", "prepare", "train", "evaluate", "deploy", "score"]
    for first, second in zip(expected, expected[1:]):
        assert second in dag.get_task(first).downstream_task_ids
    with create_session() as session:
        run = (
            session.query(DagRun)
            .filter(DagRun.dag_id == dag.dag_id)
            .order_by(DagRun.execution_date.desc())
            .first()
        )
        assert run is not None and run.state == "success", str(run)
        tasks = (
            session.query(TaskInstance)
            .filter(TaskInstance.dag_id == dag.dag_id, TaskInstance.run_id == run.run_id)
            .all()
        )
        assert len(tasks) == 7 and all(task.state == "success" for task in tasks)
        states = {task.task_id: task.state for task in tasks}
        run_id = run.run_id
    root = Path("/app/artifacts")
    key = hashlib.sha256(run_id.encode()).hexdigest()[:16]
    score_file = root / "airflow_runs" / key / "scored_customers.csv"
    assert score_file.exists(), "Scoring output missing for this run"
    scores = pd.read_csv(score_file)
    assert len(scores) == 1409 and scores.contact.sum() == 282
    result = {
        "state": "success",
        "successful_tasks": len(states),
        "tasks": states,
        "dag_id": dag.dag_id,
        "run_id": run_id,
        "scored_rows": len(scores),
        "contact_rows": int(scores.contact.sum()),
        "evidence": "Airflow metadata DB states + persisted scored CSV",
    }
    (root / "airflow-verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
