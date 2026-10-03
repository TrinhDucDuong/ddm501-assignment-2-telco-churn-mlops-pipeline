"""Weekly retraining DAG: persisted stage boundaries and isolated run artifacts."""

from datetime import timedelta
import hashlib
import logging
import os
from pathlib import Path

import pendulum
from airflow import DAG
from airflow.operators.python import PythonOperator

from churn.config import load_config
from churn.mlops import STAGES


def report_failure(context: dict) -> None:
    """Emit a structured failure event; production routes this to its alert service."""
    logging.error(
        "CHURN_PIPELINE_FAILURE dag=%s task=%s run=%s",
        context["dag"].dag_id,
        context["task_instance"].task_id,
        context["run_id"],
    )


def execute_stage(stage_name: str, **context) -> str:
    """Use one stable output directory per DAG run, including task retries."""
    config = load_config(os.environ.get("CHURN_CONFIG", "/app/config.yaml"))
    key = hashlib.sha256(context["run_id"].encode()).hexdigest()[:16]
    config["output_dir"] = str(Path(config["output_dir"]) / "airflow_runs" / key)
    return STAGES[stage_name](config)


with DAG(
    dag_id="telco_churn_training",
    description="Seven real stages, ten tracked experiments, gated local deployment",
    start_date=pendulum.datetime(2026, 1, 1, tz="Asia/Ho_Chi_Minh"),
    schedule="0 2 * * 0",
    catchup=False,
    max_active_runs=1,
    dagrun_timeout=timedelta(minutes=30),
    default_args={
        "owner": "mlops",
        "retries": 2,
        "retry_delay": timedelta(seconds=15),
        "retry_exponential_backoff": True,
        "execution_timeout": timedelta(minutes=15),
        "on_failure_callback": report_failure,
    },
    tags=["ddm501", "churn"],
) as dag:
    tasks = [
        PythonOperator(task_id=name, python_callable=execute_stage, op_kwargs={"stage_name": name})
        for name in STAGES
    ]
    for upstream, downstream in zip(tasks, tasks[1:]):
        upstream >> downstream
