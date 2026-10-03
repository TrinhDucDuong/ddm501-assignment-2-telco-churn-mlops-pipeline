# Executed verification

Project: assignment_2

- Tests: 23 passed, 0 failed, 0 errors, 0 skipped. See pytest.txt and pytest.xml.
- Dependency validation: pip check exited 0. See pip-check.txt.
- Code validation: Ruff exited 0. See lint.txt.
- Full dataset run: 7,043 records, seed 42, 4,225/1,409/1,409 split. See pipeline.txt and summary.json.
- Actual HTTP: healthy server, valid prediction 200, invalid empty batch 422. See verification.json.
- Portability: source, data and tests copied into an unrelated temporary directory without a sibling project; tests and full training ran successfully. See portability.json.
- PDF: 13 pages, identity checked, no out-of-page text, all pages rendered and visually reviewed. See pdf-qa.json.

The Python interpreter path in verification.json points to this project's own .venv. No parent or sibling environment is required. Upstream libraries emit deprecation warnings; they are retained in the logs and are not test failures.

- Airflow 2.10.2: seven real tasks completed successfully in Docker, verified against the metadata database, producing 1,409 score rows and 282 contact flags. See airflow.txt and airflow-summary.json.
- MLflow: ten configured experiments, candidate registered with numeric version and alias; metadata in champion.json.

- Standalone Python 3.11 Linux image: full ten-experiment pipeline completed with SQLAlchemy 2.0.54; produced 1,409 scores and 282 contacts. See docker-pipeline.txt and docker-summary.json.
