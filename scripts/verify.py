"""Run local tests, train on the full dataset, and probe a real HTTP server."""

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import shutil
import socket
import subprocess
import sys
import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "evidence"


def command(name: str, arguments: list[str]) -> dict:
    """Persist exact command, stdout/stderr and status for an auditable execution."""
    result = subprocess.run([sys.executable, *arguments], cwd=ROOT, capture_output=True, text=True)
    (EVIDENCE / f"{name}.txt").write_text(
        f"COMMAND: python {' '.join(arguments)}\nEXIT: {result.returncode}\n\n{result.stdout}\n{result.stderr}",
        encoding="utf-8",
    )
    if result.returncode:
        raise RuntimeError(f"{name} failed; see evidence/{name}.txt")
    return {"exit_code": result.returncode, "evidence": f"{name}.txt"}


def http_smoke() -> dict:
    """Exercise actual TCP/HTTP, with a unique port and guaranteed child cleanup."""
    with socket.socket() as candidate:
        candidate.bind(("127.0.0.1", 0))
        port = candidate.getsockname()[1]
    with (EVIDENCE / "api-server.txt").open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "uvicorn",
                "churn.api:create_app",
                "--factory",
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            deadline = time.monotonic() + 60
            while True:
                try:
                    with urlopen(url + "/health", timeout=2) as response:
                        health = json.load(response)
                    break
                except (URLError, TimeoutError):
                    if process.poll() is not None or time.monotonic() > deadline:
                        raise RuntimeError("API failed to become ready")
                    time.sleep(0.2)
            import pandas as pd

            frame = pd.read_csv(ROOT / "data/telco.csv").drop(columns="Churn").head(2)
            request = Request(
                url + "/predict",
                data=json.dumps({"records": frame.to_dict(orient="records")}).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urlopen(request, timeout=10) as response:
                status = response.status
                prediction = json.load(response)
            assert status == 200 and len(prediction["predictions"]) == 2
            assert all(0 <= row["churn_probability"] <= 1 for row in prediction["predictions"])
            invalid = Request(
                url + "/predict", data=b'{"records": []}', headers={"Content-Type": "application/json"}
            )
            try:
                urlopen(invalid, timeout=10)
                raise AssertionError("Invalid batch was accepted")
            except HTTPError as error:
                assert error.code == 422
            return {
                "transport": "real HTTP over loopback",
                "health": health,
                "valid_status": status,
                "invalid_status": 422,
                "response": prediction,
            }
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main() -> None:
    """Collect evidence without depending on sibling directories or global packages."""
    os.chdir(ROOT)
    EVIDENCE.mkdir(exist_ok=True)
    results = {
        "executed_utc": datetime.now(timezone.utc).isoformat(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "executable": sys.executable,
        "project": ROOT.name,
    }
    results["dependencies"] = command("pip-check", ["-m", "pip", "check"])
    results["lint"] = command("lint", ["-m", "ruff", "check", "churn", "tests", "scripts"])
    results["tests"] = command("pytest", ["-m", "pytest", "-q", "--junitxml=evidence/pytest.xml"])
    results["training"] = command("pipeline", ["-m", "churn"])
    results["http"] = http_smoke()
    for name in ["summary.json", "experiments.csv", "champion.json", "source.json"]:
        source = ROOT / ("data" if name == "source.json" else "artifacts") / name
        if source.exists():
            shutil.copyfile(source, EVIDENCE / name)
    freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    (ROOT / "requirements-lock-windows.txt").write_text(freeze, encoding="utf-8")
    (EVIDENCE / "verification.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
