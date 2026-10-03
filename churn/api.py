"""Local HTTP demonstrator; production access controls are specified in the report."""

from contextlib import asynccontextmanager
import os
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
import joblib
import pandas as pd
from pydantic import BaseModel, Field

from churn.data import validate


class Batch(BaseModel):
    """Bound the batch size to protect local inference memory and response time."""

    records: list[dict[str, Any]] = Field(min_length=1, max_length=1000)


def create_app(artifact: str | Path | None = None) -> FastAPI:
    """Load an explicitly trusted local model once at startup and fail fast if absent."""
    path = Path(artifact or os.environ.get("CHURN_MODEL_PATH", "artifacts/model.joblib"))

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = joblib.load(path)
        yield

    app = FastAPI(title="Telecom Retention Risk", version="1.0.0", lifespan=lifespan)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": path.name}

    @app.post("/predict")
    def predict(batch: Batch) -> dict[str, Any]:
        try:
            frame = validate(pd.DataFrame(batch.records), labeled=False)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        probabilities = app.state.model.predict_proba(frame)[:, 1]
        return {
            "predictions": [
                {"customerID": str(customer), "churn_probability": float(probability)}
                for customer, probability in zip(frame.customerID, probabilities)
            ]
        }

    return app
