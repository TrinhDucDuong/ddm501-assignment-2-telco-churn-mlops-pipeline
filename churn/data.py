"""Strict data contracts and a frozen, stratified evaluation split."""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

NUMERIC = ["tenure", "MonthlyCharges", "TotalCharges"]
CATEGORIES = {
    "Contract": ["Month-to-month", "One year", "Two year"],
    "InternetService": ["DSL", "Fiber optic", "No"],
    "PaymentMethod": [
        "Electronic check",
        "Mailed check",
        "Bank transfer (automatic)",
        "Credit card (automatic)",
    ],
    "PaperlessBilling": ["Yes", "No"],
    "OnlineSecurity": ["Yes", "No", "No internet service"],
    "TechSupport": ["Yes", "No", "No internet service"],
}
FEATURES = NUMERIC + list(CATEGORIES)


def sha256(path: str | Path) -> str:
    """Fingerprint source bytes, including their original CSV representation."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def validate(frame: pd.DataFrame, labeled: bool = True) -> pd.DataFrame:
    """Reject corrupted batches; allow blank total charges only at zero tenure."""
    required = FEATURES + ["customerID"] + (["Churn"] if labeled else [])
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if frame.empty:
        raise ValueError("Empty batch")
    result = frame[required].copy()
    ids = result.customerID
    if ids.isna().any() or ids.astype(str).str.strip().eq("").any() or ids.duplicated().any():
        raise ValueError("customerID must be nonempty and unique")
    for column in NUMERIC:
        raw = result[column]
        blank = raw.isna() | raw.astype(str).str.strip().eq("")
        numeric = pd.to_numeric(raw, errors="coerce")
        if (numeric.isna() & ~blank).any():
            raise ValueError(f"Nonnumeric values in {column}")
        if column != "TotalCharges" and numeric.isna().any():
            raise ValueError(f"Missing {column}")
        if np.isinf(numeric.to_numpy()).any() or numeric.dropna().lt(0).any():
            raise ValueError(f"Invalid range in {column}")
        result[column] = numeric.astype(float)
    if (result.TotalCharges.isna() & result.tenure.ne(0)).any():
        raise ValueError("Missing TotalCharges for established customer")
    for column, allowed in CATEGORIES.items():
        if not result[column].isin(allowed).all():
            raise ValueError(f"Unknown or missing category in {column}")
    if labeled and not result.Churn.isin(["Yes", "No"]).all():
        raise ValueError("Churn must be Yes or No")
    return result


def split_data(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Create deterministic 60/20/20 partitions before fitting any transformer."""
    train, remaining = train_test_split(frame, test_size=0.4, stratify=frame.Churn, random_state=seed)
    validation, test = train_test_split(remaining, test_size=0.5, stratify=remaining.Churn, random_state=seed)
    return {"train": train, "validation": validation, "test": test}
