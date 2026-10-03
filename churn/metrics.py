"""Ranking and probability metrics; no business uplift is inferred from labels."""

import math

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def evaluate(y: np.ndarray, probability: np.ndarray, capacity: float = 0.2) -> dict[str, float]:
    """Evaluate binary scores at fixed threshold 0.5 and a fixed contact budget."""
    y, probability = np.asarray(y), np.asarray(probability)
    if len(y) == 0 or len(y) != len(probability) or not 0 < capacity <= 1:
        raise ValueError("Nonempty aligned inputs and 0 < capacity <= 1 required")
    if not np.isin(y, [0, 1]).all() or not np.isfinite(probability).all():
        raise ValueError("Invalid labels or nonfinite probabilities")
    if ((probability < 0) | (probability > 1)).any() or len(np.unique(y)) != 2:
        raise ValueError("Both classes and probabilities in [0,1] required")
    count = math.ceil(len(y) * capacity)
    # Stable sorting makes ties reproducible; the dummy baseline remains noninformative.
    selected = np.argsort(-probability, kind="stable")[:count]
    precision = float(y[selected].mean())
    predicted = probability >= 0.5
    tn, fp, fn, tp = confusion_matrix(y, predicted, labels=[0, 1]).ravel()
    return {
        "average_precision": float(average_precision_score(y, probability)),
        "roc_auc": float(roc_auc_score(y, probability)),
        "brier": float(brier_score_loss(y, probability)),
        "f1": float(f1_score(y, predicted, zero_division=0)),
        "precision": float(precision_score(y, predicted, zero_division=0)),
        "recall": float(recall_score(y, predicted, zero_division=0)),
        "precision_at_capacity": precision,
        "recall_at_capacity": float(y[selected].sum() / y.sum()),
        "lift_at_capacity": precision / float(y.mean()),
        "selected_count": count,
        "prevalence": float(y.mean()),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }
