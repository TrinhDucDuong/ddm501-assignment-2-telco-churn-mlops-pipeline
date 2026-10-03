"""Train-only preprocessing with explicit exclusion of identity and demographics."""

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from churn.data import CATEGORIES, NUMERIC


def build_model(config: dict[str, Any], seed: int) -> Pipeline:
    """Build a serializable preprocessing/estimator unit for training and serving."""
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scale", StandardScaler())])
    categorical = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    transform = ColumnTransformer(
        [("numeric", numeric, NUMERIC), ("categorical", categorical, list(CATEGORIES))]
    )
    params = {key: value for key, value in config.items() if key not in {"name", "algorithm"}}
    algorithm = config["algorithm"]
    if algorithm == "dummy":
        estimator = DummyClassifier(strategy="prior")
    elif algorithm == "logistic":
        estimator = LogisticRegression(max_iter=2000, random_state=seed, **params)
    elif algorithm == "forest":
        estimator = RandomForestClassifier(random_state=seed, n_jobs=1, **params)
    elif algorithm == "boosting":
        estimator = GradientBoostingClassifier(random_state=seed, **params)
    else:
        raise ValueError(f"Unsupported algorithm: {algorithm}")
    return Pipeline([("preprocess", transform), ("classifier", estimator)])
