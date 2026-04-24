from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.loader import load_sales, load_web_traffic
from ..data.validate import validate_all_dataframes
from ..evaluation.metrics import regression_metrics
from ..models.sklearn_models import SklearnRegressorConfig, SklearnRegressorWrapper

def train_validate_models(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    model_config: SklearnRegressorConfig,
    date_col: str = "date",
    valid_fraction: float = 0.2,
) -> dict[str, object]:
    """Train one model per target on a time split and return validation metrics.\\
        Returns a dictionary containing:\\
            "train_df": train_df, \\
            "valid_df": valid_df,\\
            "models": models,\\
            "predictions": preds,\\
            "metrics": metrics,\\
    """
    if not features:
        raise ValueError("features must not be empty")
    if not targets:
        raise ValueError("targets must contain at least one target column")

    # Prevent target leakage automatically
    features = [f for f in features if f not in targets]

    train_df, valid_df = _split_by_time(df, date_col=date_col, valid_fraction=valid_fraction)
    X_train = train_df[features]
    X_valid = valid_df[features]

    models: dict[str, SklearnRegressorWrapper] = {}
    preds: dict[str, pd.Series] = {}
    metrics: dict[str, dict[str, float]] = {}

    for target_name in targets:
        y_train = train_df[target_name]
        y_valid = valid_df[target_name]

        model = SklearnRegressorWrapper(model_config)
        model.fit(X_train, y_train)
        pred = pd.Series(model.predict(X_valid), index=valid_df.index, dtype="float64")

        models[target_name] = model
        preds[target_name] = pred
        metrics[target_name] = regression_metrics(y_valid, pred)

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "models": models,
        "predictions": preds,
        "metrics": metrics,
    }


def fit_models_full(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    model_config: SklearnRegressorConfig,
    use_numpy: bool = True,
) -> dict[str, SklearnRegressorWrapper]:
    """Fit one model per target on full data for final submission inference."""
    if not features:
        raise ValueError("features must not be empty")
    if not targets:
        raise ValueError("targets must contain at least one target column")

    # Prevent target leakage automatically
    features = [f for f in features if f not in targets]

    if use_numpy:
        X: pd.DataFrame | np.ndarray = df[features].to_numpy(dtype="float64")
    else:
        X = df[features]

    models: dict[str, SklearnRegressorWrapper] = {}
    for target_name in targets:
        y = df[target_name]
        y_fit: pd.Series | np.ndarray = y.to_numpy(dtype="float64") if use_numpy else y

        model = SklearnRegressorWrapper(model_config)
        model.fit(X, y_fit)
        models[target_name] = model

    return models

def _split_by_time(df: pd.DataFrame, date_col: str, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Return train/validation splits based on a date column, ensuring temporal integrity for forecasting tasks.
    """
    
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be between 0 and 0.5")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - valid_fraction))
    return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()

