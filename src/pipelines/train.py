from __future__ import annotations

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..data.loader import load_sales, load_web_traffic
from ..data.validate import validate_all_dataframes
from ..evaluation.metrics import regression_metrics
from ..features.build import build_feature_table
from ..features.select import correlation_prune, low_variance_filter
from ..models.sklearn_models import SklearnRegressorConfig, SklearnRegressorWrapper


@dataclass
class TrainConfig:
    data_root: str = "data/datathon-2026-round-1"
    date_col: str = "date"
    target_col: str = "Revenue"
    model_type: str = "random_forest"
    valid_fraction: float = 0.2
    output_dir: str = "results/runs"
    run_name: str = "baseline_run"
    random_state: int = 42
    variance_threshold: float = 0.0
    correlation_threshold: float = 0.98


def _validate_blocking_issues(data_root: str) -> None:
    validation = validate_all_dataframes(data_root)
    errors = [
        issue
        for result in validation.values()
        for issue in result.issues
        if issue.severity == "error"
    ]

    if errors:
        details = "\n".join(f"- {e.table}: {e.details}" for e in errors)
        raise ValueError(f"Validation errors found:\n{details}")


def split_by_time(df: pd.DataFrame, date_col: str, valid_fraction: float) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
        Return train/validation splits based on a date column, ensuring temporal integrity for forecasting tasks.
    """
    
    if not 0.0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be between 0 and 0.5")

    sorted_df = df.sort_values(date_col).reset_index(drop=True)
    split_idx = int(len(sorted_df) * (1.0 - valid_fraction))
    return sorted_df.iloc[:split_idx].copy(), sorted_df.iloc[split_idx:].copy()


def build_revenue_ratio_targets(
    df: pd.DataFrame,
    revenue_col: str = "Revenue",
    cogs_col: str = "COGS",
    eps: float = 1e-6,
) -> dict[str, pd.Series]:
    """Build common target dictionary used by revenue/ratio modeling."""
    if revenue_col not in df.columns:
        raise ValueError(f"Missing revenue column: {revenue_col}")
    if cogs_col not in df.columns:
        raise ValueError(f"Missing COGS column: {cogs_col}")

    revenue = df[revenue_col].astype("float64")
    ratio = revenue / df[cogs_col].clip(lower=eps).astype("float64")
    return {"revenue": revenue, "ratio": ratio}


def train_validate_models(
    df: pd.DataFrame,
    features: list[str],
    targets: list[str],
    model_config: SklearnRegressorConfig,
    date_col: str = "date",
    valid_fraction: float = 0.2,
) -> dict[str, object]:
    """Train one model per target on a time split and return validation metrics."""
    if not features:
        raise ValueError("features must not be empty")
    if not targets:
        raise ValueError("targets must contain at least one target column")

    train_df, valid_df = split_by_time(df, date_col=date_col, valid_fraction=valid_fraction)
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


def run_train_pipeline(config: TrainConfig) -> dict[str, object]:
    """Run end-to-end training with deterministic feature creation and persistence."""
    _validate_blocking_issues(config.data_root)

    sales = load_sales(config.data_root)
    web_traffic = load_web_traffic(config.data_root)

    feature_df = build_feature_table(
        sales_df=sales,
        web_traffic_df=web_traffic,
        date_col=config.date_col,
        target_col=config.target_col,
    )
    feature_df = feature_df.dropna().reset_index(drop=True)

    train_df, valid_df = split_by_time(feature_df, config.date_col, config.valid_fraction)

    drop_cols = [config.date_col, config.target_col, "COGS"]
    X_train = train_df.drop(columns=drop_cols, errors="ignore")
    X_valid = valid_df.drop(columns=drop_cols, errors="ignore")
    y_train = train_df[config.target_col]
    y_valid = valid_df[config.target_col]

    X_train, selected_after_variance = low_variance_filter(X_train, threshold=config.variance_threshold)
    X_valid = X_valid.reindex(columns=selected_after_variance, fill_value=0.0)

    X_train, selected_after_corr = correlation_prune(X_train, threshold=config.correlation_threshold)
    X_valid = X_valid.reindex(columns=selected_after_corr, fill_value=0.0)

    model_cfg = SklearnRegressorConfig(
        model_type=config.model_type,
        random_state=config.random_state,
    )
    model = SklearnRegressorWrapper(model_cfg).fit(X_train, y_train)
    preds = model.predict(X_valid)

    metrics = regression_metrics(y_valid, preds)

    run_dir = Path(config.output_dir) / config.run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    model_path = run_dir / "model.pkl"
    model.save(model_path)

    selected_path = run_dir / "selected_columns.json"
    selected_path.write_text(json.dumps(selected_after_corr, indent=2), encoding="utf-8")

    config_path = run_dir / "train_config.json"
    config_path.write_text(json.dumps(asdict(config), indent=2), encoding="utf-8")

    metrics_path = run_dir / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    return {
        "run_dir": str(run_dir),
        "model_path": str(model_path),
        "metrics": metrics,
        "n_train_rows": int(len(train_df)),
        "n_valid_rows": int(len(valid_df)),
        "n_features": int(X_train.shape[1]),
    }
