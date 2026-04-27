from __future__ import annotations

import numpy as np
import pandas as pd

from ..evaluation.metrics import regression_metrics
from ..models.sklearn_models import SklearnRegressorConfig, SklearnRegressorWrapper

def train_validate_models(
    df: pd.DataFrame,
    model_config: SklearnRegressorConfig,
    train_range: tuple[str, str],
    predict_range: tuple[str, str],
    date_col: str = "date",
) -> dict[str, object]:
    """Train on one date range and validate on another using the infer-style interface.

    Returns:
        dict[str, object]:
            - "train_df": Training rows selected by train_range.
            - "valid_df": Validation rows selected by predict_range.
            - "models": Dict with fitted models for "Revenue" and "COGS".
            - "feature_cols": Numeric feature columns used for fitting/prediction.
            - "train_medians": Per-feature medians computed on train_df for NA filling.
            - "predictions": Dict with validation predictions for each target.
            - "metrics": Dict with regression metrics per target on validation rows.
    """
    frame = _prepare_frame(df, date_col=date_col)
    train_df, valid_df = _split_by_range(frame, date_col=date_col, train_range=train_range, predict_range=predict_range)
    feature_cols = _select_feature_columns(frame, date_col=date_col)

    models, train_medians = _fit_target_models(
        train_df=train_df,
        feature_cols=feature_cols,
        model_config=model_config,
    )

    X_valid = valid_df[feature_cols].fillna(train_medians).fillna(0.0)
    preds: dict[str, pd.Series] = {
        target_name: pd.Series(
            np.asarray(model.predict(X_valid), dtype="float64"),
            index=valid_df.index,
            dtype="float64",
        )
        for target_name, model in models.items()
    }
    metrics = {
        target_name: regression_metrics(valid_df[target_name].astype("float64"), pred)
        for target_name, pred in preds.items()
    }

    return {
        "train_df": train_df,
        "valid_df": valid_df,
        "models": models,
        "feature_cols": feature_cols,
        "train_medians": train_medians,
        "predictions": preds,
        "metrics": metrics,
    }


def _prepare_frame(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: '{date_col}'")

    required_targets = {"Revenue", "COGS"}
    missing = required_targets.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required target columns: {sorted(missing)}")

    frame = df.copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    if frame.empty:
        raise ValueError("No rows available after parsing date column")
    return frame


def _split_by_range(
    df: pd.DataFrame,
    date_col: str,
    train_range: tuple[str, str],
    predict_range: tuple[str, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_start, train_end = pd.to_datetime(train_range[0]), pd.to_datetime(train_range[1])
    pred_start, pred_end = pd.to_datetime(predict_range[0]), pd.to_datetime(predict_range[1])

    train_mask = df[date_col].between(train_start, train_end, inclusive="both")
    pred_mask = df[date_col].between(pred_start, pred_end, inclusive="both")
    train_df = df.loc[train_mask].copy()
    pred_df = df.loc[pred_mask].copy()

    if train_df.empty:
        raise ValueError("Train range produced no rows")
    if pred_df.empty:
        raise ValueError("Predict range produced no rows")
    return train_df, pred_df


def _select_feature_columns(df: pd.DataFrame, date_col: str) -> list[str]:
    excluded = {date_col, "Revenue", "COGS"}
    feature_cols = [
        col
        for col in df.columns
        if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
    ]
    if not feature_cols:
        raise ValueError("No numeric feature columns available after exclusions")
    return feature_cols


def _fit_target_models(
    train_df: pd.DataFrame,
    feature_cols: list[str],
    model_config: SklearnRegressorConfig,
) -> tuple[dict[str, SklearnRegressorWrapper], pd.Series]:
    train_medians = train_df[feature_cols].median(numeric_only=True)
    X_train = train_df[feature_cols].fillna(train_medians).fillna(0.0)

    models: dict[str, SklearnRegressorWrapper] = {}
    for target_name in ["Revenue", "COGS"]:
        y_train = train_df[target_name].astype("float64")
        model = SklearnRegressorWrapper(model_config)
        model.fit(X_train, y_train)
        models[target_name] = model
    return models, train_medians

