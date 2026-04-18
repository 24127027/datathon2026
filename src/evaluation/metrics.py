from __future__ import annotations

import numpy as np
import pandas as pd


def mae(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_t - y_p)))


def rmse(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean(np.square(y_t - y_p))))


def mape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, epsilon: float = 1e-8) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_t), epsilon)
    return float(np.mean(np.abs((y_t - y_p) / denom)) * 100.0)


def smape(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, epsilon: float = 1e-8) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)
    denom = np.maximum(np.abs(y_t) + np.abs(y_p), epsilon)
    return float(np.mean(2.0 * np.abs(y_t - y_p) / denom) * 100.0)


def r2(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray, epsilon: float = 1e-12) -> float:
    y_t = np.asarray(y_true, dtype=float)
    y_p = np.asarray(y_pred, dtype=float)

    ss_res = float(np.sum(np.square(y_t - y_p)))
    mean_true = float(np.mean(y_t))
    ss_tot = float(np.sum(np.square(y_t - mean_true)))

    if ss_tot <= epsilon:
        return 0.0
    return float(1.0 - (ss_res / ss_tot))


def regression_metrics(y_true: pd.Series | np.ndarray, y_pred: pd.Series | np.ndarray) -> dict[str, float]:
    return {
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
        "mape": mape(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "r2": r2(y_true, y_pred),
    }


def grouped_regression_metrics(
    df: pd.DataFrame,
    group_col: str,
    y_true_col: str,
    y_pred_col: str,
) -> pd.DataFrame:
    """Compute regression metrics for each subgroup in a dataframe.

    Args:
        df: Input dataframe containing grouping, true, and predicted columns.
        group_col: Column used to split rows into segments (e.g., region/category).
        y_true_col: Column name containing ground-truth target values.
        y_pred_col: Column name containing model predictions.

    Returns:
        A dataframe with one row per group and metric columns:
        `group`, `mae`, `rmse`, `mape`, `smape`, and `r2`.
    """
    rows: list[dict[str, float | str]] = []

    for group_value, group_df in df.groupby(group_col):
        metrics = regression_metrics(group_df[y_true_col], group_df[y_pred_col])
        rows.append({"group": str(group_value), **metrics})

    return pd.DataFrame(rows).sort_values("group").reset_index(drop=True)
