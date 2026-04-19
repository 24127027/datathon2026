from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import json

import numpy as np
import pandas as pd

from ..features.build import build_feature_table
from ..models.sklearn_models import SklearnRegressorWrapper
from ..utils import prepare_submission


def run_inference_pipeline(
    model_path: str | Path,
    sales_df: pd.DataFrame,
    web_traffic_df: pd.DataFrame | None = None,
    date_col: str = "Date",
    target_col: str = "Revenue",
    selected_columns_path: str | Path | None = None,
) -> pd.DataFrame:
    """Build features consistently and generate predictions from a saved model."""
    model = SklearnRegressorWrapper.load(model_path)

    features = build_feature_table(
        sales_df=sales_df,
        web_traffic_df=web_traffic_df,
        date_col=date_col,
        target_col=target_col,
    ).dropna().reset_index(drop=True)

    X = features.drop(columns=[date_col, target_col, "COGS"], errors="ignore")

    if selected_columns_path is not None:
        selected = json.loads(Path(selected_columns_path).read_text(encoding="utf-8"))
        X = X.reindex(columns=selected, fill_value=0.0)

    predictions = model.predict(X)

    output = features[[date_col]].copy()
    output["prediction"] = predictions
    return output


def predict_submission_from_models(
    models: dict[str, SklearnRegressorWrapper],
    feature_cols: list[str],
    start_date: str = "2023-01-01",
    end_date: str = "2024-07-01",
    revenue_model_name: str = "revenue",
    ratio_model_name: str = "ratio",
    lags: Sequence[int] = (1, 7, 14),
    windows: Sequence[int] = (7, 14),
    eps: float = 1e-6,
) -> pd.DataFrame:
    """Generate submission from revenue and ratio models using recursive daily prediction."""
    if revenue_model_name not in models:
        raise ValueError(f"Missing revenue model: {revenue_model_name}")
    if ratio_model_name not in models:
        raise ValueError(f"Missing ratio model: {ratio_model_name}")

    model_revenue = models[revenue_model_name]
    model_ratio = models[ratio_model_name]

    dates = pd.date_range(start_date, end_date, freq="D")

    feature_index = {name: idx for idx, name in enumerate(feature_cols)}
    X_row = np.zeros((1, len(feature_cols)), dtype="float64")

    lag_values = sorted(set(int(v) for v in lags if int(v) > 0))
    window_values = sorted(set(int(v) for v in windows if int(v) > 1))

    revenue_history: list[float] = []
    ratio_history: list[float] = []

    roll_buffers: dict[int, list[float]] = {w: [] for w in window_values}
    roll_sums: dict[int, float] = {w: 0.0 for w in window_values}
    roll_sumsq: dict[int, float] = {w: 0.0 for w in window_values}

    for current_date in dates:
        X_row.fill(0.0)

        if "year" in feature_index:
            X_row[0, feature_index["year"]] = float(current_date.year)
        if "month" in feature_index:
            X_row[0, feature_index["month"]] = float(current_date.month)
        if "day" in feature_index:
            X_row[0, feature_index["day"]] = float(current_date.day)
        if "day_of_week" in feature_index:
            X_row[0, feature_index["day_of_week"]] = float(current_date.dayofweek)
        if "day_of_year" in feature_index:
            X_row[0, feature_index["day_of_year"]] = float(current_date.dayofyear)
        if "week_of_year" in feature_index:
            X_row[0, feature_index["week_of_year"]] = float(current_date.isocalendar().week)
        if "is_month_end" in feature_index:
            X_row[0, feature_index["is_month_end"]] = float(int(current_date.is_month_end))
        if "is_month_start" in feature_index:
            X_row[0, feature_index["is_month_start"]] = float(int(current_date.is_month_start))
        if "is_weekend" in feature_index:
            X_row[0, feature_index["is_weekend"]] = float(int(current_date.dayofweek >= 5))

        for lag in lag_values:
            col = f"Revenue_lag_{lag}"
            if col in feature_index and len(revenue_history) >= lag:
                X_row[0, feature_index[col]] = revenue_history[-lag]

        for window in window_values:
            mean_col = f"Revenue_roll_mean_{window}"
            std_col = f"Revenue_roll_std_{window}"
            if len(roll_buffers[window]) == window:
                n = float(window)
                mean_val = roll_sums[window] / n
                var_num = roll_sumsq[window] - (roll_sums[window] * roll_sums[window]) / n
                std_val = float(np.sqrt(max(var_num / (n - 1.0), 0.0)))

                if mean_col in feature_index:
                    X_row[0, feature_index[mean_col]] = mean_val
                if std_col in feature_index:
                    X_row[0, feature_index[std_col]] = std_val

        pred_revenue = float(model_revenue.predict(X_row)[0])
        pred_ratio = max(float(model_ratio.predict(X_row)[0]), eps)

        revenue_history.append(pred_revenue)
        ratio_history.append(pred_ratio)

        for window in window_values:
            buffer = roll_buffers[window]
            buffer.append(pred_revenue)
            roll_sums[window] += pred_revenue
            roll_sumsq[window] += pred_revenue * pred_revenue

            if len(buffer) > window:
                removed = buffer.pop(0)
                roll_sums[window] -= removed
                roll_sumsq[window] -= removed * removed

    sub_revenue = pd.Series(revenue_history, dtype="float64")
    sub_ratio = pd.Series(ratio_history, dtype="float64")
    sub_cogs = sub_revenue / sub_ratio

    return prepare_submission(
        revenue=sub_revenue.values,
        cogs=sub_cogs.values,
        start_date=start_date,
        end_date=end_date,
    )
