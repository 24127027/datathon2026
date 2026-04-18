from __future__ import annotations

from pathlib import Path
import json

import pandas as pd

from ..features.build import build_feature_table
from ..models.sklearn_models import SklearnRegressorWrapper


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
