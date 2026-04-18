from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import VarianceThreshold


def low_variance_filter(
    X: pd.DataFrame,
    threshold: float = 0.0,
) -> tuple[pd.DataFrame, list[str]]:
    """Filter features with variance at or below threshold."""
    selector = VarianceThreshold(threshold=threshold)
    transformed = selector.fit_transform(X)
    selected_cols = X.columns[selector.get_support()].tolist()
    filtered = pd.DataFrame(transformed, columns=selected_cols, index=X.index)
    return filtered, selected_cols


def correlation_prune(
    X: pd.DataFrame,
    threshold: float = 0.95,
) -> tuple[pd.DataFrame, list[str]]:
    """Prune highly correlated numeric features by upper-triangle filtering."""
    numeric_X = X.select_dtypes(include=["number"])
    corr = numeric_X.corr().abs()
    mask = np.triu(np.ones(corr.shape), k=1).astype(bool)
    upper = corr.where(mask)
    to_drop = [column for column in upper.columns if (upper[column] > threshold).any()]

    pruned = X.drop(columns=to_drop, errors="ignore")
    selected_cols = pruned.columns.tolist()
    return pruned, selected_cols


def model_based_importance_select(
    X: pd.DataFrame,
    y: pd.Series,
    top_k: int = 30,
    random_state: int = 42,
) -> tuple[pd.DataFrame, list[str], pd.Series]:
    """Select top-k numeric features using random forest importance."""
    numeric_X = X.select_dtypes(include=["number"]).copy()
    if numeric_X.empty:
        raise ValueError("No numeric columns available for model-based selection.")

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(numeric_X, y)

    importances = pd.Series(model.feature_importances_, index=numeric_X.columns).sort_values(ascending=False)
    selected_cols = importances.head(top_k).index.tolist()
    selected_df = numeric_X[selected_cols]

    return selected_df, selected_cols, importances
