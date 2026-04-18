from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle

import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class SklearnRegressorConfig:
    model_type: str = "random_forest"
    random_state: int = 42
    n_estimators: int = 400
    max_depth: int | None = None
    alpha: float = 1.0
    learning_rate: float = 0.05
    num_leaves: int = 31


class SklearnRegressorWrapper:
    """Unified wrapper for common sklearn regressors with persistence helpers."""

    def __init__(self, config: SklearnRegressorConfig) -> None:
        self.config = config
        self.model = self._build_model(config)
        self.feature_columns: list[str] = []

    @staticmethod
    def _build_model(config: SklearnRegressorConfig):
        if config.model_type == "linear":
            return LinearRegression()
        if config.model_type == "ridge":
            return Ridge(alpha=config.alpha, random_state=config.random_state)
        if config.model_type == "gradient_boosting":
            return GradientBoostingRegressor(random_state=config.random_state)
        if config.model_type == "catboost":
            from catboost import CatBoostRegressor

            return CatBoostRegressor(
                iterations=config.n_estimators,
                depth=config.max_depth,
                learning_rate=config.learning_rate,
                random_seed=config.random_state,
                verbose=False,
            )
        if config.model_type == "lightgbm":
            from lightgbm import LGBMRegressor

            return LGBMRegressor(
                n_estimators=config.n_estimators,
                max_depth=-1 if config.max_depth is None else config.max_depth,
                learning_rate=config.learning_rate,
                num_leaves=config.num_leaves,
                random_state=config.random_state,
                n_jobs=-1,
            )
        if config.model_type == "random_forest":
            return RandomForestRegressor(
                n_estimators=config.n_estimators,
                max_depth=config.max_depth,
                random_state=config.random_state,
                n_jobs=-1,
            )
        raise ValueError(
            "Unsupported model_type: "
            f"{config.model_type}. Use one of: "
            "linear, ridge, gradient_boosting, random_forest, catboost, lightgbm"
        )

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SklearnRegressorWrapper":
        self.feature_columns = X.columns.tolist()
        self.model.fit(X, y)
        return self

    def predict(self, X: pd.DataFrame):
        if self.feature_columns:
            X = X.reindex(columns=self.feature_columns, fill_value=0.0)
        return self.model.predict(X)

    def save(self, artifact_path: str | Path) -> None:
        path = Path(artifact_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as file:
            pickle.dump(self, file)

    @staticmethod
    def load(artifact_path: str | Path) -> "SklearnRegressorWrapper":
        path = Path(artifact_path)
        with path.open("rb") as file:
            loaded = pickle.load(file)

        if not isinstance(loaded, SklearnRegressorWrapper):
            raise TypeError("Loaded artifact is not a SklearnRegressorWrapper")
        return loaded
