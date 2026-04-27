from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Literal
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge


@dataclass
class SklearnRegressorConfig:
    model_type: Literal[
        "gradient_boosting",
        "random_forest",
        "catboost",
        "lightgbm",
        "xgboost",
    ] = "random_forest"
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
                verbosity=-1,
            )
        if config.model_type == "xgboost":
            from xgboost import XGBRegressor

            return XGBRegressor(
                n_estimators=config.n_estimators,
                max_depth=6 if config.max_depth is None else config.max_depth,
                learning_rate=config.learning_rate,
                random_state=config.random_state,
                n_jobs=-1,
                objective="reg:squarederror",
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
            "gradient_boosting, random_forest, catboost, lightgbm, xgboost"
        )

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> "SklearnRegressorWrapper":
        if isinstance(X, pd.DataFrame):
            self.feature_columns = X.columns.tolist()
            self.model.fit(X, y)
            return self

        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(f"Expected 2D numpy array for fit, got shape {X.shape}")
            # No column labels available when fitting from ndarray.
            self.feature_columns = []
            self.model.fit(X, y)
            return self

        raise TypeError("X must be a pandas DataFrame or a 2D numpy.ndarray")

    def predict(self, X: pd.DataFrame | np.ndarray):
        if isinstance(X, pd.DataFrame):
            if self.feature_columns:
                X = X.reindex(columns=self.feature_columns, fill_value=0.0)
            return self.model.predict(X)

        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(f"Expected 2D numpy array for prediction, got shape {X.shape}")
            if self.feature_columns and X.shape[1] != len(self.feature_columns):
                raise ValueError(
                    "Numpy feature width does not match fitted feature count: "
                    f"expected {len(self.feature_columns)}, got {X.shape[1]}"
                )
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="X does not have valid feature names, but .* was fitted with feature names",
                    category=UserWarning,
                )
                return self.model.predict(X)

        raise TypeError("X must be a pandas DataFrame or a 2D numpy.ndarray")

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
