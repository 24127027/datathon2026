"""Model abstractions and wrappers for training/inference."""

from .sklearn_models import SklearnRegressorConfig, SklearnRegressorWrapper

__all__ = [
    "SklearnRegressorConfig",
    "SklearnRegressorWrapper",
]
