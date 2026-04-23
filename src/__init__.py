"""Top-level package for the Datathon 2026 forecasting project."""

from . import data, evaluation, features, models, pipelines, utils

__version__ = "0.1.0"

__all__ = [
	"__version__",
	"data",
	"evaluation",
	"features",
	"models",
	"pipelines",
	"utils",
]
