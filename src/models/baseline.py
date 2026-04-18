from __future__ import annotations

import numpy as np
import pandas as pd


class SeasonalDateRegressor:
	"""Seasonal baseline regressor that uses only a date column as input.

	Input requirement:
	- `X` must contain exactly one date column (default: `Date`).
	- No other feature columns are required.
	"""

	def __init__(self, date_col: str = "Date") -> None:
		self.date_col = date_col
		self.global_mean_: float | None = None
		self.day_of_year_mean_: dict[int, float] = {}

	def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalDateRegressor":
		"""Fit seasonal averages from target values using only date input."""
		if self.date_col not in X.columns:
			raise ValueError(f"Missing required date column: {self.date_col}")

		frame = pd.DataFrame({self.date_col: pd.to_datetime(X[self.date_col], errors="coerce"), "target": y})
		frame = frame.dropna(subset=[self.date_col, "target"]).copy()
		if frame.empty:
			raise ValueError("No valid rows after parsing dates and targets.")

		frame["day_of_year"] = frame[self.date_col].dt.dayofyear

		self.global_mean_ = float(frame["target"].mean())
		grouped = frame.groupby("day_of_year")["target"].mean()
		day_keys = grouped.index.to_numpy(dtype=int, copy=False)
		day_values = grouped.to_numpy(dtype=float, copy=False)
		self.day_of_year_mean_ = {
			int(day): float(value)
			for day, value in zip(day_keys, day_values, strict=False)
		}
		return self

	def predict(self, X: pd.DataFrame) -> np.ndarray:
		"""Predict using learned day-of-year seasonal means from date-only input."""
		if self.global_mean_ is None:
			raise ValueError("Model is not fitted. Call fit before predict.")
		if self.date_col not in X.columns:
			raise ValueError(f"Missing required date column: {self.date_col}")

		dates = pd.to_datetime(X[self.date_col], errors="coerce")
		day_of_year = dates.dt.dayofyear

		preds = [self.day_of_year_mean_.get(int(d), self.global_mean_) if pd.notna(d) else self.global_mean_ for d in day_of_year]
		return np.asarray(preds, dtype=float)
