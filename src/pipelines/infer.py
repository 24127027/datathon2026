from __future__ import annotations

import re

import numpy as np
import pandas as pd

from ..models.sklearn_models import SklearnRegressorConfig, SklearnRegressorWrapper


_LAG_RE = re.compile(r"^(?P<base>.+)_lag_(?P<value>\d+)$")
_ROLL_MEAN_RE = re.compile(r"^(?P<base>.+)_roll_mean_(?P<value>\d+)$")
_ROLL_STD_RE = re.compile(r"^(?P<base>.+)_roll_std_(?P<value>\d+)$")


def fit_and_predict(
        df: pd.DataFrame,
        model_config: SklearnRegressorConfig,
        train_range: tuple[str, str],
        predict_range: tuple[str, str],
) -> pd.DataFrame:
        """Fit models on a train window and predict Revenue/COGS for a forecast window.

        The function trains:
        - one model on target ``Revenue``
        - one model on target ratio ``Revenue / COGS``

        Then it reconstructs COGS as ``Revenue / ratio`` for the prediction range.
        Lag/rolling features are recursively recomputed for each prediction step.
        """
        date_col = "date"
        if date_col not in df.columns:
                raise ValueError("Missing required date column: 'date'")

        frame = df.copy()
        frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
        frame = frame.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

        train_start, train_end = pd.to_datetime(train_range[0]), pd.to_datetime(train_range[1])
        pred_start, pred_end = pd.to_datetime(predict_range[0]), pd.to_datetime(predict_range[1])

        train_mask = frame[date_col].between(train_start, train_end, inclusive="both")
        pred_mask = frame[date_col].between(pred_start, pred_end, inclusive="both")

        train_df = frame.loc[train_mask].copy()
        pred_df = frame.loc[pred_mask].copy()

        if train_df.empty:
                raise ValueError("Train range produced no rows")
        if pred_df.empty:
                raise ValueError("Predict range produced no rows")

        feature_cols = _select_feature_columns(frame, date_col=date_col)
        if not feature_cols:
                raise ValueError("No numeric feature columns available after exclusions")

        # Fill features using train medians to keep train/predict preprocessing aligned.
        train_medians = train_df[feature_cols].median(numeric_only=True)
        X_train = train_df[feature_cols].fillna(train_medians).fillna(0.0)

        y_revenue = train_df["Revenue"].astype("float64")
        y_ratio = _build_ratio_target(train_df)

        revenue_model = SklearnRegressorWrapper(model_config)
        revenue_model.fit(X_train, y_revenue)

        ratio_model = SklearnRegressorWrapper(model_config)
        ratio_model.fit(X_train, y_ratio)

        recursive_specs = _extract_recursive_specs(feature_cols)
        working = frame.copy()

        # Avoid leaking known future targets in recursive horizon.
        pred_idx = pred_df.index
        working.loc[pred_idx, "Revenue"] = np.nan
        working.loc[pred_idx, "COGS"] = np.nan
        if "gross_margin" in working.columns:
                working.loc[pred_idx, "gross_margin"] = np.nan

        pred_revenue = pd.Series(index=pred_idx, dtype="float64")
        pred_ratio = pd.Series(index=pred_idx, dtype="float64")
        pred_cogs = pd.Series(index=pred_idx, dtype="float64")

        for pos in pred_idx:
                row = working.loc[[pos], feature_cols].copy()
                _recompute_recursive_features_for_row(
                        working=working,
                        row_position=pos,
                        feature_row=row,
                        specs=recursive_specs,
                )
                row = row.fillna(train_medians).fillna(0.0)

                revenue_value = float(np.asarray(revenue_model.predict(row), dtype="float64")[0])
                ratio_value = float(np.asarray(ratio_model.predict(row), dtype="float64")[0])

                revenue_value = max(revenue_value, 0.0)
                ratio_value = max(ratio_value, 1e-6)
                cogs_value = max(revenue_value / ratio_value, 0.0)

                pred_revenue.at[pos] = revenue_value
                pred_ratio.at[pos] = ratio_value
                pred_cogs.at[pos] = cogs_value

                # Feed predictions back for future lag/rolling features.
                working.at[pos, "Revenue"] = revenue_value
                working.at[pos, "COGS"] = cogs_value
                if "gross_margin" in working.columns:
                        working.at[pos, "gross_margin"] = revenue_value - cogs_value

        output = pd.DataFrame(
                {
                        date_col: pred_df[date_col].dt.strftime("%Y-%m-%d"),
                        "Revenue": pred_revenue.reindex(pred_df.index).to_numpy(),
                        "COGS": pred_cogs.reindex(pred_df.index).to_numpy(),
                }
        ).reset_index(drop=True)
        output.rename(columns={date_col: "Date"}, inplace=True)
        return output


def _extract_recursive_specs(feature_cols: list[str]) -> dict[str, dict[str, set[int]]]:
        specs: dict[str, dict[str, set[int]]] = {}
        for col in feature_cols:
                lag_match = _LAG_RE.match(col)
                if lag_match:
                        base = lag_match.group("base")
                        value = int(lag_match.group("value"))
                        if value > 0:
                                specs.setdefault(base, {"lags": set(), "roll_mean": set(), "roll_std": set()})
                                specs[base]["lags"].add(value)
                        continue

                mean_match = _ROLL_MEAN_RE.match(col)
                if mean_match:
                        base = mean_match.group("base")
                        value = int(mean_match.group("value"))
                        if value > 1:
                                specs.setdefault(base, {"lags": set(), "roll_mean": set(), "roll_std": set()})
                                specs[base]["roll_mean"].add(value)
                        continue

                std_match = _ROLL_STD_RE.match(col)
                if std_match:
                        base = std_match.group("base")
                        value = int(std_match.group("value"))
                        if value > 1:
                                specs.setdefault(base, {"lags": set(), "roll_mean": set(), "roll_std": set()})
                                specs[base]["roll_std"].add(value)

        return specs


def _recompute_recursive_features_for_row(
        working: pd.DataFrame,
        row_position: int,
        feature_row: pd.DataFrame,
        specs: dict[str, dict[str, set[int]]],
) -> None:
        row_index = feature_row.index[0]

        for base_col, cfg in specs.items():
                if base_col not in working.columns:
                        continue

                history = pd.to_numeric(working[base_col], errors="coerce")

                for lag in cfg["lags"]:
                        feature_name = f"{base_col}_lag_{lag}"
                        if feature_name not in feature_row.columns:
                                continue
                        prev_pos = row_position - lag
                        value = history.iat[prev_pos] if prev_pos >= 0 else np.nan
                        feature_row.at[row_index, feature_name] = value

                for window in cfg["roll_mean"]:
                        feature_name = f"{base_col}_roll_mean_{window}"
                        if feature_name not in feature_row.columns:
                                continue
                        start = max(0, row_position - window)
                        values = history.iloc[start:row_position]
                        # Match pandas rolling(window) default min_periods=window behavior.
                        if values.notna().sum() >= window:
                                feature_row.at[row_index, feature_name] = values.mean()
                        else:
                                feature_row.at[row_index, feature_name] = np.nan

                for window in cfg["roll_std"]:
                        feature_name = f"{base_col}_roll_std_{window}"
                        if feature_name not in feature_row.columns:
                                continue
                        start = max(0, row_position - window)
                        values = history.iloc[start:row_position]
                        # Match pandas rolling std behavior (ddof=1), requiring full window.
                        if values.notna().sum() >= window:
                                feature_row.at[row_index, feature_name] = values.std(ddof=1)
                        else:
                                feature_row.at[row_index, feature_name] = np.nan


def _select_feature_columns(df: pd.DataFrame, date_col: str) -> list[str]:
        excluded = {date_col, "Revenue", "COGS"}
        return [
                col
                for col in df.columns
                if col not in excluded and pd.api.types.is_numeric_dtype(df[col])
        ]


def _build_ratio_target(train_df: pd.DataFrame) -> pd.Series:
        cogs = train_df["COGS"].astype("float64")
        revenue = train_df["Revenue"].astype("float64")

        ratio = (revenue / cogs.where(cogs.abs() > 1e-9)).replace([np.inf, -np.inf], np.nan)
        median_ratio = ratio.median(skipna=True)
        if pd.isna(median_ratio):
                median_ratio = 1.0
        return ratio.fillna(float(median_ratio)).astype("float64")