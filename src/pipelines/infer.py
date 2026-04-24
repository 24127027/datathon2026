from __future__ import annotations

import numpy as np
import pandas as pd

from ..models.sklearn_models import SklearnRegressorConfig, SklearnRegressorWrapper


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
        """
        date_col = "date"
        if date_col not in df.columns:
                raise ValueError("Missing required date column: 'date'")

        frame = df.copy()

        train_start, train_end = pd.to_datetime(train_range[0]), pd.to_datetime(train_range[1])
        pred_start, pred_end = pd.to_datetime(predict_range[0]), pd.to_datetime(predict_range[1])

        train_mask = frame[date_col].between(train_start, train_end, inclusive="both")
        pred_mask = frame[date_col].between(pred_start, pred_end, inclusive="both")

        train_df = frame.loc[train_mask].copy()
        pred_df = frame.loc[pred_mask].copy()

        feature_cols = _select_feature_columns(frame, date_col=date_col)

        # Fill features using train medians to keep train/predict preprocessing aligned.
        train_medians = train_df[feature_cols].median(numeric_only=True)
        X_train = train_df[feature_cols].fillna(train_medians).fillna(0.0)
        X_pred = pred_df[feature_cols].fillna(train_medians).fillna(0.0)

        y_revenue = train_df["Revenue"].astype("float64")
        y_ratio = _build_ratio_target(train_df)

        revenue_model = SklearnRegressorWrapper(model_config)
        revenue_model.fit(X_train, y_revenue)

        ratio_model = SklearnRegressorWrapper(model_config)
        ratio_model.fit(X_train, y_ratio)

        pred_revenue = pd.Series(
                np.asarray(revenue_model.predict(X_pred), dtype="float64"),
                index=pred_df.index,
                dtype="float64",
        )
        pred_ratio = pd.Series(
                np.asarray(ratio_model.predict(X_pred), dtype="float64"),
                index=pred_df.index,
                dtype="float64",
        )

        pred_revenue = pred_revenue.clip(lower=0.0)
        pred_ratio = pred_ratio.clip(lower=1e-6)
        pred_cogs = (pred_revenue / pred_ratio).clip(lower=0.0)

        output = pd.DataFrame(
                {
                        date_col: pred_df[date_col].dt.strftime("%Y-%m-%d"),
                        "Revenue": pred_revenue.to_numpy(),
                        "COGS": pred_cogs.to_numpy(),
                }
        ).reset_index(drop=True)
        output.rename(columns={date_col: "Date"}, inplace=True)
        return output


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