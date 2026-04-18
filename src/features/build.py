from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def add_time_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Add deterministic calendar features from a date column."""
    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")

    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["day_of_year"] = out[date_col].dt.dayofyear
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype("Int64")
    out["is_month_end"] = out[date_col].dt.is_month_end.astype(int)
    out["is_month_start"] = out[date_col].dt.is_month_start.astype(int)
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    return out


def add_lag_features(
    df: pd.DataFrame,
    target_col: str,
    lags: Iterable[int],
    date_col: str,
) -> pd.DataFrame:
    """Create lag features for a target using chronological order."""
    out = df.copy().sort_values(date_col)
    for lag in sorted(set(lags)):
        if lag <= 0:
            continue
        out[f"{target_col}_lag_{lag}"] = out[target_col].shift(lag)
    return out


def add_rolling_features(
    df: pd.DataFrame,
    target_col: str,
    windows: Iterable[int],
    date_col: str,
) -> pd.DataFrame:
    """Create rolling mean/std features from lagged target values."""
    out = df.copy().sort_values(date_col)
    shifted = out[target_col].shift(1)

    for window in sorted(set(windows)):
        if window <= 1:
            continue
        out[f"{target_col}_roll_mean_{window}"] = shifted.rolling(window).mean()
        out[f"{target_col}_roll_std_{window}"] = shifted.rolling(window).std()

    return out


def add_ratio_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create stable ratio features from known business columns when available."""
    out = df.copy()

    if {"Revenue", "COGS"}.issubset(out.columns):
        out["gross_margin"] = out["Revenue"] - out["COGS"]
        out["gross_margin_ratio"] = (out["gross_margin"] / out["Revenue"].replace(0, pd.NA)).fillna(0.0)

    if {"sessions", "unique_visitors"}.issubset(out.columns):
        out["sessions_per_visitor"] = (
            out["sessions"] / out["unique_visitors"].replace(0, pd.NA)
        ).fillna(0.0)

    if {"page_views", "sessions"}.issubset(out.columns):
        out["pages_per_session"] = (out["page_views"] / out["sessions"].replace(0, pd.NA)).fillna(0.0)

    return out


def one_hot_encode(df: pd.DataFrame, categorical_cols: list[str]) -> pd.DataFrame:
    """Apply deterministic one-hot encoding with sorted output columns."""
    cols = [col for col in categorical_cols if col in df.columns]
    out = pd.get_dummies(df, columns=cols, dummy_na=True)
    return out.reindex(sorted(out.columns), axis=1)


def build_feature_table(
    sales_df: pd.DataFrame,
    web_traffic_df: pd.DataFrame | None = None,
    date_col: str = "Date",
    target_col: str = "Revenue",
    lag_values: tuple[int, ...] = (1, 7, 14, 28),
    rolling_windows: tuple[int, ...] = (7, 14, 28),
) -> pd.DataFrame:
    """Build a deterministic feature table for daily forecasting tasks."""
    if date_col not in sales_df.columns:
        raise ValueError(f"Missing required date column: {date_col}")
    if target_col not in sales_df.columns:
        raise ValueError(f"Missing required target column: {target_col}")

    features = sales_df.copy()
    features[date_col] = pd.to_datetime(features[date_col], errors="coerce")
    features = features.sort_values(date_col).reset_index(drop=True)

    if web_traffic_df is not None and {"date", "sessions", "unique_visitors", "page_views"}.issubset(web_traffic_df.columns):
        web_daily = web_traffic_df.copy()
        web_daily["date"] = pd.to_datetime(web_daily["date"], errors="coerce")
        web_daily = web_daily.groupby("date", as_index=False)[["sessions", "unique_visitors", "page_views"]].sum()
        features = features.merge(web_daily, how="left", left_on=date_col, right_on="date").drop(columns=["date"], errors="ignore")

    features = add_time_features(features, date_col=date_col)
    features = add_ratio_features(features)
    features = add_lag_features(features, target_col=target_col, lags=lag_values, date_col=date_col)
    features = add_rolling_features(features, target_col=target_col, windows=rolling_windows, date_col=date_col)

    return features
