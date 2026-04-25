from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
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
    """Apply deterministxic one-hot encoding with sorted output columns."""
    cols = [col for col in categorical_cols if col in df.columns]
    out = pd.get_dummies(df, columns=cols, dummy_na=True)
    return out.reindex(sorted(out.columns), axis=1)


def drop_columns(df: pd.DataFrame, columns: Iterable[str]) -> pd.DataFrame:
    """Drop multiple columns safely and return a new dataframe."""
    cols = [col for col in columns]
    return df.drop(columns=cols)


def add_column(
    df: pd.DataFrame,
    column_name: str,
    values: pd.Series | list[Any] | Any,
    overwrite: bool = True,
) -> pd.DataFrame:
    """Add a column to a dataframe with optional overwrite protection."""
    if column_name in df.columns and not overwrite:
        raise ValueError(f"Column '{column_name}' already exists and overwrite=False")

    out = df.copy()
    out[column_name] = values
    return out


def add_temporal_transforms(
    df: pd.DataFrame,
    date_col: str,
    lags: Iterable[int] = (1, 7, 14, 28),
    rolling_windows: Iterable[int] = (7, 14, 28),
    cols: Iterable[str] | None = None,
) -> pd.DataFrame:
    """Add lag/rolling and related leakage-safe temporal transforms for numeric columns.

    Args:
        cols: Optional whitelist of columns to apply transforms to.
              If None, all numeric columns (except date_col) are used.
    """
    if date_col not in df.columns:
        raise ValueError(f"Missing required date column: {date_col}")

    out = df.copy().sort_values(date_col).reset_index(drop=True)
    valid_lags = sorted({lag for lag in lags if lag > 0})
    valid_windows = sorted({window for window in rolling_windows if window > 1})
    if not valid_lags and not valid_windows:
        return out

    all_numeric = [
        col
        for col in out.columns
        if col != date_col and pd.api.types.is_numeric_dtype(out[col])
    ]
    if cols is not None:
        numeric_cols = [c for c in cols if c in out.columns and pd.api.types.is_numeric_dtype(out[c])]
    else:
        numeric_cols = all_numeric
    if not numeric_cols:
        return out

    generated: dict[str, pd.Series] = {}
    for col in numeric_cols:
        series = out[col]
        for lag in valid_lags:
            generated[f"{col}_lag_{lag}"] = series.shift(lag)

        shifted = series.shift(1)
        for window in valid_windows:
            generated[f"{col}_roll_mean_{window}"] = shifted.rolling(window).mean()
            generated[f"{col}_roll_std_{window}"] = shifted.rolling(window).std()

        # day-over-day and week-over-week momentum (leak-safe: both use shift>=1)
        generated[f"{col}_diff_1"] = series.shift(1) - series.shift(2)
        generated[f"{col}_pct_change_1"] = (
            series.shift(1) / series.shift(2).replace(0, pd.NA)
        ) - 1.0

    if generated:
        out = pd.concat([out, pd.DataFrame(generated, index=out.index)], axis=1)
    return out


def list_feature_columns(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: list[str]= ["Revenue", "COGS"],
) -> dict[str, list[str]]:
    """List feature columns by common forecasting groups for notebook inspection."""
    excluded = {date_col, *target_col}
    feature_cols = [col for col in df.columns if col not in excluded]

    numeric = [col for col in feature_cols if pd.api.types.is_numeric_dtype(df[col])]
    categorical = [col for col in feature_cols if col not in numeric]

    return {
        "all_features": feature_cols,
        "numeric_features": numeric,
        "categorical_features": categorical,
        "lag_features": [col for col in feature_cols if "_lag_" in col],
        "rolling_features": [col for col in feature_cols if "_roll_" in col],
        "time_features": [
            col
            for col in feature_cols
            if col
            in {
                "year",
                "month",
                "day",
                "day_of_week",
                "day_of_year",
                "week_of_year",
                "is_month_end",
                "is_month_start",
                "is_weekend",
            }
        ],
        "ratio_features": [col for col in feature_cols if "_ratio" in col or "_per_" in col],
    }


def build_feature_matrix(
    sales_df: pd.DataFrame,
    web_traffic_df: pd.DataFrame | None = None,
    date_col: str = "date",
    target_col: str = "Revenue",
    extra_drop_cols: Iterable[str] = ("COGS",),
    lag_values: tuple[int, ...] = (1, 7, 14, 28),
    rolling_windows: tuple[int, ...] = (7, 14, 28),
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Build features and return X, y, and processed feature dataframe for notebooks."""
    feature_df = build_feature_table(
        sales_df=sales_df,
        web_traffic_df=web_traffic_df,
        date_col=date_col,
        target_col=target_col,
        lag_values=lag_values,
        rolling_windows=rolling_windows,
    )
    feature_df = feature_df.dropna().reset_index(drop=True)

    drop_cols = [date_col, target_col, *list(extra_drop_cols)]
    X = drop_columns(feature_df, drop_cols)
    y = feature_df[target_col].copy()
    return X, y, feature_df


def build_feature_table(
    sales_df: pd.DataFrame,
    web_traffic_df: pd.DataFrame | None = None,
    date_col: str = "date",
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


def _require_cols(df: pd.DataFrame, cols: set[str], label: str) -> None:
    """Raise ValueError listing any columns missing from df."""
    missing = cols.difference(df.columns)
    if missing:
        raise ValueError(f"{label} missing columns: {sorted(missing)}")


def _safe_ratio(frame: pd.DataFrame, numerator: str, denominator: str) -> pd.Series:
    return (frame[numerator] / frame[denominator].replace(0, pd.NA)).fillna(0.0)


def _category_share_daily(
    frame: pd.DataFrame,
    date_col: str,
    cat_col: str,
    prefix: str,
    max_unique: int = 8,
) -> pd.DataFrame:
    """Return a date-indexed table of daily share per top-N category value."""
    if cat_col not in frame.columns:
        return pd.DataFrame(columns=["date"])
    tmp = frame[[date_col, cat_col]].dropna(subset=[date_col]).copy()
    if tmp.empty:
        return pd.DataFrame(columns=["date"])

    def _norm(v: Any) -> str:
        t = str(v).strip().lower()
        if t in {"", "nan", "none", "<na>"}:
            return "unknown"
        n = "".join(ch if ch.isalnum() else "_" for ch in t)
        while "__" in n:
            n = n.replace("__", "_")
        return n.strip("_") or "unknown"

    tmp[cat_col] = tmp[cat_col].map(_norm)
    top = tmp[cat_col].value_counts().head(max_unique).index
    if top.empty:
        return pd.DataFrame(columns=["date"])
    tmp[cat_col] = tmp[cat_col].where(tmp[cat_col].isin(top), "other")
    counts = tmp.groupby([date_col, cat_col]).size().unstack(fill_value=0).reset_index()
    val_cols = [c for c in counts.columns if c != date_col]
    total = counts[val_cols].sum(axis=1).replace(0, pd.NA)
    for c in val_cols:
        counts[f"{prefix}_share_{c}"] = (counts[c] / total).fillna(0.0)
    keep = [date_col, *[f"{prefix}_share_{c}" for c in val_cols]]
    return counts[keep].rename(columns={date_col: "date"})


def _merge_mix(parts: list[pd.DataFrame]) -> pd.DataFrame:
    """Outer-merge a list of category-share DataFrames on 'date'."""
    out: pd.DataFrame = pd.DataFrame(columns=["date"])
    for part in parts:
        if not part.empty:
            out = part if out.empty else out.merge(part, on="date", how="outer")
    return out


# Ratio feature definitions: (output_col, numerator_col, denominator_col).
# All columns must exist before _add_business_features is called.
_RATIO_PAIRS: list[tuple[str, str, str]] = [
    ("gross_margin_ratio",       "gross_margin",           "Revenue"),
    ("aov_revenue",              "Revenue",                "orders_count"),
    ("units_per_order",          "quantity_sold",          "orders_count"),
    ("discount_rate",            "discount_total",         "gross_merch_value"),
    ("effective_discount_rate",  "discount_total",         "list_merch_value"),
    ("refund_rate",              "refund_amount_total",    "net_merch_value"),
    ("returns_event_rate",       "returned_items_events",  "orders_count"),
    ("returns_quantity_rate",    "returned_items_quantity","quantity_sold"),
    ("refund_to_revenue_rate",   "returned_refund_value",  "Revenue"),
    ("pages_per_session",        "page_views",             "sessions"),
    ("sessions_per_visitor",     "sessions",               "unique_visitors"),
    ("pages_per_visitor",        "page_views",             "unique_visitors"),
    ("revenue_per_session",      "Revenue",                "sessions"),
    ("conversion_rate",          "orders_count",           "sessions"),
    ("delivery_throughput_rate", "delivered_orders_count", "orders_count"),
    ("checkout_efficiency",      "delivered_orders",       "orders_count"),
    ("aov_true",                 "net_merch_value",        "delivered_orders"),
    ("avg_item_price",           "net_merch_value",        "quantity_sold"),
    ("basket_size",              "quantity_sold",          "delivered_orders"),
    ("orders_per_customer",      "orders_count",           "customers_count"),
    ("product_diversity",        "unique_products_sold",   "quantity_sold"),
    ("refund_per_order",         "refund_amount_total",    "orders_count"),
    ("return_items_ratio",       "return_quantity_total",  "quantity_sold"),
    ("discount_per_order",       "discount_total",         "orders_count"),
    ("discount_efficiency",      "net_merch_value",        "discount_total"),
    ("stackable_promo_rate",     "stackable_promo_lines",  "items_count"),
    ("demand_supply_ratio",      "quantity_sold",          "inv_stock_on_hand"),
    ("inventory_pressure",       "inv_units_sold",         "inv_units_received"),
    ("inventory_turnover_proxy", "items_cogs_total",       "inv_stock_on_hand"),
    ("review_coverage_rate",     "reviews_count",          "delivered_orders_count"),
]

# Columns that must exist before _add_business_features; defaulted to 0 if absent.
_REQUIRED_ZERO_DEFAULTS: list[str] = [
    "orders_count", "customers_count", "delivered_orders", "cancelled_orders", "returned_orders",
    "sessions", "unique_visitors", "page_views", "quantity_sold",
    "gross_merch_value", "discount_total", "net_merch_value", "list_merch_value",
    "refund_amount_total", "return_quantity_total", "rating_mean",
    "promo_1_usage_count", "promo_2_usage_count", "bounce_rate", "avg_session_duration_sec",
    "unique_products_sold", "inv_stockout_products", "inv_stock_on_hand",
    "inv_units_sold", "inv_units_received", "items_cogs_total",
    "returned_items_events", "returned_items_quantity", "returned_refund_value",
    "reviews_count", "reviews_rating_mean", "shipped_orders_count", "delivered_orders_count",
    "same_day_ship_orders", "same_day_delivery_orders", "weekend_orders",
    "promo_1_active_lines", "promo_2_active_lines", "stackable_promo_lines",
]


def _add_business_features(daily: pd.DataFrame) -> pd.DataFrame:
    """Compute all engineered business metrics from the merged daily table."""
    out = daily.copy()
    out["gross_margin"] = out["Revenue"] - out["COGS"]
    for new_col, num, den in _RATIO_PAIRS:
        out[new_col] = _safe_ratio(out, num, den)

    out["returning_users"] = (out["sessions"] - out["unique_visitors"]).clip(lower=0)
    out["returning_rate"] = _safe_ratio(out, "returning_users", "sessions")
    # Guard engagement metrics on days with actual traffic to avoid
    # misleading values when bounce_rate/avg_session_duration_sec are
    # zero-filled due to missing source data.
    _has_traffic = out["sessions"] > 0
    out["engagement_score"] = (
        out["pages_per_session"]
        * (1 - out["bounce_rate"])
        * np.log1p(out["avg_session_duration_sec"].clip(lower=0))
    ).where(_has_traffic, 0.0).fillna(0.0)
    out["quality_sessions"] = (
        (out["sessions"] * (1 - out["bounce_rate"])).where(_has_traffic, 0.0).fillna(0.0)
    )
    out["loyal_engagement"] = (out["returning_rate"] * out["engagement_score"]).fillna(0.0)
    out["order_loss_rate"] = (
        (out["cancelled_orders"] + out["returned_orders"])
        / out["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    # Prefer reviews_rating_mean (by review date) over rating_mean (by order date)
    # when both are available, as it reflects post-purchase satisfaction.
    _rating_col = "reviews_rating_mean" if "reviews_rating_mean" in out.columns else "rating_mean"
    out["satisfaction_demand"] = (out["delivered_orders"] * out[_rating_col]).fillna(0.0)
    out["promo_intensity"] = (
        (out["promo_1_usage_count"] + out["promo_2_usage_count"])
        / out["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    out["promo_active_line_rate"] = (
        (out["promo_1_active_lines"] + out["promo_2_active_lines"])
        / out["items_count"].replace(0, pd.NA)
    ).fillna(0.0)
    out["engaged_sessions"] = (out["sessions"] * (1 - out["bounce_rate"])).fillna(0.0)
    out["stock_availability"] = (1 - _safe_ratio(out, "inv_stockout_products", "unique_products_sold")).fillna(0.0)
    out["inventory_gap"] = (out["inv_units_received"] - out["quantity_sold"]).fillna(0.0)
    out["revenue_driver"] = (
        out["sessions"] * out["conversion_rate"] * out["aov_true"]
    ).fillna(0.0)
    out["quality_revenue"] = (
        out["net_merch_value"] * (1 - out["refund_rate"]) * (1 - out["cancel_rate_orders"])
    ).fillna(0.0)
    out["demand_strength"] = (
        out["quantity_sold"] * (1 - out["return_items_ratio"])
    ).fillna(0.0)
    out["review_sentiment_score"] = (
        out["reviews_rating_mean"] * np.log1p(out["reviews_count"].clip(lower=0))
    ).fillna(0.0)
    out["revenue_to_cogs_items_gap"] = (out["Revenue"] - out["items_cogs_total"]).fillna(0.0)
    return out


def _add_calendar_cyclical_features(daily: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    out = daily.copy()
    out[date_col] = pd.to_datetime(out[date_col])
    out["year"] = out[date_col].dt.year
    out["month"] = out[date_col].dt.month
    out["day"] = out[date_col].dt.day
    out["day_of_week"] = out[date_col].dt.dayofweek
    out["day_of_year"] = out[date_col].dt.dayofyear
    out["week_of_year"] = out[date_col].dt.isocalendar().week.astype("Int64")
    out["is_weekend"] = (out["day_of_week"] >= 5).astype(int)

    out["dow_sin"] = np.sin(2 * np.pi * out["day_of_week"] / 7)
    out["dow_cos"] = np.cos(2 * np.pi * out["day_of_week"] / 7)
    out["month_sin"] = np.sin(2 * np.pi * (out["month"] - 1) / 12)
    out["month_cos"] = np.cos(2 * np.pi * (out["month"] - 1) / 12)
    out["doy_sin"] = np.sin(2 * np.pi * (out["day_of_year"] - 1) / 365.25)
    out["doy_cos"] = np.cos(2 * np.pi * (out["day_of_year"] - 1) / 365.25)

    # Biennial cycle features — captures the systematic Revenue anomaly
    # in odd-year Augusts where COGS > Revenue (margin -32% to -42%).
    out["is_odd_year"] = (out["year"] % 2).astype(int)
    out["is_august_odd"] = ((out["month"] == 8) & (out["is_odd_year"] == 1)).astype(int)
    out["is_q3_odd"] = ((out["month"].isin([7, 8, 9])) & (out["is_odd_year"] == 1)).astype(int)
    out["month_x_odd_year"] = out["month"] * out["is_odd_year"]
    # Fourier encoding for the 2-year cycle
    day_idx = (out[date_col] - out[date_col].min()).dt.days
    out["biennial_sin"] = np.sin(2 * np.pi * day_idx / 730.5)
    out["biennial_cos"] = np.cos(2 * np.pi * day_idx / 730.5)

    return out


def _add_yoy_lag_features(
    daily: pd.DataFrame,
    date_col: str = "date",
    cols: list[str] | None = None,
) -> pd.DataFrame:
    """Add year-over-year lag features (365 and 730 days) for specified columns.

    These features provide same-period-last-year and same-period-two-years-ago
    context, which is critical for capturing biennial Revenue cycles.
    Experimentally proven to reduce Revenue RMSE by ~27%.
    """
    out = daily.copy().sort_values(date_col).reset_index(drop=True)
    if cols is None:
        cols = ["Revenue", "COGS", "gross_margin", "orders_count"]

    for col in cols:
        if col not in out.columns:
            continue
        series = out[col]
        out[f"{col}_lag_365"] = series.shift(365)
        out[f"{col}_lag_730"] = series.shift(730)
        # YoY change ratio (2-year-ago baseline, leakage-safe)
        prev_2y = series.shift(730).replace(0, pd.NA)
        out[f"{col}_yoy_ratio_2y"] = (series.shift(365) / prev_2y).fillna(1.0)
    return out


def _add_regime_and_context_features(
    daily: pd.DataFrame, date_col: str = "date"
) -> pd.DataFrame:
    """Add features that capture the 2019 structural break and related context.

    Addresses three data anomalies:
    1. Business metrics dropped ~50% in 2019 while web traffic kept growing.
    2. Conversion rate collapsed from ~1% to ~0.35%.
    3. Month-end Revenue spikes (+55%) with lower margin.
    """
    out = daily.copy()

    # --- S1: Regime indicator ---
    # Clean break at 2019: orders/customers -53%, Revenue -42%, but AOV +23%.
    out["is_post_2019"] = (out["year"] >= 2019).astype(int)
    # Year index within each regime (0-based), so tree models can learn
    # intra-regime trends without confusing the two eras.
    out["regime_year_idx"] = out["year"].apply(
        lambda y: y - 2013 if y < 2019 else y - 2019
    )

    # --- S2: Web-traffic normalised business metrics ---
    # Sessions grow +8-10%/yr monotonically while business collapsed in 2019;
    # normalising removes the diverging trend and exposes the real signal.
    _sess = out["sessions"].replace(0, pd.NA)
    _vis  = out["unique_visitors"].replace(0, pd.NA)
    _pv   = out["page_views"].replace(0, pd.NA)
    out["orders_per_session"]   = (out["orders_count"]   / _sess).fillna(0.0)
    out["revenue_per_visitor"]  = (out["Revenue"]        / _vis).fillna(0.0)
    out["items_per_pageview"]   = (out["quantity_sold"]  / _pv).fillna(0.0)
    out["cogs_per_session"]     = (out["COGS"]           / _sess).fillna(0.0)
    out["customers_per_session"]= (out["customers_count"]/ _sess).fillna(0.0)

    # --- S4: Month-end effect ---
    # Last 5 days of each month show +55% Revenue but lower margin.
    _month_end_date = out[date_col] + pd.offsets.MonthEnd(0)
    out["days_to_month_end"] = (_month_end_date - out[date_col]).dt.days
    out["is_month_end_5d"] = (out["days_to_month_end"] <= 4).astype(int)

    # --- S6: Smoothed conversion rate ---
    # Raw conversion_rate is noisy; smoothed versions capture regime shift.
    _conv = out["conversion_rate"].shift(1)  # leak-safe
    out["conversion_rate_7d"]  = _conv.rolling(7,  min_periods=1).mean()
    out["conversion_rate_28d"] = _conv.rolling(28, min_periods=1).mean()

    # --- S8: AOV × Volume interaction ---
    # Revenue ≈ AOV × orders; decomposing helps because AOV rose +23%
    # while orders fell -53% across the 2019 break.
    out["aov_x_orders"] = (out["aov_revenue"] * out["orders_count"]).fillna(0.0)

    return out


_TOP_FEATURES_RANKED = [
    "gross_merch_value", "items_cogs_total", "unique_products_sold", "discount_total_lag_7", "quality_revenue",
    "list_merch_value", "orders_count", "payment_value_total", "inv_days_of_supply", "sessions_roll_std_7",
    "unique_visitors_roll_std_28", "refund_rate_lag_14", "refund_amount_total", "demand_strength", "item_color_share_blue",
    "review_sentiment_score", "refund_rate_lag_7", "gender_share_non_binary", "item_color_share_yellow", "gross_margin_lag_730",
    "product_diversity", "reviews_rating_mean_lag_14", "order_source_share_referral", "shipping_fee_total", "reviews_rating_mean_roll_std_7",
    "gross_margin_lag_28", "return_items_ratio_lag_1", "item_size_share_l", "stackable_promo_lines", "refund_rate_roll_mean_7",
    "item_color_share_other", "month_sin", "Revenue_lag_7", "discount_per_order", "gender_share_female",
    "reviews_rating_mean_roll_std_14", "cancelled_orders", "discount_total", "shipping_lead_days_mean", "acquisition_share_paid_search",
    "biennial_cos", "cancel_rate_orders", "delivered_orders", "reviews_rating_mean", "district_share_district_33",
    "net_merch_value_lag_1", "return_items_ratio_roll_std_7", "Revenue_roll_std_7", "Revenue_lag_28", "device_share_tablet",
    "bounce_rate", "cancelled_orders_roll_mean_7", "device_share_desktop", "day", "COGS_lag_28",
    "reviews_count_lag_14", "avg_session_duration_sec", "gender_share_male", "age_group_share_25_34", "delivered_orders_roll_std_7",
    "payment_method_share_credit_card", "order_source_share_paid_search", "cancelled_orders_diff_1", "rating_mean", "promo_intensity_roll_std_7",
    "item_color_share_silver", "day_of_year", "payment_method_share_bank_transfer", "COGS_lag_365", "item_segment_share_performance",
    "acquisition_share_referral", "acquisition_share_email_campaign", "unique_visitors_lag_28", "aov_revenue", "district_share_district_36",
    "returned_orders_lag_28", "delivery_throughput_rate", "reviews_count_lag_28", "item_color_share_orange", "COGS_lag_7",
    "acquisition_share_social_media", "gross_margin_roll_mean_7", "units_per_order", "same_day_ship_orders", "inv_fill_rate",
    "delivered_orders_pct_change_1", "orders_count_yoy_ratio_2y", "orders_count_lag_730", "return_items_ratio_lag_7", "installments_mean",
    "return_quantity_total", "item_size_share_xl", "reviews_rating_std", "orders_count_lag_365", "sessions_roll_std_14",
    "returned_orders_diff_1", "reviews_rating_mean_lag_28", "return_items_ratio_roll_std_14", "orders_count_roll_std_7", "district_share_district_37",
    "refund_rate_lag_1", "reviews_count_roll_std_28", "age_group_share_45_54", "district_share_district_34", "fulfillment_days_mean",
    "effective_discount_rate", "conversion_rate_7d", "sessions_lag_7", "stock_availability", "orders_count_diff_1",
    "returned_orders_roll_mean_7", "return_rate_orders", "region_share_central", "Revenue_roll_std_28", "Revenue_lag_365",
    "returned_orders_roll_std_28", "discount_total_diff_1", "discount_total_roll_std_28", "district_share_district_23", "sessions_roll_std_28",
    "device_share_mobile", "order_source_share_email_campaign", "COGS_diff_1", "age_group_share_55", "item_segment_share_standard",
    "return_items_ratio_lag_14", "item_category_share_outdoor", "inv_stock_on_hand_lag_28", "customer_tenure_days_mean", "refund_rate_diff_1",
    "gross_margin_diff_1", "item_segment_share_everyday", "delivered_orders_diff_1", "unique_visitors_diff_1", "same_day_ship_rate",
    "item_category_share_casual", "demand_strength_pct_change_1", "doy_sin", "net_merch_value_roll_mean_14", "shipped_orders_count",
    "reviews_rating_mean_lag_7", "cancelled_orders_roll_std_7", "sessions_lag_28", "age_group_share_35_44", "regime_year_idx",
    "Revenue_diff_1", "returned_orders_roll_std_7", "conversion_rate_28d", "item_size_share_s", "demand_strength_diff_1",
]

def build_daily_one_row_per_day(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    web_traffic_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    sales_date_col: str = "date",
    lag_values: tuple[int, ...] = (1, 7, 14, 28),
    rolling_windows: tuple[int, ...] = (7, 14, 28),
    keep_feature_count: int | None = None,
    drop_years: Iterable[int] | None = None,
) -> pd.DataFrame:
    """Build a single daily table with exactly one row per calendar day.

    This function is designed to match the output schema produced by
    `load_orders`, `load_order_items`, `load_web_traffic`, `load_inventory`, and `load_sales`.
    """
    _require_cols(orders_df,      {"order_id", "date", "customer_id"},                "orders_df")
    _require_cols(order_items_df, {"order_id", "quantity", "unit_price", "discount_amount"}, "order_items_df")
    _require_cols(web_traffic_df, {"date", "sessions", "unique_visitors", "page_views"},     "web_traffic_df")
    _require_cols(inventory_df,   {"snapshot_date", "stock_on_hand", "units_received", "units_sold"}, "inventory_df")
    _require_cols(sales_df,       {sales_date_col, "Revenue", "COGS"},                      "sales_df")

    sales_daily = sales_df.copy()
    sales_daily["date"] = pd.to_datetime(sales_daily[sales_date_col], errors="coerce")
    sales_daily = (
        sales_daily.dropna(subset=["date"])
        .groupby("date", as_index=False)[["Revenue", "COGS"]]
        .sum()
        .sort_values("date")
    )

    orders_tmp = orders_df.copy()
    orders_tmp["date"] = pd.to_datetime(orders_tmp["date"], errors="coerce")
    orders_tmp = orders_tmp.dropna(subset=["date"])

    if "signup_date" in orders_tmp.columns:
        orders_tmp["signup_date"] = pd.to_datetime(orders_tmp["signup_date"], errors="coerce")
        orders_tmp["customer_tenure_days"] = (
            orders_tmp["date"] - orders_tmp["signup_date"]
        ).dt.days.clip(lower=0)

    orders_tmp["order_weekend_flag"] = (orders_tmp["date"].dt.dayofweek >= 5).astype(int)

    if "order_status" in orders_tmp.columns:
        normalized_status = orders_tmp["order_status"].astype(str).str.lower().str.strip()
        orders_tmp["is_returned"] = (normalized_status == "returned").astype(int)
        orders_tmp["is_cancelled"] = normalized_status.isin({"cancelled", "canceled"}).astype(int)
    else:
        orders_tmp["is_returned"] = 0
        orders_tmp["is_cancelled"] = 0

    if {"ship_date", "delivery_date"}.issubset(orders_tmp.columns):
        orders_tmp["ship_date"] = pd.to_datetime(orders_tmp["ship_date"], errors="coerce")
        orders_tmp["delivery_date"] = pd.to_datetime(orders_tmp["delivery_date"], errors="coerce")
        orders_tmp["delivered_flag"] = orders_tmp["delivery_date"].notna().astype(int)
        orders_tmp["same_day_ship_flag"] = (
            orders_tmp["ship_date"] == orders_tmp["date"]
        ).astype(int)
        orders_tmp["same_day_delivery_flag"] = (
            orders_tmp["delivery_date"] == orders_tmp["date"]
        ).astype(int)
        orders_tmp["shipping_lead_days"] = (
            orders_tmp["delivery_date"] - orders_tmp["ship_date"]
        ).dt.days
        orders_tmp["fulfillment_days"] = (
            orders_tmp["delivery_date"] - orders_tmp["date"]
        ).dt.days
    else:
        orders_tmp["delivered_flag"] = 0
        orders_tmp["same_day_ship_flag"] = 0
        orders_tmp["same_day_delivery_flag"] = 0
        orders_tmp["shipping_lead_days"] = pd.NA
        orders_tmp["fulfillment_days"] = pd.NA

    orders_agg: dict[str, tuple[str, str]] = {
        "orders_count": ("order_id", "nunique"),
        "customers_count": ("customer_id", "nunique"),
        "returned_orders": ("is_returned", "sum"),
        "cancelled_orders": ("is_cancelled", "sum"),
        "delivered_orders": ("delivered_flag", "sum"),
    }
    if "payment_value" in orders_tmp.columns:
        orders_agg["payment_value_total"] = ("payment_value", "sum")
    if "installments" in orders_tmp.columns:
        orders_agg["installments_mean"] = ("installments", "mean")
    if "shipping_fee" in orders_tmp.columns:
        orders_agg["shipping_fee_total"] = ("shipping_fee", "sum")
    if "shipping_lead_days" in orders_tmp.columns:
        orders_agg["shipping_lead_days_mean"] = ("shipping_lead_days", "mean")
        orders_agg["shipping_lead_days_p90"] = ("shipping_lead_days", lambda s: s.quantile(0.9))
    if "fulfillment_days" in orders_tmp.columns:
        orders_agg["fulfillment_days_mean"] = ("fulfillment_days", "mean")
    if "payment_value" in orders_tmp.columns:
        orders_agg["payment_value_mean"] = ("payment_value", "mean")
        orders_agg["payment_value_std"] = ("payment_value", "std")
        orders_agg["payment_value_median"] = ("payment_value", "median")
    if "customer_tenure_days" in orders_tmp.columns:
        orders_agg["customer_tenure_days_mean"] = ("customer_tenure_days", "mean")
        orders_agg["customer_tenure_days_p90"] = ("customer_tenure_days", lambda s: s.quantile(0.9))

    orders_agg["weekend_orders"] = ("order_weekend_flag", "sum")
    orders_agg["same_day_ship_orders"] = ("same_day_ship_flag", "sum")
    orders_agg["same_day_delivery_orders"] = ("same_day_delivery_flag", "sum")

    orders_daily = orders_tmp.groupby("date", as_index=False).agg(**orders_agg)
    orders_daily = orders_daily.rename(columns={"date": "date"})
    orders_daily["return_rate_orders"] = (
        orders_daily["returned_orders"] / orders_daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    orders_daily["cancel_rate_orders"] = (
        orders_daily["cancelled_orders"] / orders_daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    orders_daily["weekend_order_share"] = (
        orders_daily["weekend_orders"] / orders_daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    orders_daily["same_day_ship_rate"] = (
        orders_daily["same_day_ship_orders"] / orders_daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    orders_daily["same_day_delivery_rate"] = (
        orders_daily["same_day_delivery_orders"] / orders_daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)

    order_mix_features = _merge_mix([
        _category_share_daily(orders_tmp, "date", col, pfx)
        for col, pfx in [
            ("payment_method",     "payment_method"),
            ("device_type",        "device"),
            ("order_source",       "order_source"),
            ("acquisition_channel","acquisition"),
            ("gender",             "gender"),
            ("age_group",          "age_group"),
            # Geography: order zip == customer zip (100% overlap verified),
            # so region/district from customer→geography join is sufficient.
            ("region",             "region"),
            ("district",           "district"),
        ]
    ])

    items_tmp = order_items_df.copy()
    items_tmp["line_gross"] = items_tmp["quantity"] * items_tmp["unit_price"]
    items_tmp["line_net"] = items_tmp["line_gross"] - items_tmp["discount_amount"]
    if "price" in items_tmp.columns:
        items_tmp["line_list_price"] = items_tmp["quantity"] * items_tmp["price"]
    else:
        items_tmp["line_list_price"] = items_tmp["line_gross"]

    if "cogs" in items_tmp.columns:
        items_tmp["line_cogs"] = items_tmp["quantity"] * items_tmp["cogs"]
    else:
        items_tmp["line_cogs"] = 0.0

    dates = orders_tmp[["order_id", "date"]].drop_duplicates("order_id")
    items_with_dates = items_tmp.merge(dates, on="order_id", how="left")
    items_with_dates = items_with_dates.dropna(subset=["date"])

    for promo_date_col in ["start_date_promo_1", "end_date_promo_1", "start_date_promo_2", "end_date_promo_2"]:
        if promo_date_col in items_with_dates.columns:
            items_with_dates[promo_date_col] = pd.to_datetime(items_with_dates[promo_date_col], errors="coerce")

    if {"promo_id", "start_date_promo_1", "end_date_promo_1"}.issubset(items_with_dates.columns):
        items_with_dates["promo_1_active_flag"] = (
            items_with_dates["promo_id"].notna()
            & (items_with_dates["date"] >= items_with_dates["start_date_promo_1"])
            & (items_with_dates["date"] <= items_with_dates["end_date_promo_1"])
        ).astype(int)
    else:
        items_with_dates["promo_1_active_flag"] = 0

    if {"promo_id_2", "start_date_promo_2", "end_date_promo_2"}.issubset(items_with_dates.columns):
        items_with_dates["promo_2_active_flag"] = (
            items_with_dates["promo_id_2"].notna()
            & (items_with_dates["date"] >= items_with_dates["start_date_promo_2"])
            & (items_with_dates["date"] <= items_with_dates["end_date_promo_2"])
        ).astype(int)
    else:
        items_with_dates["promo_2_active_flag"] = 0

    if "stackable_flag_promo_1" in items_with_dates.columns:
        items_with_dates["stackable_flag_promo_1"] = items_with_dates["stackable_flag_promo_1"].fillna(0).astype(float)
    else:
        items_with_dates["stackable_flag_promo_1"] = 0.0
    if "stackable_flag_promo_2" in items_with_dates.columns:
        items_with_dates["stackable_flag_promo_2"] = items_with_dates["stackable_flag_promo_2"].fillna(0).astype(float)
    else:
        items_with_dates["stackable_flag_promo_2"] = 0.0
    # A line item is stackable if *either* promo slot is stackable.
    items_with_dates["_any_stackable"] = (
        (items_with_dates["stackable_flag_promo_1"] > 0)
        | (items_with_dates["stackable_flag_promo_2"] > 0)
    ).astype(float)

    items_agg: dict[str, tuple[str, str]] = {
        "items_count": ("order_id", "size"),
        "quantity_sold": ("quantity", "sum"),
        "gross_merch_value": ("line_gross", "sum"),
        "discount_total": ("discount_amount", "sum"),
        "net_merch_value": ("line_net", "sum"),
        "list_merch_value": ("line_list_price", "sum"),
        "items_cogs_total": ("line_cogs", "sum"),
        "unique_products_sold": ("product_id", "nunique"),
        "unit_price_mean": ("unit_price", "mean"),
        "unit_price_std": ("unit_price", "std"),
        "promo_1_active_lines": ("promo_1_active_flag", "sum"),
        "promo_2_active_lines": ("promo_2_active_flag", "sum"),
        "stackable_promo_lines": ("_any_stackable", "sum"),  # counts lines where promo_1 OR promo_2 is stackable
    }
    if "refund_amount" in items_with_dates.columns:
        items_agg["refund_amount_total"] = ("refund_amount", "sum")
    if "return_quantity" in items_with_dates.columns:
        items_agg["return_quantity_total"] = ("return_quantity", "sum")
    if "rating" in items_with_dates.columns:
        items_agg["rating_mean"] = ("rating", "mean")
    if "promo_id" in items_with_dates.columns:
        items_agg["promo_1_usage_count"] = ("promo_id", lambda s: s.notna().sum())
    if "promo_id_2" in items_with_dates.columns:
        items_agg["promo_2_usage_count"] = ("promo_id_2", lambda s: s.notna().sum())
    if "discount_value_promo_1" in items_with_dates.columns:
        items_agg["promo_1_discount_value_mean"] = ("discount_value_promo_1", "mean")
    if "discount_value_promo_2" in items_with_dates.columns:
        items_agg["promo_2_discount_value_mean"] = ("discount_value_promo_2", "mean")
    if "min_order_value_promo_1" in items_with_dates.columns:
        items_agg["promo_1_min_order_value_mean"] = ("min_order_value_promo_1", "mean")
    if "min_order_value_promo_2" in items_with_dates.columns:
        items_agg["promo_2_min_order_value_mean"] = ("min_order_value_promo_2", "mean")

    items_daily = items_with_dates.groupby("date", as_index=False).agg(**items_agg)
    items_daily = items_daily.rename(columns={"date": "date"})

    item_mix_features = _merge_mix([
        _category_share_daily(items_with_dates, "date", col, pfx)
        for col, pfx in [
            ("category",            "item_category"),
            ("segment",             "item_segment"),
            ("size",                "item_size"),
            ("color",               "item_color"),
            ("promo_type_promo_1",  "promo_type_1"),
            ("promo_type_promo_2",  "promo_type_2"),
            ("promo_channel_promo_1","promo_channel_1"),
            ("promo_channel_promo_2","promo_channel_2"),
        ]
    ])

    returns_daily = pd.DataFrame(columns=["date"])
    if {"return_date", "refund_amount", "return_quantity"}.issubset(items_with_dates.columns):
        returns_tmp = items_with_dates.copy()
        returns_tmp["return_date"] = pd.to_datetime(returns_tmp["return_date"], errors="coerce")
        returns_tmp = returns_tmp.dropna(subset=["return_date"])
        if not returns_tmp.empty:
            returns_daily = (
                returns_tmp.groupby("return_date", as_index=False)
                .agg(
                    returned_items_events=("order_id", "size"),
                    returned_items_quantity=("return_quantity", "sum"),
                    returned_refund_value=("refund_amount", "sum"),
                )
                .rename(columns={"return_date": "date"})
            )

    reviews_daily = pd.DataFrame(columns=["date"])
    if {"review_date", "rating"}.issubset(items_with_dates.columns):
        reviews_tmp = items_with_dates.copy()
        reviews_tmp["review_date"] = pd.to_datetime(reviews_tmp["review_date"], errors="coerce")
        reviews_tmp = reviews_tmp.dropna(subset=["review_date"])
        if not reviews_tmp.empty:
            reviews_daily = (
                reviews_tmp.groupby("review_date", as_index=False)
                .agg(
                    reviews_count=("order_id", "size"),
                    reviews_rating_mean=("rating", "mean"),
                    reviews_rating_std=("rating", "std"),
                )
                .rename(columns={"review_date": "date"})
            )

    web_tmp = web_traffic_df.copy()
    web_tmp["date"] = pd.to_datetime(web_tmp["date"], errors="coerce")
    web_tmp = web_tmp.dropna(subset=["date"])
    web_agg: dict[str, tuple[str, str]] = {
        "sessions": ("sessions", "sum"),
        "unique_visitors": ("unique_visitors", "sum"),
        "page_views": ("page_views", "sum"),
    }
    for optional_col in ["bounce_rate", "avg_session_duration_sec"]:
        if optional_col in web_tmp.columns:
            web_agg[optional_col] = (optional_col, "mean")
    web_daily = web_tmp.groupby("date", as_index=False).agg(**web_agg)

    web_source_mix = _category_share_daily(web_tmp, "date", "traffic_source", "traffic_source")

    inv_tmp = inventory_df.copy()
    inv_tmp["snapshot_date"] = pd.to_datetime(inv_tmp["snapshot_date"], errors="coerce")
    inv_tmp = inv_tmp.dropna(subset=["snapshot_date"])
    inventory_agg: dict[str, tuple[str, str]] = {
        "inv_stock_on_hand": ("stock_on_hand", "sum"),
        "inv_units_received": ("units_received", "sum"),
        "inv_units_sold": ("units_sold", "sum"),
    }
    optional_inventory_sums = {
        "stockout_days": "inv_stockout_days",
        "stockout_flag": "inv_stockout_products",
        "overstock_flag": "inv_overstock_products",
        "reorder_flag": "inv_reorder_products",
    }
    optional_inventory_means = {
        "days_of_supply": "inv_days_of_supply",
        "fill_rate": "inv_fill_rate",
        "sell_through_rate": "inv_sell_through_rate",
    }
    for source_col, out_col in optional_inventory_sums.items():
        if source_col in inv_tmp.columns:
            inventory_agg[out_col] = (source_col, "sum")
    for source_col, out_col in optional_inventory_means.items():
        if source_col in inv_tmp.columns:
            inventory_agg[out_col] = (source_col, "mean")

    inventory_daily = inv_tmp.groupby("snapshot_date", as_index=False).agg(**inventory_agg)
    inventory_daily = inventory_daily.rename(columns={"snapshot_date": "date"})

    ship_daily = pd.DataFrame(columns=["date"])
    if "ship_date" in orders_tmp.columns:
        ship_daily_tmp = orders_tmp.dropna(subset=["ship_date"])
        if not ship_daily_tmp.empty:
            ship_daily = (
                ship_daily_tmp.groupby("ship_date", as_index=False)
                .agg(
                    shipped_orders_count=("order_id", "nunique"),
                    shipped_fee_total=("shipping_fee", "sum") if "shipping_fee" in ship_daily_tmp.columns else ("order_id", "size"),
                )
                .rename(columns={"ship_date": "date"})
            )

    delivery_daily = pd.DataFrame(columns=["date"])
    if "delivery_date" in orders_tmp.columns:
        delivery_daily_tmp = orders_tmp.dropna(subset=["delivery_date"])
        if not delivery_daily_tmp.empty:
            delivery_daily = (
                delivery_daily_tmp.groupby("delivery_date", as_index=False)
                .agg(
                    delivered_orders_count=("order_id", "nunique"),
                )
                .rename(columns={"delivery_date": "date"})
            )

    date_series: list[pd.Series] = [sales_daily["date"], orders_daily["date"], web_daily["date"], inventory_daily["date"]]
    if not items_daily.empty:
        date_series.append(items_daily["date"])
    if not returns_daily.empty:
        date_series.append(returns_daily["date"])
    if not reviews_daily.empty:
        date_series.append(reviews_daily["date"])
    if not ship_daily.empty:
        date_series.append(ship_daily["date"])
    if not delivery_daily.empty:
        date_series.append(delivery_daily["date"])
    all_dates = pd.concat(date_series, ignore_index=True).dropna()
    if all_dates.empty:
        raise ValueError("No valid dates found across all inputs")

    base_calendar = pd.DataFrame(
        {
            "date": pd.date_range(
                all_dates.min(),
                all_dates.max(),
                freq="D",
            )
        }
    )

    daily = (
        base_calendar.merge(sales_daily, on="date", how="left")
        .merge(orders_daily, on="date", how="left")
        .merge(items_daily, on="date", how="left")
        .merge(order_mix_features, on="date", how="left")
        .merge(item_mix_features, on="date", how="left")
        .merge(returns_daily, on="date", how="left")
        .merge(reviews_daily, on="date", how="left")
        .merge(ship_daily, on="date", how="left")
        .merge(delivery_daily, on="date", how="left")
        .merge(web_daily, on="date", how="left")
        .merge(web_source_mix, on="date", how="left")
        .merge(inventory_daily, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    inventory_state_cols = [
        col for col in daily.columns if col.startswith("inv_")
    ]
    if inventory_state_cols:
        daily[inventory_state_cols] = daily[inventory_state_cols].ffill()

    numeric_cols = daily.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 0:
        daily[numeric_cols] = daily[numeric_cols].fillna(0.0)

    # Ratios are computed after all numeric missing values have been filled.
    for col in _REQUIRED_ZERO_DEFAULTS:
        if col not in daily.columns:
            daily[col] = 0.0

    # De-fragment after many merge operations before adding many engineered columns.
    daily = daily.copy()
    daily = _add_business_features(daily)
    daily = _add_calendar_cyclical_features(daily, date_col="date")

    # Whitelist of meaningful columns for temporal transforms.
    # Keeping this list small avoids feature explosion from applying
    # lag/rolling to every engineered ratio, cyclical, and share column.
    # Per-column cost: 4 lags + 3 windows x 2 stats (mean/std) + 2 mom = 12 features.
    _TEMPORAL_WHITELIST: list[str] = [
        # Core P&L
        "Revenue", "COGS", "gross_margin",
        # Order volume & fulfilment
        "orders_count", "quantity_sold", "delivered_orders",
        "returned_orders", "cancelled_orders",
        # Merchandise value
        "net_merch_value", "discount_total",
        # Web traffic
        "sessions", "unique_visitors",
        # Key derived KPIs (already ratio-form — informative lagged baselines)
        "aov_true", "conversion_rate", "gross_margin_ratio",
        "refund_rate", "return_items_ratio", "promo_intensity",
        "engagement_score", "demand_strength",
        # Inventory health
        "inv_stock_on_hand", "inv_units_sold",
        # Social signal
        "reviews_count", "reviews_rating_mean",
    ]
    daily = add_temporal_transforms(
        df=daily,
        date_col="date",
        lags=lag_values,
        rolling_windows=rolling_windows,
        cols=_TEMPORAL_WHITELIST,
    )

    # Year-over-year lag features — experimentally proven to reduce
    # Revenue RMSE by 27% by providing same-period-last-year context.
    _YOY_COLS: list[str] = ["Revenue", "COGS", "gross_margin", "orders_count"]
    daily = _add_yoy_lag_features(daily, date_col="date", cols=_YOY_COLS)

    # Regime-aware features — addresses 2019 structural break,
    # web-traffic / business divergence, and month-end effects.
    daily = _add_regime_and_context_features(daily, date_col="date")

    # Prune features if requested to reduce training time
    if keep_feature_count is not None:
        keep_cols = ["date", "Revenue", "COGS"] + _TOP_FEATURES_RANKED[:keep_feature_count]
        # Only keep columns that actually exist in the dataframe
        valid_cols = [c for c in keep_cols if c in daily.columns]
        daily = daily[valid_cols]

    # Optional year-level exclusion for controlled ablation experiments.
    if drop_years:
        years_to_drop = {int(year) for year in drop_years}
        daily = daily[~daily["date"].dt.year.isin(years_to_drop)].reset_index(drop=True)

    return daily

