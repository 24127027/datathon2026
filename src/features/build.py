from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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


def list_feature_columns(
    df: pd.DataFrame,
    date_col: str = "date",
    target_col: str = "Revenue",
) -> dict[str, list[str]]:
    """List feature columns by common forecasting groups for notebook inspection."""
    excluded = {date_col, target_col}
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


def build_daily_one_row_per_day(
    orders_df: pd.DataFrame,
    order_items_df: pd.DataFrame,
    web_traffic_df: pd.DataFrame,
    inventory_df: pd.DataFrame,
    sales_df: pd.DataFrame,
    sales_date_col: str = "date",
) -> pd.DataFrame:
    """Build a single daily table with exactly one row per calendar day.

    This function is designed to match the output schema produced by
    `load_orders`, `load_order_items`, `load_web_traffic`, `load_inventory`, and `load_sales`.
    """
    required_orders = {"order_id", "order_date", "customer_id"}
    missing_orders = required_orders.difference(orders_df.columns)
    if missing_orders:
        raise ValueError(f"orders_df missing required columns: {sorted(missing_orders)}")

    required_items = {"order_id", "quantity", "unit_price", "discount_amount"}
    missing_items = required_items.difference(order_items_df.columns)
    if missing_items:
        raise ValueError(f"order_items_df missing required columns: {sorted(missing_items)}")

    required_web = {"date", "sessions", "unique_visitors", "page_views"}
    missing_web = required_web.difference(web_traffic_df.columns)
    if missing_web:
        raise ValueError(f"web_traffic_df missing required columns: {sorted(missing_web)}")

    required_inventory = {"snapshot_date", "stock_on_hand", "units_received", "units_sold"}
    missing_inventory = required_inventory.difference(inventory_df.columns)
    if missing_inventory:
        raise ValueError(f"inventory_df missing required columns: {sorted(missing_inventory)}")

    required_sales = {sales_date_col, "Revenue", "COGS"}
    missing_sales = required_sales.difference(sales_df.columns)
    if missing_sales:
        raise ValueError(f"sales_df missing required columns: {sorted(missing_sales)}")

    def _existing(cols: Iterable[str], frame: pd.DataFrame) -> list[str]:
        return [col for col in cols if col in frame.columns]

    sales_daily = sales_df.copy()
    sales_daily["date"] = pd.to_datetime(sales_daily[sales_date_col], errors="coerce")
    sales_daily = (
        sales_daily.dropna(subset=["date"])
        .groupby("date", as_index=False)[["Revenue", "COGS"]]
        .sum()
        .sort_values("date")
    )

    orders_tmp = orders_df.copy()
    orders_tmp["order_date"] = pd.to_datetime(orders_tmp["order_date"], errors="coerce")
    orders_tmp = orders_tmp.dropna(subset=["order_date"])
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
        orders_tmp["shipping_lead_days"] = (
            orders_tmp["delivery_date"] - orders_tmp["ship_date"]
        ).dt.days
    else:
        orders_tmp["delivered_flag"] = 0
        orders_tmp["shipping_lead_days"] = pd.NA

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

    orders_daily = orders_tmp.groupby("order_date", as_index=False).agg(**orders_agg)
    orders_daily = orders_daily.rename(columns={"order_date": "date"})
    orders_daily["return_rate_orders"] = (
        orders_daily["returned_orders"] / orders_daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    orders_daily["cancel_rate_orders"] = (
        orders_daily["cancelled_orders"] / orders_daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)

    items_tmp = order_items_df.copy()
    items_tmp["line_gross"] = items_tmp["quantity"] * items_tmp["unit_price"]
    items_tmp["line_net"] = items_tmp["line_gross"] - items_tmp["discount_amount"]

    order_dates = orders_tmp[["order_id", "order_date"]].drop_duplicates("order_id")
    items_with_dates = items_tmp.merge(order_dates, on="order_id", how="left")
    items_with_dates = items_with_dates.dropna(subset=["order_date"])

    items_agg: dict[str, tuple[str, str]] = {
        "items_count": ("order_id", "size"),
        "quantity_sold": ("quantity", "sum"),
        "gross_merch_value": ("line_gross", "sum"),
        "discount_total": ("discount_amount", "sum"),
        "net_merch_value": ("line_net", "sum"),
        "unique_products_sold": ("product_id", "nunique"),
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

    items_daily = items_with_dates.groupby("order_date", as_index=False).agg(**items_agg)
    items_daily = items_daily.rename(columns={"order_date": "date"})

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

    date_series: list[pd.Series] = [sales_daily["date"], orders_daily["date"], web_daily["date"], inventory_daily["date"]]
    if not items_daily.empty:
        date_series.append(items_daily["date"])
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
        .merge(web_daily, on="date", how="left")
        .merge(inventory_daily, on="date", how="left")
        .sort_values("date")
        .reset_index(drop=True)
    )

    for col in daily.columns:
        if col == "date":
            continue
        if pd.api.types.is_numeric_dtype(daily[col]):
            daily[col] = daily[col].fillna(0.0)

    # Ratios are computed after all numeric missing values have been filled.
    for needed_col in [
        "orders_count",
        "sessions",
        "quantity_sold",
        "gross_merch_value",
        "discount_total",
        "refund_amount_total",
    ]:
        if needed_col not in daily.columns:
            daily[needed_col] = 0.0

    daily["gross_margin"] = daily["Revenue"] - daily["COGS"]
    daily["gross_margin_ratio"] = (
        daily["gross_margin"] / daily["Revenue"].replace(0, pd.NA)
    ).fillna(0.0)
    daily["aov_revenue"] = (
        daily["Revenue"] / daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    daily["conversion_proxy"] = (
        daily["orders_count"] / daily["sessions"].replace(0, pd.NA)
    ).fillna(0.0)
    daily["units_per_order"] = (
        daily["quantity_sold"] / daily["orders_count"].replace(0, pd.NA)
    ).fillna(0.0)
    daily["discount_rate"] = (
        daily["discount_total"] / daily["gross_merch_value"].replace(0, pd.NA)
    ).fillna(0.0)
    daily["refund_rate"] = (
        daily["refund_amount_total"] / daily["net_merch_value"].replace(0, pd.NA)
    ).fillna(0.0)
    daily["pages_per_session"] = (
        daily["page_views"] / daily["sessions"].replace(0, pd.NA)
    ).fillna(0.0)
    daily["sessions_per_visitor"] = (
        daily["sessions"] / daily["unique_visitors"].replace(0, pd.NA)
    ).fillna(0.0)

    daily["date"] = pd.to_datetime(daily["date"])
    daily["year"] = daily["date"].dt.year
    daily["month"] = daily["date"].dt.month
    daily["day"] = daily["date"].dt.day
    daily["day_of_week"] = daily["date"].dt.dayofweek
    daily["day_of_year"] = daily["date"].dt.dayofyear
    daily["week_of_year"] = daily["date"].dt.isocalendar().week.astype("Int64")
    daily["is_weekend"] = (daily["day_of_week"] >= 5).astype(int)

    return daily

