from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .loader import (
    load_customers,
    load_inventory,
    load_order_items,
    load_orders,
    load_products,
    load_promotions,
    load_sales,
    load_web_traffic,
)


EXPECTED_COLUMNS: dict[str, list[str]] = {
    "customers": [
        "customer_id",
        "zip",
        "city",
        "signup_date",
        "gender",
        "age_group",
        "acquisition_channel",
    ],
    "products": [
        "product_id",
        "product_name",
        "category",
        "segment",
        "size",
        "color",
        "price",
        "cogs",
    ],
    "promotions": [
        "promo_id",
        "promo_name",
        "promo_type",
        "discount_value",
        "start_date",
        "end_date",
        "applicable_category",
        "promo_channel",
        "stackable_flag",
        "min_order_value",
    ],
    "orders": [
        "order_id",
        "date",
        "customer_id",
        "zip",
        "order_status",
        "device_type",
        "order_source",
        "payment_value",
        "installments",
        "ship_date",
        "delivery_date",
        "shipping_fee",
    ],
    "order_items": [
        "order_id",
        "product_id",
        "quantity",
        "unit_price",
        "discount_amount",
        "promo_id",
        "promo_id_2",
        "return_id",
        "return_date",
        "return_reason",
        "return_quantity",
        "refund_amount",
        "review_id",
        "review_date",
        "rating",
        "review_title",
    ],
    "inventory": [
        "snapshot_date",
        "product_id",
        "stock_on_hand",
        "units_received",
        "units_sold",
        "stockout_days",
        "days_of_supply",
        "fill_rate",
        "stockout_flag",
        "overstock_flag",
        "reorder_flag",
        "sell_through_rate",
        "product_name",
        "category",
        "segment",
        "year",
        "month",
    ],
    "web_traffic": [
        "date",
        "sessions",
        "unique_visitors",
        "page_views",
        "bounce_rate",
        "avg_session_duration_sec",
        "traffic_source",
    ],
    "sales": ["date", "Revenue", "COGS"],
}

DATE_COLUMNS: dict[str, list[str]] = {
    "customers": ["signup_date"],
    "promotions": ["start_date", "end_date"],
    "orders": ["date", "ship_date", "delivery_date"],
    "order_items": ["return_date", "review_date"],
    "inventory": ["snapshot_date"],
    "web_traffic": ["date"],
    "sales": ["date"],
}

NON_NEGATIVE_COLUMNS: dict[str, list[str]] = {
    "products": ["price", "cogs"],
    "orders": ["payment_value", "installments", "shipping_fee"],
    "order_items": ["quantity", "unit_price", "discount_amount", "return_quantity", "refund_amount", "rating"],
    "inventory": [
        "stock_on_hand",
        "units_received",
        "units_sold",
        "stockout_days",
        "days_of_supply",
        "fill_rate",
        "sell_through_rate",
    ],
    "web_traffic": ["sessions", "unique_visitors", "page_views", "bounce_rate", "avg_session_duration_sec"],
    "sales": ["Revenue", "COGS"],
}

UNIQUE_KEY_COLUMNS: dict[str, list[str]] = {
    "customers": ["customer_id"],
    "products": ["product_id"],
    "promotions": ["promo_id"],
    "orders": ["order_id"],
}


@dataclass
class ValidationIssue:
    table: str
    check: str
    severity: str
    details: str


@dataclass
class ValidationResult:
    table: str
    passed: bool
    issues: list[ValidationIssue]


LOADER_MAP = {
    "customers": load_customers,
    "products": load_products,
    "promotions": load_promotions,
    "orders": load_orders,
    "order_items": load_order_items,
    "inventory": load_inventory,
    "web_traffic": load_web_traffic,
    "sales": load_sales,
}


def load_all_dataframes(data_root: str | Path = "data/datathon-2026-round-1") -> dict[str, pd.DataFrame]:
    return {name: loader(data_root) for name, loader in LOADER_MAP.items()}


def _missing_columns(df: pd.DataFrame, expected_columns: list[str]) -> list[str]:
    return [col for col in expected_columns if col not in df.columns]


def _date_parse_failures(df: pd.DataFrame, column: str) -> int:
    parsed = pd.to_datetime(df[column], errors="coerce")
    original_non_null = df[column].notna().sum()
    parsed_non_null = parsed.notna().sum()
    return int(original_non_null - parsed_non_null)


def _non_negative_failures(df: pd.DataFrame, columns: list[str]) -> dict[str, int]:
    failures: dict[str, int] = {}
    for col in columns:
        if col in df.columns and pd.api.types.is_numeric_dtype(df[col]):
            bad_count = int((df[col] < 0).sum())
            if bad_count > 0:
                failures[col] = bad_count
    return failures


def _null_rate_by_column(df: pd.DataFrame) -> dict[str, float]:
    if len(df) == 0:
        return {col: 0.0 for col in df.columns}
    null_rates = (df.isna().sum() / len(df) * 100.0).round(2)
    return {str(col): float(rate) for col, rate in null_rates.items()}


def validate_dataframe(table_name: str, df: pd.DataFrame) -> ValidationResult:
    issues: list[ValidationIssue] = []

    expected = EXPECTED_COLUMNS.get(table_name, [])
    if expected:
        missing = _missing_columns(df, expected)
        if missing:
            issues.append(
                ValidationIssue(
                    table=table_name,
                    check="expected_columns",
                    severity="error",
                    details=f"Missing columns: {missing}",
                )
            )

    key_columns = UNIQUE_KEY_COLUMNS.get(table_name)
    if key_columns and all(col in df.columns for col in key_columns):
        duplicated = int(df.duplicated(subset=key_columns).sum())
        if duplicated > 0:
            issues.append(
                ValidationIssue(
                    table=table_name,
                    check="unique_key",
                    severity="warning",
                    details=f"Found {duplicated} duplicate rows on key {key_columns}",
                )
            )

    for col in DATE_COLUMNS.get(table_name, []):
        if col in df.columns:
            failures = _date_parse_failures(df, col)
            if failures > 0:
                issues.append(
                    ValidationIssue(
                        table=table_name,
                        check="date_parse",
                        severity="warning",
                        details=f"Column '{col}' has {failures} unparsable non-null date values",
                    )
                )

    failures = _non_negative_failures(df, NON_NEGATIVE_COLUMNS.get(table_name, []))
    for col, count in failures.items():
        issues.append(
            ValidationIssue(
                table=table_name,
                check="non_negative",
                severity="warning",
                details=f"Column '{col}' has {count} negative values",
            )
        )

    return ValidationResult(table=table_name, passed=len(issues) == 0, issues=issues)


def validate_all_dataframes(data_root: str | Path = "data/datathon-2026-round-1") -> dict[str, ValidationResult]:
    dataframes = load_all_dataframes(data_root)
    return {table: validate_dataframe(table, df) for table, df in dataframes.items()}


def profile_dataframe(table_name: str, df: pd.DataFrame) -> dict[str, Any]:
    profile: dict[str, Any] = {
        "table": table_name,
        "rows": int(df.shape[0]),
        "columns": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "null_rate_pct": _null_rate_by_column(df),
        "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
    }

    numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()
    if numeric_cols:
        describe = df[numeric_cols].describe().transpose()[["mean", "std", "min", "max"]]
        profile["numeric_summary"] = describe.round(4).to_dict(orient="index")
    else:
        profile["numeric_summary"] = {}

    return profile


def profile_all_dataframes(data_root: str | Path = "data/datathon-2026-round-1") -> dict[str, dict[str, Any]]:
    dataframes = load_all_dataframes(data_root)
    return {name: profile_dataframe(name, df) for name, df in dataframes.items()}


def print_validation_report(results: dict[str, ValidationResult]) -> None:
    for table_name, result in results.items():
        status = "PASS" if result.passed else "FAIL"
        print(f"\n[{status}] {table_name}")
        if not result.issues:
            print("  - No issues found")
            continue

        for issue in result.issues:
            print(f"  - ({issue.severity}) {issue.check}: {issue.details}")


def print_profile_report(profiles: dict[str, dict[str, Any]]) -> None:
    for table_name, profile in profiles.items():
        print(f"\nTABLE: {table_name}")
        print(f"  Shape: {profile['rows']} x {profile['columns']}")
        print(f"  Duplicate rows: {profile['duplicate_rows']}")

        null_rate = profile["null_rate_pct"]
        top_nulls = sorted(null_rate.items(), key=lambda item: item[1], reverse=True)[:5]
        print("  Top null-rate columns:")
        for col, pct in top_nulls:
            print(f"    - {col}: {pct:.2f}%")


def inspect_data(data_root: str | Path = "data/datathon-2026-round-1") -> None:
    print("Running data validation...")
    validation_results = validate_all_dataframes(data_root)
    print_validation_report(validation_results)

    print("\nBuilding data profiles...")
    profiles = profile_all_dataframes(data_root)
    print_profile_report(profiles)
