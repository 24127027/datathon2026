from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def _to_float_series(values: Iterable[float], expected_len: int, name: str) -> pd.Series:
    series = pd.Series(values, dtype="float64")
    if len(series) != expected_len:
        raise ValueError(
            f"{name} length mismatch: expected {expected_len} values, got {len(series)}"
        )
    return series.reset_index(drop=True)


def prepare_submission(
    revenue: Iterable[float] | None = None,
    cogs: Iterable[float] | None = None,
    start_date: str = "2023-01-01",
    end_date: str = "2024-07-01",
) -> pd.DataFrame:
    """Create a submission dataframe for the configured date horizon.

    Defaults to the required range 2023-01-01 through 2024-07-01 (inclusive).
    If `revenue` or `cogs` is not provided, the missing column is filled with 0.0.
    """
    dates = pd.date_range(start=start_date, end=end_date, freq="D")
    n_rows = len(dates)

    if revenue is None:
        revenue_series = pd.Series([0.0] * n_rows, dtype="float64")
    else:
        revenue_series = _to_float_series(revenue, expected_len=n_rows, name="revenue")

    if cogs is None:
        cogs_series = pd.Series([0.0] * n_rows, dtype="float64")
    else:
        cogs_series = _to_float_series(cogs, expected_len=n_rows, name="cogs")

    submission = pd.DataFrame(
        {
            "date": dates.strftime("%Y-%m-%d"),
            "Revenue": revenue_series,
            "COGS": cogs_series,
        }
    )
    return submission
