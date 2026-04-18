from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class TimeSplit:
    fold: int
    train_idx: list[int]
    valid_idx: list[int]


def _sorted_indices_by_date(df: pd.DataFrame, date_col: str) -> pd.Index:
    if date_col not in df.columns:
        raise ValueError(f"Missing date column: {date_col}")

    dates = pd.to_datetime(df[date_col], errors="coerce")
    if dates.isna().any():
        raise ValueError(f"Date column '{date_col}' contains invalid timestamps")

    return df.assign(_date=dates).sort_values("_date").index


def rolling_window_splits(
    df: pd.DataFrame,
    date_col: str,
    train_size: int,
    valid_size: int,
    step_size: int,
) -> list[TimeSplit]:
    """Generate rolling-window splits for time-series validation."""
    if train_size <= 0 or valid_size <= 0 or step_size <= 0:
        raise ValueError("train_size, valid_size, and step_size must be positive")

    sorted_idx = _sorted_indices_by_date(df, date_col).tolist()
    total = len(sorted_idx)
    splits: list[TimeSplit] = []

    fold = 0
    start = 0
    while start + train_size + valid_size <= total:
        train_idx = sorted_idx[start : start + train_size]
        valid_idx = sorted_idx[start + train_size : start + train_size + valid_size]
        splits.append(TimeSplit(fold=fold, train_idx=train_idx, valid_idx=valid_idx))
        fold += 1
        start += step_size

    return splits


def expanding_window_splits(
    df: pd.DataFrame,
    date_col: str,
    initial_train_size: int,
    valid_size: int,
    step_size: int,
) -> list[TimeSplit]:
    """Generate expanding-window splits for time-series validation."""
    if initial_train_size <= 0 or valid_size <= 0 or step_size <= 0:
        raise ValueError("initial_train_size, valid_size, and step_size must be positive")

    sorted_idx = _sorted_indices_by_date(df, date_col).tolist()
    total = len(sorted_idx)
    splits: list[TimeSplit] = []

    fold = 0
    train_end = initial_train_size
    while train_end + valid_size <= total:
        train_idx = sorted_idx[:train_end]
        valid_idx = sorted_idx[train_end : train_end + valid_size]
        splits.append(TimeSplit(fold=fold, train_idx=train_idx, valid_idx=valid_idx))
        fold += 1
        train_end += step_size

    return splits
