"""plot_features.py
===================
Generic plotting utilities for daily feature analysis.

Functions
---------
build_daily       – aggregate a dataframe to one row per date.
plot_by_year      – per-year line plots with a rolling-average overlay.
plot_all_year     – single figure spanning the full date range.
plot_overlay      – overlay multiple series on a single axes per year (no MA).
"""

from __future__ import annotations

import math

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.dates import DateFormatter, YearLocator


def build_daily(
    source_df: pd.DataFrame,
    date_col: str = "date",
    fields: list[str] | None = None,
    agg: str = "sum",
) -> pd.DataFrame:
    """Aggregate *source_df* to one row per date.

    Parameters
    ----------
    source_df : DataFrame containing *date_col* and numeric feature columns.
    date_col  : Name of the date column.
    fields    : Columns to aggregate. ``None`` → all numeric columns except *date_col*.
    agg       : Aggregation function (``"sum"``, ``"mean"``, …).
    """
    if date_col not in source_df.columns:
        raise ValueError(f"Missing date column: {date_col!r}")

    out = source_df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    out = out.dropna(subset=[date_col]).sort_values(date_col)

    if fields is None:
        fields = [
            c for c in out.columns
            if c != date_col and pd.api.types.is_numeric_dtype(out[c])
        ]
    else:
        fields = [c for c in fields if c in out.columns]

    if not fields:
        raise ValueError("No numeric fields found to aggregate")

    return out.groupby(date_col, as_index=False).agg({c: agg for c in fields})


def plot_by_year(
    daily: pd.DataFrame,
    fields: list[str],
    date_col: str = "date",
    ma_window: int = 7,
    years: list[int] | None = None,
    skip_years: list[int] | None = None,
    max_cols: int = 2,
    figsize_per_row: float = 3.2,
    title_prefix: str = "",
) -> None:
    """One figure per year; each figure has one subplot per field.

    Parameters
    ----------
    daily          : Daily dataframe (output of :func:`build_daily`).
    fields         : Column names to plot.
    date_col       : Name of the date column.
    ma_window      : Rolling-average window (days).
    years          : Explicit list of years. ``None`` → all years present.
    skip_years     : Years to exclude (default: ``[2012]``).
    max_cols       : Max subplot columns per figure.
    figsize_per_row: Figure height per row (inches).
    title_prefix   : Optional string prepended to each figure's suptitle.
    """
    if skip_years is None:
        skip_years = [2012]

    frame = daily[[date_col] + [c for c in fields if c in daily.columns]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    frame["year"] = frame[date_col].dt.year

    plot_fields = [c for c in fields if c in frame.columns and c != date_col]
    if not plot_fields:
        raise ValueError("No matching fields found in dataframe")

    if years is None:
        years = sorted(frame["year"].dropna().unique().tolist())
    years = [y for y in years if y not in skip_years]

    for year in years:
        yearly = frame[frame["year"] == year]
        if yearly.empty:
            continue

        n = len(plot_fields)
        ncols = min(max_cols, n)
        nrows = math.ceil(n / ncols)

        fig, axes = plt.subplots(
            nrows, ncols,
            figsize=(8 * ncols, figsize_per_row * nrows),
            sharex=True,
        )
        axes_list = [axes] if isinstance(axes, plt.Axes) else list(axes.flat)

        for idx, col in enumerate(plot_fields):
            ax = axes_list[idx]
            ax.plot(
                yearly[date_col], yearly[col],
                color="#1f77b4", linewidth=1.2, alpha=0.55, label=col,
            )
            ax.plot(
                yearly[date_col],
                yearly[col].rolling(ma_window, min_periods=1).mean(),
                color="#d62728", linewidth=1.9, label=f"ma{ma_window}",
            )
            ax.set_title(f"{col} – {year}")
            ax.grid(alpha=0.25)
            ax.legend(loc="upper left")

        for j in range(n, len(axes_list)):
            axes_list[j].axis("off")

        prefix = f"{title_prefix} | " if title_prefix else ""
        fig.suptitle(f"{prefix}{year} – daily trends (MA{ma_window})", fontsize=14, y=1.01)
        plt.tight_layout()
        plt.show()


def plot_all_year(
    daily: pd.DataFrame,
    fields: list[str],
    date_col: str = "date",
    ma_window: int = 7,
    start_date: str = "2013-01-01",
    max_cols: int = 2,
    figsize_per_row: float = 3.4,
    title_prefix: str = "",
) -> None:
    """Single figure covering the full date range from *start_date*.

    Parameters
    ----------
    daily          : Daily dataframe (output of :func:`build_daily`).
    fields         : Column names to plot.
    date_col       : Name of the date column.
    ma_window      : Rolling-average window (days).
    start_date     : Earliest date to include.
    max_cols       : Max subplot columns.
    figsize_per_row: Figure height per row (inches).
    title_prefix   : Optional string prepended to the figure suptitle.
    """
    frame = daily[[date_col] + [c for c in fields if c in daily.columns]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    frame = frame[frame[date_col] >= pd.Timestamp(start_date)].copy()

    plot_fields = [c for c in fields if c in frame.columns and c != date_col]
    if not plot_fields or frame.empty:
        raise ValueError(f"No data or no matching fields from {start_date!r}")

    x_start = pd.Timestamp(start_date)
    x_end   = frame[date_col].max()

    n = len(plot_fields)
    ncols = min(max_cols, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(8 * ncols, figsize_per_row * nrows),
        sharex=True,
    )
    axes_list = [axes] if isinstance(axes, plt.Axes) else list(axes.flat)

    for i, col in enumerate(plot_fields):
        ax = axes_list[i]
        ax.plot(
            frame[date_col], frame[col],
            color="#1f77b4", linewidth=1.2, alpha=0.5, label=col,
        )
        ax.plot(
            frame[date_col],
            frame[col].rolling(ma_window, min_periods=1).mean(),
            color="#d62728", linewidth=1.9, label=f"{col}_ma{ma_window}",
        )
        ax.set_xlim(x_start, x_end)
        ax.xaxis.set_major_locator(YearLocator(1))
        ax.xaxis.set_major_formatter(DateFormatter("%Y"))
        ax.set_title(f"{col} | {x_start.year} to {x_end.year}")
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")

    for j in range(n, len(axes_list)):
        axes_list[j].axis("off")

    prefix = f"{title_prefix} | " if title_prefix else ""
    fig.suptitle(
        f"{prefix}full timeline from {start_date} (MA{ma_window})",
        fontsize=13, y=1.01,
    )
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# COLORS used by plot_overlay (cycle-safe palette)
# ---------------------------------------------------------------------------
_OVERLAY_COLORS = [
    "#1f77b4",  # blue
    "#ff7f0e",  # orange
    "#2ca02c",  # green
    "#d62728",  # red
    "#9467bd",  # purple
    "#8c564b",  # brown
]


def plot_overlay(
    daily: pd.DataFrame,
    fields: list[str],
    date_col: str = "date",
    years: list[int] | None = None,
    skip_years: list[int] | None = None,
    start_date: str = "2013-01-01",
    figsize: tuple[float, float] = (14, 4),
    alpha: float = 0.75,
    fill: bool = True,
    title_prefix: str = "",
) -> None:
    """Overlay multiple series on **one axes** per year – no rolling average.

    Designed for comparing a small number of closely related series, e.g.
    Revenue vs COGS, where seeing them on the same scale matters.

    Parameters
    ----------
    daily        : Daily dataframe (output of :func:`build_daily`).
    fields       : Column names to overlay (all drawn on the same axes).
    date_col     : Name of the date column.
    years        : Explicit list of years. ``None`` → all years present.
    skip_years   : Years to exclude (default: ``[2012]``).
    start_date   : Only used when *years* is ``None`` to filter early data.
    figsize      : ``(width, height)`` per figure in inches.
    alpha        : Line / fill opacity.
    fill         : If ``True`` and exactly 2 fields, shade the area between them.
    title_prefix : Optional string prepended to each figure's suptitle.
    """
    if skip_years is None:
        skip_years = [2012]

    frame = daily[[date_col] + [c for c in fields if c in daily.columns]].copy()
    frame[date_col] = pd.to_datetime(frame[date_col], errors="coerce")
    frame = frame.dropna(subset=[date_col]).sort_values(date_col)
    frame["_year"] = frame[date_col].dt.year

    plot_fields = [c for c in fields if c in frame.columns and c != date_col]
    if not plot_fields:
        raise ValueError("No matching fields found in dataframe")

    if years is None:
        years = sorted(frame["_year"].dropna().unique().tolist())
    years = [y for y in years if y not in skip_years]

    for year in years:
        yearly = frame[frame["_year"] == year].copy()
        if yearly.empty:
            continue

        fig, ax = plt.subplots(figsize=figsize)

        for i, col in enumerate(plot_fields):
            color = _OVERLAY_COLORS[i % len(_OVERLAY_COLORS)]
            ax.plot(
                yearly[date_col], yearly[col],
                color=color, linewidth=1.8, alpha=alpha, label=col,
            )

        # Optional fill between first two series
        if fill and len(plot_fields) == 2:
            s0 = yearly[plot_fields[0]]
            s1 = yearly[plot_fields[1]]
            ax.fill_between(
                yearly[date_col], s0, s1,
                where=(s0 >= s1),
                interpolate=True,
                color=_OVERLAY_COLORS[0], alpha=0.12,
            )
            ax.fill_between(
                yearly[date_col], s0, s1,
                where=(s0 < s1),
                interpolate=True,
                color=_OVERLAY_COLORS[1], alpha=0.12,
            )

        prefix = f"{title_prefix} | " if title_prefix else ""
        ax.set_title(f"{prefix}{year} – {' vs '.join(plot_fields)}", fontsize=13)
        ax.grid(alpha=0.25)
        ax.legend(loc="upper left")
        plt.tight_layout()
        plt.show()


def plot_correlation_heatmap(
    df: pd.DataFrame,
    cols: list[str],
    group_label: str = "",
) -> None:
    """Draw a full pairwise correlation heatmap for *cols* in *df*.

    Filters *cols* to numeric-only columns present in *df*, then plots a
    coolwarm heatmap.  If fewer than 2 numeric columns are found, prints a
    warning and returns without plotting.

    Parameters
    ----------
    df          : The daily dataframe (e.g. ``daily_df``).
    cols        : Column names belonging to the feature group.
    group_label : Label shown in the figure title.
    """
    import seaborn as sns

    subset = df[cols].select_dtypes(include="number")

    if subset.shape[1] < 2:
        print(f"[{group_label}] fewer than 2 numeric columns — skipping.")
        return

    corr_matrix = subset.corr()

    plt.figure(figsize=(16, 12))
    sns.heatmap(corr_matrix, cmap="coolwarm", center=0)
    plt.title(f"Multicollinearity Matrix — {group_label}")
    plt.tight_layout()
    plt.show()
