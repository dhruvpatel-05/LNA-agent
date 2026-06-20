"""
market_adjust.py — Equal-weighted cross-sectional market adjustment for forward returns.

Proxy: simple mean of ret_{h}m across all tickers present at the same bar_time,
using only the tickers already in the panel (no new data).

SPY/QQQ benchmarking deferred post-deadline.

Events at bar_times with fewer than 2 tickers get NaN for the adjusted return.
"""

import numpy as np
import pandas as pd

HORIZONS = [1, 5, 15]


def compute_ew_proxy(pool: pd.DataFrame, horizons: list[int] = HORIZONS) -> pd.DataFrame:
    """
    Equal-weighted cross-sectional mean forward return at each bar_time.

    pool must have columns: bar_time, stock, ret_{h}m for h in horizons.
    Returns a DataFrame indexed by bar_time with columns:
        proxy_{h}m      — EW mean forward return (NaN if <2 tickers)
        n_tickers_{h}m  — number of tickers contributing at that timestamp

    Only bar_times where at least 2 distinct tickers have a non-NaN ret_{h}m
    get a non-NaN proxy.
    """
    df = pool.copy()
    df["bar_time"] = pd.to_datetime(df["bar_time"])

    agg: dict[str, dict] = {}
    for ts, grp in df.groupby("bar_time", sort=True):
        row: dict = {}
        for h in horizons:
            col = f"ret_{h}m"
            if col not in grp.columns:
                row[f"proxy_{h}m"]     = np.nan
                row[f"n_tickers_{h}m"] = 0
                continue
            vals = grp[col].dropna()
            n    = len(vals)
            row[f"proxy_{h}m"]     = float(vals.mean()) if n >= 2 else np.nan
            row[f"n_tickers_{h}m"] = n
        agg[ts] = row

    proxy = pd.DataFrame.from_dict(agg, orient="index")
    proxy.index.name = "bar_time"
    return proxy


def add_madj_returns(
    panel: pd.DataFrame,
    proxy: pd.DataFrame,
    horizons: list[int] = HORIZONS,
) -> pd.DataFrame:
    """
    Add market-adjusted forward return columns to panel.

    Merges proxy (indexed by bar_time) onto panel and computes:
        ret_{h}m_madj = ret_{h}m - proxy_{h}m

    Events with NaN proxy (fewer than 2 tickers at that timestamp) get
    NaN for the adjusted return. Returns a copy of panel with new columns.
    """
    out = panel.copy()
    out["bar_time"] = pd.to_datetime(out["bar_time"])

    proxy_cols = [f"proxy_{h}m" for h in horizons if f"proxy_{h}m" in proxy.columns]
    out = out.join(proxy[proxy_cols], on="bar_time")

    for h in horizons:
        raw_col   = f"ret_{h}m"
        proxy_col = f"proxy_{h}m"
        adj_col   = f"ret_{h}m_madj"
        if raw_col in out.columns and proxy_col in out.columns:
            out[adj_col] = out[raw_col] - out[proxy_col]

    return out


def proxy_coverage_report(panel: pd.DataFrame, horizons: list[int] = HORIZONS) -> dict:
    """
    Report how many events lack a valid proxy (proxy_{h}m is NaN).

    Returns dict: {h: {'n_total': ..., 'n_no_proxy': ..., 'frac_no_proxy': ...}}
    """
    report = {}
    for h in horizons:
        proxy_col = f"proxy_{h}m"
        adj_col   = f"ret_{h}m_madj"
        if adj_col not in panel.columns:
            continue
        n_total    = len(panel)
        n_no_proxy = int(panel[adj_col].isna().sum() - panel[f"ret_{h}m"].isna().sum()
                         if f"ret_{h}m" in panel.columns else panel[adj_col].isna().sum())
        # simpler: count rows where proxy was NaN (hence adj is NaN even though raw is not)
        if proxy_col in panel.columns:
            n_no_proxy = int(panel[proxy_col].isna().sum())
        else:
            n_no_proxy = int(panel[adj_col].isna().sum())
        report[h] = {
            "n_total":    n_total,
            "n_no_proxy": n_no_proxy,
            "frac":       n_no_proxy / n_total if n_total else np.nan,
        }
    return report
