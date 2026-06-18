"""
join_lob_quotes.py — Attach L1 bid/ask quotes from LOBSTER to the event panel.
LNA-Agent: Lobster News Alpha

For each event in the panel, finds the LOBSTER file for that (ticker, date),
loads it with the legacy load_lob(), reconstructs full timestamps, and backward
merge_asofs the nearest preceding quote onto bar_time.

Resolved column mapping (from inspection):
  panel ticker column  : "stock"  (not "ticker")
  panel timestamp col  : "bar_time" (full datetime "YYYY-MM-DD HH:MM:SS")
  load_lob index       : time-only, stored as 1900-01-01 HH:MM:SS (dummy date)
  load_lob columns     : ask, bid, ask_size, bid_size  (after internal rename)
  LOB filename pattern : {TICKER}_{YYYY-MM-DD}*.csv  (flat dir per ticker)
  Price scale          : load_lob divides by 10 000 if median > 100 000

Output columns added to panel:
  ask, bid, ask_size, bid_size  — L1 best quotes at bar_time
  mid          — (ask + bid) / 2
  half_spread  — (ask - bid) / 2  in price units
  spread_bps   — half_spread / mid * 1e4  (basis points)
  lob_matched  — bool: True if a quote was found within tolerance
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Callable

# Pull legacy loader; do not reimplement
_LEGACY = Path(__file__).parent.parent / "legacy" / "src"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from ofi import load_lob as _load_lob  # type: ignore


def _find_lob_file(lob_dir: Path, ticker: str, date_str: str) -> Path | None:
    """
    Locate the LOBSTER CSV for one (ticker, date).

    Searches two layouts:
      1. {lob_dir}/{ticker}/{ticker}_{date_str}*.csv   (standard repo layout)
      2. {lob_dir}/{ticker}_{date_str}*.csv            (flat Downloads layout)

    Returns the first match, or None.
    """
    date_nodash = date_str.replace("-", "")   # "20190102" pattern fallback
    for base in [lob_dir / ticker, lob_dir]:
        for pat in [f"{ticker}_{date_str}*.csv", f"{ticker}_{date_nodash}*.csv"]:
            matches = sorted(base.glob(pat))
            if matches:
                return matches[0]
    return None


def _load_and_stamp(filepath: Path, date_str: str) -> pd.DataFrame:
    """
    Call load_lob() and reconstruct full datetime index from the time-only index.

    load_lob returns a DataFrame indexed by "1900-01-01 HH:MM:SS" because it
    parses the "time" column with format="%H:%M:%S" and no date.  We replace
    the dummy 1900 date with the actual trading date.
    """
    df = _load_lob(filepath)

    # Index is Datetime with dummy year 1900 — extract HH:MM:SS and prepend date
    time_strs = df.index.strftime("%H:%M:%S")
    df.index  = pd.to_datetime(date_str + " " + time_strs)
    df.index.name = "lob_ts"
    return df


def join_l1_quotes(
    panel: pd.DataFrame,
    lob_dir: str | Path,
    ts_col: str = "bar_time",
    ticker_col: str = "stock",     # real panel column name (not "ticker")
    tolerance: str = "1min",
) -> pd.DataFrame:
    """
    Attach L1 bid/ask quotes to each event row via backward merge_asof.

    Parameters
    ----------
    panel      : aligned event-level panel (one row per news event)
    lob_dir    : root directory of LOBSTER files.
                 Supports both {lob_dir}/{TICKER}/*.csv and flat {lob_dir}/*.csv.
    ts_col     : column in panel holding the bar timestamp ("bar_time")
    ticker_col : column in panel holding the ticker symbol ("stock")
    tolerance  : max lag for the backward asof merge (default 1 minute)

    Returns
    -------
    Copy of panel with added columns:
      ask, bid, ask_size, bid_size, mid, half_spread, spread_bps, lob_matched
    Rows where no LOB file was found or the quote is outside tolerance get
    NaN for the quote columns and lob_matched=False.
    """
    lob_dir  = Path(lob_dir)
    tol      = pd.Timedelta(tolerance)
    panel    = panel.copy()

    # Pre-allocate output columns
    for col in ["ask", "bid", "ask_size", "bid_size"]:
        panel[col] = np.nan
    panel["lob_matched"] = False

    # Group by (ticker, date) to load each LOB file once
    ts_series  = pd.to_datetime(panel[ts_col])
    date_strs  = ts_series.dt.strftime("%Y-%m-%d")
    tickers    = panel[ticker_col].astype(str)

    groups = panel.groupby([tickers, date_strs], sort=False)

    for (ticker, date_str), idx in groups.groups.items():
        filepath = _find_lob_file(lob_dir, ticker, date_str)
        if filepath is None:
            # No LOB file for this (ticker, date) — leave as NaN
            continue

        try:
            lob_df = _load_and_stamp(filepath, date_str)
        except Exception as e:
            print(f"  [join_lob] Could not load {filepath}: {e}")
            continue

        # Subset of panel rows for this (ticker, date)
        sub = panel.loc[idx].copy()
        sub_ts = pd.to_datetime(sub[ts_col]).sort_values()
        sub    = sub.loc[sub_ts.index]

        lob_reset = lob_df.reset_index()   # columns: lob_ts, ask, ask_size, bid, bid_size

        merged = pd.merge_asof(
            sub[[ts_col]].reset_index().rename(columns={"index": "_orig_idx"}),
            lob_reset,
            left_on=ts_col,
            right_on="lob_ts",
            direction="backward",
            tolerance=tol,
        )

        for _, mrow in merged.iterrows():
            oi = mrow["_orig_idx"]
            if pd.notna(mrow.get("ask")):
                panel.at[oi, "ask"]        = mrow["ask"]
                panel.at[oi, "bid"]        = mrow["bid"]
                panel.at[oi, "ask_size"]   = mrow["ask_size"]
                panel.at[oi, "bid_size"]   = mrow["bid_size"]
                panel.at[oi, "lob_matched"] = True

    # Derived columns — only where we have real quotes
    valid = panel["lob_matched"]
    mid         = (panel.loc[valid, "ask"] + panel.loc[valid, "bid"]) / 2
    half_spread = (panel.loc[valid, "ask"] - panel.loc[valid, "bid"]) / 2
    panel["mid"]         = np.nan
    panel["half_spread"] = np.nan
    panel["spread_bps"]  = np.nan
    panel.loc[valid, "mid"]         = mid
    panel.loc[valid, "half_spread"] = half_spread
    panel.loc[valid, "spread_bps"]  = half_spread / mid.replace(0, np.nan) * 1e4

    coverage = float(valid.mean())
    print(f"  [join_lob] Coverage: {valid.sum()}/{len(panel)} events "
          f"({coverage:.1%}) matched to a LOB quote.")

    return panel


def l1_slippage_bps(
    row: "pd.Series",
    trade_shares: float,
    side: str,
) -> tuple[float, bool]:
    """
    Estimate depth-aware slippage at L1 for one event row.

    Parameters
    ----------
    row          : one row from a panel that has been through join_l1_quotes
    trade_shares : order size in shares
    side         : "buy" or "sell"

    Returns
    -------
    (slippage_bps, exceeds_l1_flag)
      slippage_bps  — spread half-cost + any impact beyond L1 depth (in bps)
      exceeds_l1_flag — True if trade_shares exceeds the L1 quoted depth

    Note: multi-level impact requires L2+ data; only L1 depth is currently
    available from load_lob().  TODO(schema): extend when L2 is loaded.
    """
    ask      = float(row.get("ask",      np.nan))
    bid      = float(row.get("bid",      np.nan))
    ask_size = float(row.get("ask_size", np.nan))
    bid_size = float(row.get("bid_size", np.nan))

    if np.isnan(ask) or np.isnan(bid):
        return np.nan, False

    mid = (ask + bid) / 2
    if mid <= 0:
        return 0.0, False

    half_sp_bps = (ask - bid) / 2 / mid * 1e4

    if side == "buy":
        depth         = ask_size if (not np.isnan(ask_size) and ask_size > 0) else np.inf
        exceeds_l1    = trade_shares > depth
        # Linear price impact: fraction of order beyond depth * tick / mid
        impact_bps    = (
            max(0.0, trade_shares - depth) / depth * (ask - mid) / mid * 1e4
            if exceeds_l1 and depth < np.inf else 0.0
        )
    else:
        depth         = bid_size if (not np.isnan(bid_size) and bid_size > 0) else np.inf
        exceeds_l1    = trade_shares > depth
        impact_bps    = (
            max(0.0, trade_shares - depth) / depth * (mid - bid) / mid * 1e4
            if exceeds_l1 and depth < np.inf else 0.0
        )

    return float(half_sp_bps + impact_bps), bool(exceeds_l1)
