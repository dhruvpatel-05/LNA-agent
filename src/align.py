"""
align.py — Snap headlines to 1-min LOB bars and join OFI + forward returns
LNA-Agent: Lobster News Alpha

The OFI CSV (from ofi.py) stores timestamps split across two columns:
  time  — "1900-01-01 09:31:00"  (dummy 1900 date + real intraday time)
  date  — "2019-01-02"           (actual trading date as string)

This module reconstructs full bar timestamps, then uses merge_asof (backward,
2-min tolerance) to snap each headline to the immediately preceding 1-min bar.

The resulting aligned panel has one row per headline that matched a LOB bar,
with columns from both the headline side (lm_score, llm_score) and the OFI
side (ofi, ofi_z, mid, ret_1m, ret_5m, ret_15m).

Usage:
  python src/align.py --ticker AAPL --suffix 2019-01-01_2020-12-31
"""

import sys as _sys, os as _os
if "signal" not in _sys.modules:
    _src = _os.path.dirname(_os.path.abspath(__file__))
    _to_remove = [(i, p) for i, p in enumerate(_sys.path)
                  if _os.path.abspath(p) == _src]
    for i, _ in reversed(_to_remove):
        _sys.path.pop(i)
    import signal as _s; _sys.modules["signal"] = _s
    for i, p in _to_remove:
        _sys.path.insert(i, p)

import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS  = Path("results")
ALIGN_DIR = Path("results/align")
HORIZONS = [1, 5, 15]
TOLERANCE = pd.Timedelta("2min")   # max lag between headline and LOB bar
# ─────────────────────────────────────────────────────────────────────────────


def load_ofi(ticker: str, suffix: str) -> pd.DataFrame:
    """
    Load OFI panel from results/ofi_data_{ticker}_{suffix}.csv.

    Reconstructs full DatetimeIndex from the split time/date columns:
      bar_time = date_str + time_str[11:]   →  "2020-03-09 09:31:00"

    Returns DataFrame indexed by bar_time with columns:
      ofi, ofi_z, mid, ret_1m, ret_5m, ret_15m
    """
    for ext in [".csv", ".parquet"]:
        path = RESULTS / f"ofi/ofi_data_{ticker}_{suffix}{ext}"
        if not path.exists():
            continue
        if ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path)

        # Reconstruct full bar timestamps from ofi.py's split time/date storage.
        # ofi.py writes HH:MM:SS-only times (parsed as 1900-01-01 HH:MM:SS) and
        # the actual trading date in a separate "date" column.  This split appears
        # as a "time" column in CSV output, but as the DatetimeIndex itself in
        # parquet (because parquet round-trips the index intact).
        date_s  = None   # "YYYY-MM-DD" strings
        time_s  = None   # "HH:MM:SS" strings

        if "date" in df.columns:
            # Normalise to plain YYYY-MM-DD regardless of dtype (str or datetime)
            date_s = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

        if "time" in df.columns:
            # CSV path: "time" column holds "1900-01-01 HH:MM:SS"
            time_s = pd.to_datetime(df["time"]).dt.strftime("%H:%M:%S")
        elif (isinstance(df.index, pd.DatetimeIndex)
              and df.index.min().year == 1900
              and date_s is not None):
            # Parquet path: time-only index preserved; year==1900 is the sentinel
            time_s = df.index.strftime("%H:%M:%S")

        if date_s is not None and time_s is not None:
            df.index = pd.to_datetime(date_s + " " + time_s)
            df = df.drop(columns=[c for c in ("time", "date") if c in df.columns])
        elif "bar_time" in df.columns:
            df.index = pd.to_datetime(df["bar_time"])
            df = df.drop(columns=["bar_time"])

        df.index.name = "bar_time"

        keep = [c for c in ["ofi", "ofi_z", "mid", "ret_1m", "ret_5m", "ret_15m"]
                if c in df.columns]
        return df[keep].sort_index()

    raise FileNotFoundError(
        f"No OFI file for {ticker}_{suffix} in {RESULTS}/ofi.\n"
        f"Run: python src/ofi.py --batch {ticker} --data-dir <LOB_DIR> "
        f"--date-from ... --date-to ..."
    )


def filter_market_hours(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
) -> pd.DataFrame:
    """Keep rows whose timestamp falls within 09:30–16:00 (market hours)."""
    mins = df[ts_col].dt.hour * 60 + df[ts_col].dt.minute
    return df[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)].copy()


def align(headlines_df: pd.DataFrame, ofi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Snap each headline to the nearest preceding 1-min LOB bar.

    Uses pd.merge_asof with direction='backward' and a 2-minute tolerance:
      - headline at 10:32:17  →  matched to bar 10:32
      - headline before first bar, or >2 min gap  →  dropped

    Both DataFrames must use timezone-naive timestamps.
    headlines_df must have columns: timestamp, headline, lm_score, llm_score
    ofi_df must be indexed by bar_time with columns: ofi, ofi_z, ret_1m/5m/15m

    Returns merged DataFrame with one row per matched headline.
    """
    hl = headlines_df.copy().sort_values("timestamp").reset_index(drop=True)

    # Strip tz if present
    if hl["timestamp"].dt.tz is not None:
        hl["timestamp"] = hl["timestamp"].dt.tz_localize(None)

    ofi = ofi_df.copy().sort_index()
    if ofi.index.tz is not None:
        ofi.index = ofi.index.tz_localize(None)

    # Force the index name before reset so the resulting column is always "bar_time".
    # DatetimeIndex built from Series arithmetic can silently lose its name,
    # causing reset_index() to produce an "index" column instead of "bar_time".
    ofi.index.name = "bar_time"
    ofi_reset = ofi.reset_index()

    merged = pd.merge_asof(
        hl,
        ofi_reset,
        left_on="timestamp",
        right_on="bar_time",
        direction="backward",
        tolerance=TOLERANCE,
    )

    n_dropped = merged["bar_time"].isna().sum()
    if n_dropped:
        print(f"  [align] Dropped {n_dropped} headlines with no LOB bar within {TOLERANCE}")
    merged = merged.dropna(subset=["bar_time"])

    return merged.reset_index(drop=True)


def save_aligned(ticker: str, df: pd.DataFrame, suffix: str) -> Path:
    """Save aligned panel to results/aligned_{ticker}_{suffix}.csv (human-readable)."""
    ALIGN_DIR.mkdir(exist_ok=True)
    path = ALIGN_DIR / f"aligned_{ticker}_{suffix}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} aligned events → {path}")
    return path


def load_aligned(ticker: str, suffix: str) -> pd.DataFrame:
    """Load aligned panel saved by this module. Prefers CSV; falls back to parquet."""
    for ext in [".csv", ".parquet"]:
        path = ALIGN_DIR / f"aligned_{ticker}_{suffix}{ext}"
        if path.exists():
            if ext == ".parquet":
                return pd.read_parquet(path)
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            return df
    raise FileNotFoundError(
        f"No aligned panel for {ticker}_{suffix}. "
        f"Run: python src/align.py --ticker {ticker} --suffix {suffix}"
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys
    parser = argparse.ArgumentParser(description="Align headlines to LOB bars — LNA-Agent")
    parser.add_argument("--ticker", type=str, required=True)
    parser.add_argument("--suffix", type=str, required=True)
    args = parser.parse_args()

    sys.path.insert(0, str(Path(__file__).parent))
    from sentiment import load_scored

    print(f"\n  Loading scored headlines  ({args.ticker} | {args.suffix})")
    scored = load_scored(args.ticker, args.suffix)
    print(f"  Headlines: {len(scored)}")

    print(f"  Loading OFI data  ({args.ticker} | {args.suffix})")
    ofi = load_ofi(args.ticker, args.suffix)
    print(f"  OFI bars: {len(ofi)}  |  {ofi.index.min().date()} → {ofi.index.max().date()}")

    print(f"  Aligning…")
    aligned = align(scored, ofi)
    print(f"  Aligned events: {len(aligned)}")

    if aligned.empty:
        print(
            "\n  WARNING: 0 events aligned. Check that news timestamps "
            "overlap with the OFI date range."
        )
        print(f"  News window:  {scored['timestamp'].min().date()} → {scored['timestamp'].max().date()}")
        print(f"  OFI  window:  {ofi.index.min().date()} → {ofi.index.max().date()}")
        raise SystemExit(1)

    save_aligned(args.ticker, aligned, args.suffix)

    print(f"\n  ── Summary ──────────────────────────────────────────")
    print(f"  Aligned rows : {len(aligned)}")
    print(f"  Date range   : {aligned['timestamp'].min().date()} → {aligned['timestamp'].max().date()}")
    for h in HORIZONS:
        col = f"ret_{h}m"
        if col in aligned.columns:
            print(f"  ret_{h}m      : "
                  f"mean={aligned[col].mean():+.5f}  "
                  f"std={aligned[col].std():.5f}")

    pd.set_option("display.float_format", "{:+.4f}".format)
    preview_cols = [c for c in
                    ["timestamp", "bar_time", "headline", "lm_score",
                     "llm_score", "ofi_z", "ret_1m", "ret_5m"]
                    if c in aligned.columns]
    print(f"\n  Preview (first 5):")
    print(aligned[preview_cols].head().to_string(index=False))
