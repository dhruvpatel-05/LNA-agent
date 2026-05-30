"""
align.py — Align news headlines to 1-min LOB bars
LNA-Agent: Lobster News Alpha

Core event-study join: for each news event we record the state of the LOB
(OFI, mid-price) at the bar where the news arrived, plus forward returns at
1 / 5 / 15 min.  This panel feeds signal.py for IC and regression analysis.

Design:
  - merge_asof (backward): headline at 10:32:17 → matched to 10:32 bar
  - tolerance=2 min: headlines with no nearby LOB bar are dropped
    (pre-market releases, data gaps)
  - Market-hours filter removes pre/post-market noise
  - Relevance filter drops tangentially related articles (RavenPack metric)

Usage:
  python src/align.py --ticker AAPL --suffix 2019-01-01_2020-12-31
  python src/align.py --test          # synthetic smoke-test, no files needed
"""

import sys as _sys, os as _os
# src/signal.py shadows Python's stdlib signal module when src/ is in sys.path.
# anyio/pyarrow (used by anthropic/pandas) import stdlib signal at init time;
# without this guard, scripts in src/ cause ImportError for signal.Signals.
# Guard handles both absolute and relative path entries (e.g. 'src' vs '/abs/src').
if "signal" not in _sys.modules:
    _src = _os.path.dirname(_os.path.abspath(__file__))
    _to_remove = [(i, p) for i, p in enumerate(_sys.path)
                  if _os.path.abspath(p) == _src]
    for i, _ in reversed(_to_remove):
        _sys.path.pop(i)
    import signal as _s; _sys.modules["signal"] = _s
    for i, p in _to_remove:
        _sys.path.insert(i, p)

from dotenv import load_dotenv
load_dotenv()

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS  = Path("results")
NEWS_DIR = Path("data/news/raw")
HORIZONS = [1, 5, 15]
# ──────────────────────────────────────────────────────────────────────────────


def filter_market_hours(
    df: pd.DataFrame,
    ts_col: str = "timestamp",
    open_time: str = "09:30",
    close_time: str = "16:00",
) -> pd.DataFrame:
    """
    Keep only rows where timestamp falls within regular US equity market hours.

    Pre-market and after-hours headlines are excluded because:
      - The 1-min LOB data covers 09:30-16:00 only
      - OFI is not meaningful outside continuous auction hours
    """
    t      = df[ts_col].dt.time
    open_  = pd.Timestamp(open_time).time()
    close_ = pd.Timestamp(close_time).time()
    return df[(t >= open_) & (t <= close_)].copy()


def filter_relevance(
    df: pd.DataFrame,
    min_relevance: float = 0.3,
    col: str = "relevance_score",
) -> pd.DataFrame:
    """
    Drop headlines whose RavenPack relevance_score is below min_relevance.

    Relevance [0, 1] measures how directly the article concerns the specific
    ticker. Low-relevance articles are tangentially related (sector/macro news)
    and are expected to contribute noise rather than alpha.
    If the relevance_score column is absent the DataFrame is returned unchanged.
    """
    if col not in df.columns:
        return df
    return df[df[col] >= min_relevance].copy()


def align(headlines_df: pd.DataFrame, ofi_df: pd.DataFrame) -> pd.DataFrame:
    """
    Snap each headline to the nearest preceding 1-min LOB bar and join OFI data.

    For each headline:
      - Finds the most recent LOB bar with bar_time <= headline timestamp
      - Joins: ofi, ofi_z, mid, ret_1m, ret_5m, ret_15m from that bar
      - Adds bar_time column to show which LOB bar was matched

    tolerance=2min: if no bar exists within 2 minutes before the headline
    (e.g. news published at open before first snapshot) the row is dropped.

    ofi_df must have a DatetimeIndex with full date+time timestamps.
    Both DataFrames must be timezone-naive.

    Returns a DataFrame with one row per successfully matched headline,
    retaining all original headline columns plus the joined LOB columns.
    """
    headlines = headlines_df.copy().sort_values("timestamp").reset_index(drop=True)

    # ── Strip timezone info if present ───────────────────────────────────────
    if headlines["timestamp"].dt.tz is not None:
        headlines["timestamp"] = headlines["timestamp"].dt.tz_localize(None)

    ofi = ofi_df.copy().sort_index()
    if ofi.index.tz is not None:
        ofi.index = ofi.index.tz_localize(None)

    # ── Flatten OFI index to a column for merge_asof ─────────────────────────
    ofi_reset  = ofi.reset_index()
    # The former index becomes the first column; rename it bar_time
    ofi_reset  = ofi_reset.rename(columns={ofi_reset.columns[0]: "bar_time"})

    # ── Backward merge: each headline gets the most recent preceding LOB bar ─
    merged = pd.merge_asof(
        headlines,
        ofi_reset.sort_values("bar_time"),
        left_on="timestamp",
        right_on="bar_time",
        direction="backward",
        tolerance=pd.Timedelta("2min"),
    )

    # Drop headlines that had no nearby LOB bar
    n_before = len(merged)
    merged   = merged.dropna(subset=["bar_time"])
    n_after  = len(merged)
    if n_before != n_after:
        print(f"  [align] Dropped {n_before - n_after} headlines outside LOB coverage")

    return merged.reset_index(drop=True)


def load_ofi_data(ticker: str, suffix: str) -> pd.DataFrame:
    """
    Load OFI panel data saved by ofi.py, reconstructing full DatetimeIndex.

    ofi.py sets the index to the time portion only (format %H:%M:%S, parsed
    with a dummy 1900-01-01 date) and stores the actual date in a 'date' column.
    This function reconstructs proper timestamps: date + time → DatetimeIndex.

    Tries parquet first (preferred, preserves dtypes), then CSV.
    """
    for ext in [".parquet", ".csv"]:
        path = RESULTS / f"ofi_data_{ticker}_{suffix}{ext}"
        if not path.exists():
            continue

        if ext == ".parquet":
            df = pd.read_parquet(path)
        else:
            df = pd.read_csv(path, index_col=0)
            df.index = pd.to_datetime(df.index)

        # Rebuild full timestamps when index is time-only (dummy 1900-01-01 date)
        if "date" in df.columns:
            time_part = df.index.strftime("%H:%M:%S")
            df.index  = pd.to_datetime(df["date"].astype(str) + " " + time_part)
            df.index.name = "bar_time"
            df = df.drop(columns=["date"])

        return df.sort_index()

    raise FileNotFoundError(
        f"No OFI data found for {ticker}_{suffix} in {RESULTS}.\n"
        f"Run: python src/ofi.py --batch {ticker} --date-from ... --date-to ..."
    )


def load_sentiment_data(ticker: str, suffix: str) -> pd.DataFrame:
    """Load scored headlines saved by sentiment.py."""
    path = RESULTS / f"sentiment_{ticker}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No sentiment file at {path}.\n"
            f"Run: python src/sentiment.py --ticker {ticker} --suffix {suffix}"
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


def save_aligned(aligned: pd.DataFrame, ticker: str, suffix: str) -> Path:
    """Save the aligned event panel for downstream use by signal.py."""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"aligned_{ticker}_{suffix}.parquet"
    aligned.to_parquet(path, index=False)
    print(f"\n  Saved {len(aligned)} aligned events → {path}")
    return path


def load_aligned(ticker: str, suffix: str) -> pd.DataFrame:
    """Load aligned event panel saved by align.py."""
    path = RESULTS / f"aligned_{ticker}_{suffix}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"No aligned data at {path}.\n"
            f"Run: python src/align.py --ticker {ticker} --suffix {suffix}"
        )
    return pd.read_parquet(path)


# ── Synthetic data generators (shared with signal.py --test) ──────────────────

def _make_test_ofi_df(date: str = "2020-01-02") -> pd.DataFrame:
    """
    Generate a synthetic 1-min OFI panel for one trading day.
    OFI and returns are seeded for reproducibility.
    The index has full date+time timestamps (as expected by align()).
    """
    np.random.seed(1)
    times = pd.date_range(f"{date} 09:30", f"{date} 16:00", freq="1min")
    n     = len(times)
    ofi   = np.random.randn(n) * 1000
    # Simulate a random walk for mid-price
    mid   = 150.0 + np.cumsum(np.random.randn(n) * 0.05)
    mid_s = pd.Series(mid, index=times)
    rets  = {
        f"ret_{h}m": np.log(mid_s.shift(-h) / mid_s).values
        for h in HORIZONS
    }
    df = pd.DataFrame(
        {
            "ofi":   ofi,
            "ofi_z": (ofi - ofi.mean()) / (ofi.std() + 1e-8),
            "mid":   mid,
            **rets,
        },
        index=times,
    )
    df.index.name = "bar_time"
    return df


def _make_test_headlines_df(date: str = "2020-01-02") -> pd.DataFrame:
    """
    Generate synthetic scored headlines for one trading day.
    Sentiment scores are correlated so the test IC table is non-trivial.
    """
    np.random.seed(2)
    n        = 15
    # Random minute offsets within market hours (570 = 09:30, 960 = 16:00)
    offsets  = np.sort(np.random.choice(np.arange(0, 390), size=n, replace=False))
    times    = pd.Timestamp(f"{date} 09:30") + pd.to_timedelta(offsets, unit="min")
    texts    = [
        "Company posts strong quarterly earnings",
        "CEO steps down amid investigation",
        "Record revenue announced for fiscal year",
        "Lawsuit filed, company faces substantial damages",
        "Product launch exceeds analyst expectations",
        "Supply chain disruption warning issued",
        "Dividend increased, buyback program expanded",
        "Regulatory probe into accounting practices",
        "Strategic acquisition completed at premium",
        "Guidance lowered for full fiscal year",
        "New market expansion announced",
        "Credit downgrade issued by rating agency",
        "Analyst upgrades stock to buy",
        "Inventory writedown of $2B disclosed",
        "Major partnership deal signed",
    ]
    # Construct correlated scores: LLM is LM + small noise; RP is LM + small noise
    lm = np.random.uniform(-1, 1, n)
    llm = np.clip(lm + np.random.normal(0, 0.15, n), -1, 1)
    rp  = np.clip(lm + np.random.normal(0, 0.10, n), -1, 1)
    return pd.DataFrame(
        {
            "timestamp":       times,
            "headline":        texts,
            "ticker":          "AAPL",
            "lm_score":        lm,
            "llm_score":       llm,
            "ravenpack_score": rp,
            "relevance_score": np.random.uniform(0.3, 1.0, n),
        }
    )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Align headlines to LOB bars — LNA-Agent")
    parser.add_argument("--ticker",        type=str,   default="AAPL")
    parser.add_argument("--suffix",        type=str,   default="")
    parser.add_argument("--min-relevance", type=float, default=0.3)
    parser.add_argument("--test",          action="store_true",
                        help="Run on synthetic data (no files needed)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    suffix = args.suffix

    if args.test:
        print("\n[TEST MODE] Generating synthetic OFI and headline data")
        ofi_df       = _make_test_ofi_df()
        headlines_df = _make_test_headlines_df()
        suffix       = "test"
    else:
        ofi_df       = load_ofi_data(args.ticker, suffix)
        headlines_df = load_sentiment_data(args.ticker, suffix)

    print(f"\n  Raw headlines:              {len(headlines_df)}")
    headlines_df = filter_market_hours(headlines_df)
    print(f"  After market-hours filter:  {len(headlines_df)}")
    headlines_df = filter_relevance(headlines_df, min_relevance=args.min_relevance)
    print(f"  After relevance filter:     {len(headlines_df)}")

    aligned = align(headlines_df, ofi_df)
    print(f"  After LOB alignment:        {len(aligned)}")

    save_aligned(aligned, args.ticker, suffix)

    # Preview
    preview_cols = [
        "timestamp", "bar_time", "headline",
        "lm_score", "llm_score", "ravenpack_score",
        "ofi_z", "ret_1m", "ret_5m",
    ]
    show = [c for c in preview_cols if c in aligned.columns]
    print(f"\n  Preview (first 5 rows):")
    pd.set_option("display.max_colwidth", 35)
    pd.set_option("display.float_format", "{:+.4f}".format)
    print(aligned[show].head().to_string(index=False))
