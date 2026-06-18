"""
news.py — Load SentARL headlines from local parquet cache
LNA-Agent: Lobster News Alpha

Data source:
  data/news/sentarl_full.parquet   — HuggingFace SentARL dataset (318k rows, 1997–2021)

Schema:
  date          — ISO 8601 UTC string  "2020-01-15T14:30:00Z"
  text          — full article text; first line (before first \\n) is the headline
  extra_fields  — JSON string with keys:
                    stocks         : list of ticker symbols tagged to this article
                    source         : original outlet name
                    text_type      : always "headline+subhead+abstract"
                    time_precision : always "minute"
                    dataset, dataset_source, raw_type, tz_hint

Pipeline:
  1. Load parquet (full file, 318k rows, ~0.05s JSON parse)
  2. Vectorised filter: ticker in extra_fields.stocks
  3. Filter to [date_from, date_to]
  4. Parse UTC timestamps → convert to US/Eastern
  5. Filter to market hours 09:30–16:00 ET
  6. Extract headline = first line of text field
  7. Deduplicate on (timestamp, headline)

Usage:
  python src/news.py --ticker NVDA --date-from 2016-01-01 --date-to 2020-12-31
  python src/news.py --scan --ticker NVDA          # count only, no save
  python src/news.py --scan --ticker NVDA AAPL AMZN  # multi-ticker scan
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

import json
import argparse
import pandas as pd
from pathlib import Path

# ── Config ────────────────────────────────────────────────────────────────────
SENTARL_FILE = Path("data/news/sentarl_full.parquet")
OUT_DIR      = Path("data/news/processed")
ET_TZ        = "US/Eastern"
SCAN_WINDOW  = ("2016-01-01", "2020-12-31")   # default window shown in --scan
# ─────────────────────────────────────────────────────────────────────────────


def _load_sentarl() -> tuple[ pd.DataFrame, pd.Series]:
    """
    Load the full SentARL parquet and return (df, ef_parsed) where
    ef_parsed is a Series of already-decoded extra_fields dicts.

    Called once per process; cheap to re-read (~0.05s JSON parse for 318k rows).
    """
    df = pd.read_parquet(SENTARL_FILE)
    ef_parsed = df["extra_fields"].apply(json.loads)
    return df, ef_parsed


def load_headlines(ticker: str, date_from: str, date_to: str) -> pd.DataFrame:
    """
    Load SentARL headlines for `ticker` in [date_from, date_to] during market hours.

    Steps:
      1. Filter rows where ticker in extra_fields.stocks
      2. Filter to [date_from, date_to]
      3. Parse UTC timestamps → US/Eastern
      4. Filter to 09:30–16:00 ET
      5. Extract headline = first line of text (before first \\n)
      6. Deduplicate on (timestamp, headline)

    Returns DataFrame with columns:
      timestamp  — tz-naive ET datetime  (ready for merge_asof with LOB bars)
      headline   — first line of article text
      stock      — ticker symbol
      source     — original outlet (from extra_fields.source)
      date       — YYYY-MM-DD string (ET local date)
      time       — HH:MM:SS string (ET local time)
    """
    df, ef_parsed = _load_sentarl()

    # ── Ticker filter (vectorised) ────────────────────────────────────────────
    ticker_mask = ef_parsed.apply(lambda d: ticker in d.get("stocks", []))
    sub = df[ticker_mask].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["timestamp", "headline", "stock", "source", "date", "time"]
        )

    # ── Parse UTC timestamps ──────────────────────────────────────────────────
    sub["ts_utc"] = pd.to_datetime(sub["date"], utc=True)

    # ── Date-range filter (UTC is fine for coarse date comparison) ───────────
    d_from = pd.Timestamp(date_from, tz="UTC")
    d_to   = pd.Timestamp(date_to,   tz="UTC") + pd.Timedelta(days=1)
    sub = sub[(sub["ts_utc"] >= d_from) & (sub["ts_utc"] < d_to)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["timestamp", "headline", "stock", "source", "date", "time"]
        )

    # ── Convert to ET ─────────────────────────────────────────────────────────
    sub["ts_et"] = sub["ts_utc"].dt.tz_convert(ET_TZ)

    # ── Market-hours filter (09:30–16:00 ET) ─────────────────────────────────
    mins = sub["ts_et"].dt.hour * 60 + sub["ts_et"].dt.minute
    sub  = sub[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)].copy()
    if sub.empty:
        return pd.DataFrame(
            columns=["timestamp", "headline", "stock", "source", "date", "time"]
        )

    # ── Extract headline = first line of text ────────────────────────────────
    sub["headline"] = sub["text"].str.split("\n").str[0].str.strip()

    # ── Build source column ───────────────────────────────────────────────────
    sub_ef_filtered = ef_parsed.loc[sub.index]
    sub["source"]   = sub_ef_filtered.apply(lambda d: d.get("source", "SentARL"))

    # ── Drop timezone for merge_asof ──────────────────────────────────────────
    sub["timestamp"] = sub["ts_et"].dt.tz_localize(None)
    sub["stock"]     = ticker
    sub["date"]      = sub["ts_et"].dt.strftime("%Y-%m-%d")
    sub["time"]      = sub["ts_et"].dt.strftime("%H:%M:%S")

    # ── Drop empty headlines, deduplicate ─────────────────────────────────────
    sub = sub[sub["headline"].str.len() > 0]
    sub = sub.drop_duplicates(subset=["timestamp", "headline"])

    return (
        sub[["timestamp", "headline", "stock", "source", "date", "time"]]
        .sort_values("timestamp")
        .reset_index(drop=True)
    )


def save_headlines(ticker: str, df: pd.DataFrame, suffix: str) -> Path:
    """Save to data/news/processed/{ticker}_headlines_{suffix}.csv"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{ticker}_headlines_{suffix}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} headlines → {path}")
    return path


def load_saved(ticker: str, suffix: str) -> pd.DataFrame:
    """Load saved headlines for downstream sentiment.py / align.py."""
    path = OUT_DIR / f"{ticker}_headlines_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No headlines at {path}. "
            f"Run: python src/news.py --ticker {ticker} --date-from ... --date-to ..."
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── --scan mode ───────────────────────────────────────────────────────────────

def scan_ticker(ticker: str, window: tuple[str, str] = SCAN_WINDOW) -> None:
    """
    Print headline counts and sample headlines for a ticker without saving.
    Used for quick data-coverage checks before committing to a full pipeline run.
    """
    df, ef_parsed = _load_sentarl()
    ticker_mask   = ef_parsed.apply(lambda d: ticker in d.get("stocks", []))
    sub           = df[ticker_mask].copy()

    if sub.empty:
        print(f"  {ticker}: 0 rows — not found in SentARL")
        return

    sub["ts_utc"] = pd.to_datetime(sub["date"], utc=True)
    sub["ts_et"]  = sub["ts_utc"].dt.tz_convert(ET_TZ)

    # Window subset
    w_from, w_to = window
    in_window = sub[
        (sub["ts_utc"] >= pd.Timestamp(w_from, tz="UTC")) &
        (sub["ts_utc"] <  pd.Timestamp(w_to,   tz="UTC") + pd.Timedelta(days=1))
    ]

    # Market-hours subset (within window)
    mins = in_window["ts_et"].dt.hour * 60 + in_window["ts_et"].dt.minute
    in_mkt = in_window[(mins >= 9 * 60 + 30) & (mins <= 16 * 60)]

    print(f"\n  {ticker}  [{w_from} → {w_to}]")
    print(f"  {'─' * 48}")
    print(f"  Total in dataset  : {len(sub):>7,}")
    print(f"  In window         : {len(in_window):>7,}")
    print(f"  In market hours   : {len(in_mkt):>7,}")

    if len(sub) > 0:
        print(f"  Full date range   : {sub['ts_et'].min().date()} → {sub['ts_et'].max().date()}")

    # Year distribution within window
    if not in_window.empty:
        by_year = in_window["ts_utc"].dt.year.value_counts().sort_index()
        year_str = "  ".join(f"{y}:{n}" for y, n in by_year.items())
        print(f"  By year           : {year_str}")

    # Sample headlines from market-hours window
    if not in_mkt.empty:
        in_mkt = in_mkt.copy()
        in_mkt["headline"] = in_mkt["text"].str.split("\n").str[0].str.strip()
        sample = in_mkt.sample(min(5, len(in_mkt)), random_state=42)
        print(f"\n  Sample headlines (market hours):")
        for _, row in sample.sort_values("ts_et").iterrows():
            ts = row["ts_et"].strftime("%Y-%m-%d %H:%M")
            print(f"    [{ts}]  {row['headline'][:70]}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Load SentARL headlines — LNA-Agent")
    parser.add_argument("--ticker",    type=str,  nargs="+", required=True,
                        help="One or more ticker symbols")
    parser.add_argument("--date-from", type=str,  default=None)
    parser.add_argument("--date-to",   type=str,  default=None)
    parser.add_argument("--scan",      action="store_true",
                        help="Print counts and samples only — do not save")
    args = parser.parse_args()

    tickers = args.ticker

    if args.scan:
        print(f"\nSentARL ticker scan  (window: {SCAN_WINDOW[0]} → {SCAN_WINDOW[1]})")
        for ticker in tickers:
            scan_ticker(ticker)
        raise SystemExit(0)

    # Normal load-and-save mode — date args required
    if not args.date_from or not args.date_to:
        parser.error("--date-from and --date-to are required unless --scan is used")

    for ticker in tickers:
        suffix = f"{args.date_from}_{args.date_to}"
        print(f"\n  Loading {ticker} | {args.date_from} → {args.date_to}")

        df = load_headlines(ticker, args.date_from, args.date_to)

        if df.empty:
            print(f"  No headlines found for {ticker} in this date range.")
            continue

        save_headlines(ticker, df, suffix)

        print(f"  Headlines       : {len(df)}")
        print(f"  Date range      : {df['timestamp'].min().date()} → {df['timestamp'].max().date()}")
        print(f"  Unique days     : {df['timestamp'].dt.date.nunique()}")
        print(f"\n  Sample (first 5):")
        for _, row in df.head(5).iterrows():
            print(f"  [{row['timestamp']}]  {row['headline'][:70]}")
