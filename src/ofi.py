"""
ofi.py — Order Flow Imbalance from 1-minute LOB snapshots
LNA-Agent: Lobster News Alpha

OFI definition (Cont, Kukanov & Stoikov 2014, adapted for snapshot data):
  OFI_t = dBidSize_t * I(Bid_t >= Bid_{t-1}) - dAskSize_t * I(Ask_t <= Ask_{t-1})

Where:
  - dBidSize_t = BidSize_t - BidSize_{t-1}
  - dAskSize_t = AskSize_t - AskSize_{t-1}
  - Indicator conditions capture whether the bid/ask price level held
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr


# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS = [1, 5, 15]   # minutes ahead for IC computation
LOB_DIR  = Path("data/lobster/raw")
# ─────────────────────────────────────────────────────────────────────────────


def load_lob(filepath: str | Path) -> pd.DataFrame:
    """Load a single LOB CSV file and parse time index."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    # Normalize column names
    rename = {
        "time": "time",
        "ask_1": "ask", "ask1": "ask",
        "ask_size_1": "ask_size", "asksize1": "ask_size",
        "bid_1": "bid", "bid1": "bid",
        "bid_size_1": "bid_size", "bidsize1": "bid_size",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})

    # Parse time — handle both "09:30:00" and full datetime strings
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    df = df.set_index("time").sort_index()

    # Prices in LOBSTER are in units of 1/10000 — normalize if needed
    for col in ["ask", "bid"]:
        if df[col].median() > 100_000:
            df[col] = df[col] / 10_000

    return df[["ask", "ask_size", "bid", "bid_size"]].dropna()


def compute_ofi(df: pd.DataFrame) -> pd.Series:
    """
    Compute 1-minute OFI from LOB snapshot data.

    OFI_t = bid_size_t * I(bid_t >= bid_{t-1}) 
           - bid_size_{t-1} * I(bid_t < bid_{t-1})
           - ask_size_t * I(ask_t <= ask_{t-1})
           + ask_size_{t-1} * I(ask_t > ask_{t-1})
    """
    bid      = df["bid"]
    bid_size = df["bid_size"]
    ask      = df["ask"]
    ask_size = df["ask_size"]

    # Bid side contribution
    bid_up   = bid >= bid.shift(1)   # bid price held or rose
    bid_down = bid < bid.shift(1)    # bid price fell

    bid_ofi = (bid_size * bid_up.astype(int)
               - bid_size.shift(1) * bid_down.astype(int))

    # Ask side contribution
    ask_down = ask <= ask.shift(1)   # ask price held or fell
    ask_up   = ask > ask.shift(1)    # ask price rose

    ask_ofi = (-ask_size * ask_down.astype(int)
               + ask_size.shift(1) * ask_up.astype(int))

    ofi = bid_ofi + ask_ofi
    ofi.name = "ofi"
    return ofi


def compute_midprice(df: pd.DataFrame) -> pd.Series:
    """Mid-quote price."""
    mid = (df["ask"] + df["bid"]) / 2
    mid.name = "mid"
    return mid


def compute_forward_returns(mid: pd.Series, horizons: list[int]) -> pd.DataFrame:
    """Log returns at each horizon h minutes ahead."""
    returns = {}
    for h in horizons:
        fwd = np.log(mid.shift(-h) / mid)
        returns[f"ret_{h}m"] = fwd
    return pd.DataFrame(returns, index=mid.index)


def information_coefficient(signal: pd.Series, forward_ret: pd.Series) -> float:
    """Spearman rank IC between signal and forward return."""
    mask = signal.notna() & forward_ret.notna()
    if mask.sum() < 10:
        return np.nan
    ic, _ = spearmanr(signal[mask], forward_ret[mask])
    return ic


def run_ofi_analysis(filepath: str | Path, ticker: str = "") -> pd.DataFrame:
    """
    Full pipeline: load LOB → compute OFI → compute ICs at each horizon.
    Returns a DataFrame with OFI, mid, returns, and summary IC table.
    """
    print(f"\n{'─'*50}")
    print(f"  {ticker or filepath}")
    print(f"{'─'*50}")

    df  = load_lob(filepath)
    ofi = compute_ofi(df)
    mid = compute_midprice(df)
    ret = compute_forward_returns(mid, HORIZONS)

    # Combine into single frame
    out = pd.concat([ofi, mid, ret], axis=1).dropna(subset=["ofi"])

    # Normalize OFI to z-score for comparability across stocks
    out["ofi_z"] = (out["ofi"] - out["ofi"].mean()) / out["ofi"].std()

    # Compute IC at each horizon
    print(f"\n  OFI Information Coefficients (Spearman):")
    print(f"  {'Horizon':<12} {'IC':>8}")
    print(f"  {'-------':<12} {'--':>8}")
    for h in HORIZONS:
        col = f"ret_{h}m"
        ic  = information_coefficient(out["ofi_z"], out[col])
        print(f"  {h} min{'':<8} {ic:>8.4f}")

    # Basic LOB stats
    spread = ((df["ask"] - df["bid"]) / df["bid"] * 10_000)  # bps
    print(f"\n  LOB Summary:")
    print(f"  Rows:           {len(df):>8,}")
    print(f"  Avg spread:     {spread.mean():>8.1f} bps")
    print(f"  OFI mean:       {ofi.mean():>8.2f}")
    print(f"  OFI std:        {ofi.std():>8.2f}")

    return out


def run_all(ticker: str) -> pd.DataFrame:
    """
    Run OFI analysis for all CSVs found for a given ticker.
    Expects files in: data/lobster/raw/{ticker}/*.csv
    """
    folder = LOB_DIR / ticker
    files  = sorted(folder.glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    frames = []
    for f in files:
        try:
            out = run_ofi_analysis(f, ticker=f"{ticker} | {f.stem}")
            frames.append(out)
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")

    combined = pd.concat(frames).sort_index()

    # Overall IC across all days
    print(f"\n{'═'*50}")
    print(f"  {ticker} — OVERALL IC ({len(combined):,} rows)")
    print(f"{'═'*50}")
    for h in HORIZONS:
        ic = information_coefficient(combined["ofi_z"], combined[f"ret_{h}m"])
        print(f"  {h} min IC: {ic:.4f}")

    return combined


# ── Quick single-file test ────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        # Usage: python ofi.py path/to/file.csv AAPL
        path   = sys.argv[1]
        ticker = sys.argv[2] if len(sys.argv) > 2 else ""
        run_ofi_analysis(path, ticker)
    else:
        # Default: run AAPL if data exists
        aapl_dir = LOB_DIR / "AAPL"
        if aapl_dir.exists():
            run_all("AAPL")
        else:
            print("Usage: python src/ofi.py <path_to_lob_csv> [TICKER]")
            print("       python src/ofi.py data/lobster/raw/AAPL/AAPL_2022-01-03.csv AAPL")