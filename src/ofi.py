"""
ofi.py — Order Flow Imbalance from 1-minute LOB snapshots
LNA-Agent: Lobster News Alpha

OFI definition (Cont, Kukanov & Stoikov 2014, adapted for snapshot data):
  OFI_t = bid_size_t * I(bid_t >= bid_{t-1}) - bid_size_{t-1} * I(bid_t < bid_{t-1})
         - ask_size_t * I(ask_t <= ask_{t-1}) + ask_size_{t-1} * I(ask_t > ask_{t-1})

Usage:
  # Single file test
  python src/ofi.py data/lobster/raw/A/A_2020-12-31.csv A

  # Batch: all files for a ticker → saves results/ofi_baseline_{ticker}.csv
  python src/ofi.py --batch A
  python src/ofi.py --batch A --data-dir /path/to/folder
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import argparse
import sys

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS = [1, 5, 15]          # minutes ahead for IC computation
LOB_DIR  = Path("data/lobster/raw")
RESULTS  = Path("results")
# ─────────────────────────────────────────────────────────────────────────────


def load_lob(filepath: str | Path) -> pd.DataFrame:
    """Load a single LOB CSV file and parse time index."""
    df = pd.read_csv(filepath)
    df.columns = df.columns.str.strip().str.lower()

    rename = {
        "ask_1": "ask", "ask1": "ask",
        "ask_size_1": "ask_size", "asksize1": "ask_size",
        "bid_1": "bid", "bid1": "bid",
        "bid_size_1": "bid_size", "bidsize1": "bid_size",
    }
    df = df.rename(columns={k: v for k, v in rename.items() if k in df.columns})
    df["time"] = pd.to_datetime(df["time"], format="%H:%M:%S")
    df = df.set_index("time").sort_index()

    # Prices in LOBSTER are in units of 1/10000 — normalize if needed
    for col in ["ask", "bid"]:
        if df[col].median() > 100_000:
            df[col] = df[col] / 10_000

    return df[["ask", "ask_size", "bid", "bid_size"]].dropna()


def compute_ofi(df: pd.DataFrame) -> pd.Series:
    """Compute 1-minute OFI from LOB snapshot data."""
    bid, bid_size = df["bid"], df["bid_size"]
    ask, ask_size = df["ask"], df["ask_size"]

    bid_ofi = (bid_size * (bid >= bid.shift(1)).astype(int)
               - bid_size.shift(1) * (bid < bid.shift(1)).astype(int))

    ask_ofi = (-ask_size * (ask <= ask.shift(1)).astype(int)
               + ask_size.shift(1) * (ask > ask.shift(1)).astype(int))

    ofi = bid_ofi + ask_ofi
    ofi.name = "ofi"
    return ofi


def compute_midprice(df: pd.DataFrame) -> pd.Series:
    mid = (df["ask"] + df["bid"]) / 2
    mid.name = "mid"
    return mid


def compute_forward_returns(mid: pd.Series, horizons: list[int]) -> pd.DataFrame:
    return pd.DataFrame(
        {f"ret_{h}m": np.log(mid.shift(-h) / mid) for h in horizons},
        index=mid.index
    )


def information_coefficient(signal: pd.Series, forward_ret: pd.Series) -> float:
    mask = signal.notna() & forward_ret.notna()
    if mask.sum() < 10:
        return np.nan
    ic, _ = spearmanr(signal[mask], forward_ret[mask])
    return ic


def process_file(filepath: str | Path, ticker: str = "", verbose: bool = True) -> tuple[pd.DataFrame, dict]:
    """
    Full pipeline for one file.
    Returns (data_df, ic_dict)
    """
    df  = load_lob(filepath)
    ofi = compute_ofi(df)
    mid = compute_midprice(df)
    ret = compute_forward_returns(mid, HORIZONS)

    out = pd.concat([ofi, mid, ret], axis=1).dropna(subset=["ofi"])
    out["ofi_z"] = (out["ofi"] - out["ofi"].mean()) / (out["ofi"].std() + 1e-8)

    spread = ((df["ask"] - df["bid"]) / df["bid"] * 10_000).mean()

    ic_dict = {"ticker": ticker, "file": Path(filepath).stem, "spread_bps": round(spread, 2), "n_rows": len(out)}
    for h in HORIZONS:
        ic_dict[f"ic_{h}m"] = information_coefficient(out["ofi_z"], out[f"ret_{h}m"])

    if verbose:
        print(f"\n{'─'*50}")
        print(f"  {ticker or filepath}")
        print(f"{'─'*50}")
        print(f"  {'Horizon':<12} {'IC':>8}")
        print(f"  {'-------':<12} {'--':>8}")
        for h in HORIZONS:
            print(f"  {h} min{'':<8} {ic_dict[f'ic_{h}m']:>8.4f}")
        print(f"\n  Rows: {len(out):,}  |  Avg spread: {spread:.1f} bps")

    return out, ic_dict


def run_single(filepath: str, ticker: str = "") -> None:
    """Single file test mode."""
    process_file(filepath, ticker, verbose=True)


def run_batch(ticker: str, data_dir: Path | None = None) -> None:
    """
    Batch mode: process all CSVs for a ticker.
    Saves per-day IC table to results/ofi_baseline_{ticker}.csv
    Saves combined signal data to results/ofi_data_{ticker}.parquet
    """
    folder = data_dir or (LOB_DIR / ticker)
    files  = sorted(Path(folder).glob("*.csv"))

    if not files:
        raise FileNotFoundError(f"No CSV files found in {folder}")

    RESULTS.mkdir(exist_ok=True)

    print(f"\n{'═'*50}")
    print(f"  BATCH: {ticker} — {len(files)} files")
    print(f"{'═'*50}")

    all_data   = []
    all_ic     = []

    for i, f in enumerate(files):
        try:
            data, ic = process_file(f, ticker=ticker, verbose=False)
            # Tag with date extracted from filename
            date_str = _extract_date(f.stem)
            data["date"] = date_str
            all_data.append(data)
            all_ic.append(ic)
            if (i + 1) % 50 == 0:
                print(f"  Processed {i+1}/{len(files)} files...")
        except Exception as e:
            print(f"  [SKIP] {f.name}: {e}")

    if not all_data:
        print("No files processed successfully.")
        return

    # ── Save per-day IC table ──────────────────────────────────────────────
    ic_df = pd.DataFrame(all_ic)
    ic_path = RESULTS / f"ofi_baseline_{ticker}.csv"
    ic_df.to_csv(ic_path, index=False)
    print(f"\n  Per-day IC saved → {ic_path}")

    # ── Aggregate IC across all days ───────────────────────────────────────
    combined = pd.concat(all_data).sort_index()
    print(f"\n{'═'*50}")
    print(f"  {ticker} — AGGREGATE IC ({len(combined):,} rows, {len(files)} days)")
    print(f"{'═'*50}")
    print(f"  {'Horizon':<12} {'IC':>8}  {'Mean daily IC':>14}  {'Std':>8}")
    print(f"  {'-------':<12} {'--':>8}  {'-------------':>14}  {'---':>8}")
    for h in HORIZONS:
        agg_ic    = information_coefficient(combined["ofi_z"], combined[f"ret_{h}m"])
        daily_ic  = ic_df[f"ic_{h}m"].dropna()
        print(f"  {h} min{'':<8} {agg_ic:>8.4f}  {daily_ic.mean():>14.4f}  {daily_ic.std():>8.4f}")

    # ── Save combined data ─────────────────────────────────────────────────
    data_path = RESULTS / f"ofi_data_{ticker}.parquet"
    try:
        combined.to_parquet(data_path)
        print(f"\n  Combined data saved → {data_path}")
    except Exception:
        csv_path = RESULTS / f"ofi_data_{ticker}.csv"
        combined.to_csv(csv_path)
        print(f"\n  Combined data saved → {csv_path}")

    print(f"\n  Done. Files written to results/")


def _extract_date(stem: str) -> str:
    """Extract YYYY-MM-DD from filename like A_2020-12-31_34200000_57600000_60_1"""
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if len(p) == 4 and p.isdigit() and i + 2 < len(parts):
            try:
                return f"{parts[i]}-{parts[i+1]}-{parts[i+2]}"
            except Exception:
                pass
    return stem


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="OFI computation from 1-min LOB data")
    parser.add_argument("path_or_ticker", help="File path (single mode) or ticker symbol (batch mode)")
    parser.add_argument("ticker", nargs="?", default="", help="Ticker label for single file mode")
    parser.add_argument("--batch", action="store_true", help="Batch mode: process all files for ticker")
    parser.add_argument("--data-dir", type=str, default=None, help="Override data directory for batch mode")
    args = parser.parse_args()

    if args.batch:
        data_dir = Path(args.data_dir) if args.data_dir else None
        run_batch(args.path_or_ticker, data_dir=data_dir)
    else:
        run_single(args.path_or_ticker, args.ticker)