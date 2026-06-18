"""
convert_ofi.py — Convert raw per-minute OFI csvs (data/ofi/ofi_{TICKER}_{YEAR}.csv)
into the format expected by align.load_ofi
(results/ofi/ofi_data_{TICKER}_{suffix}.csv).

Raw columns: minute, ofi, fwd_ret_1min, fwd_ret_5min, fwd_ret_15min, date (YYYYMMDD), ticker
Output columns: time (HH:MM:SS), date (YYYY-MM-DD), ofi, ofi_z, ret_1m, ret_5m, ret_15m

ofi_z is a per-day z-score: (ofi - day_mean) / (day_std + 1e-8), matching ofi.py.

Usage:
  python src/convert_ofi.py
"""

import pandas as pd
from pathlib import Path

DATA_DIR = Path("data/ofi")
OUT_DIR  = Path("results/ofi")

TICKERS = ["AMD", "NVDA", "META", "TSLA", "AAPL"]
SUFFIX  = "2025-08-01_2025-12-01"
START   = pd.Timestamp("2025-08-01")
END     = pd.Timestamp("2025-12-01")


def convert(ticker: str) -> None:
    parts = []
    for year in (2024, 2025):
        path = DATA_DIR / f"ofi_{ticker}_{year}.csv"
        parts.append(pd.read_csv(path))
    df = pd.concat(parts, ignore_index=True)

    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df = df[(df["date"] >= START) & (df["date"] <= END)].copy()

    df["time"] = (
        pd.to_datetime(df["minute"], unit="m", origin="1900-01-01")
        .dt.strftime("%H:%M:%S")
    )

    df["ofi_z"] = df.groupby("date")["ofi"].transform(
        lambda x: (x - x.mean()) / x.std()
    )

    df = df.rename(columns={
        "fwd_ret_1min":  "ret_1m",
        "fwd_ret_5min":  "ret_5m",
        "fwd_ret_15min": "ret_15m",
    })

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    out = df[["time", "date", "ofi", "ofi_z", "ret_1m", "ret_5m", "ret_15m"]].sort_values(
        ["date", "time"]
    ).reset_index(drop=True)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"ofi_data_{ticker}_{SUFFIX}.csv"
    out.to_csv(out_path, index=False)

    print(f"  {ticker}: {len(out)} rows  |  {out['date'].min()} -> {out['date'].max()}  -> {out_path}")


if __name__ == "__main__":
    for t in TICKERS:
        convert(t)
