"""
ofi.py — Order Flow Imbalance from 1-minute LOB snapshots
LNA-Agent: Lobster News Alpha

OFI definition (Cont, Kukanov & Stoikov 2014, adapted for snapshot data):
  OFI_t = bid_size_t * I(bid_t >= bid_{t-1}) - bid_size_{t-1} * I(bid_t < bid_{t-1})
         - ask_size_t * I(ask_t <= ask_{t-1}) + ask_size_{t-1} * I(ask_t > ask_{t-1})

Usage:
  python src/ofi.py data/lobster/raw/A/A_2020-12-31.csv A
  python src/ofi.py --batch A --data-dir /path/to/folder
  python src/ofi.py --batch A --data-dir /path/to/folder \
      --date-from 2019-01-01 --date-to 2020-12-31
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
import argparse

# ── Config ────────────────────────────────────────────────────────────────────
HORIZONS = [1, 5, 15]
LOB_DIR  = Path("data/lobster/raw")
RESULTS  = Path("results")
OFI_DIR  = RESULTS / "ofi"
# ─────────────────────────────────────────────────────────────────────────────


def load_lob(filepath: str | Path) -> pd.DataFrame:
    df = pd.read_csv(filepath)

    df.columns = df.columns.str.strip().str.lower()

    rename = {
        "ask_1": "ask",
        "ask1": "ask",
        "ask_size_1": "ask_size",
        "asksize1": "ask_size",
        "bid_1": "bid",
        "bid1": "bid",
        "bid_size_1": "bid_size",
        "bidsize1": "bid_size",
    }

    df = df.rename(
        columns={k: v for k, v in rename.items() if k in df.columns}
    )

    df["time"] = pd.to_datetime(
        df["time"],
        format="%H:%M:%S"
    )

    df = df.set_index("time").sort_index()

    for col in ["ask", "bid"]:
        if df[col].median() > 100_000:
            df[col] = df[col] / 10_000

    return df[
        ["ask", "ask_size", "bid", "bid_size"]
    ].dropna()


def compute_ofi(df: pd.DataFrame) -> pd.Series:
    bid = df["bid"]
    bid_size = df["bid_size"]

    ask = df["ask"]
    ask_size = df["ask_size"]

    bid_ofi = (
        bid_size * (bid >= bid.shift(1)).astype(int)
        - bid_size.shift(1) * (bid < bid.shift(1)).astype(int)
    )

    ask_ofi = (
        -ask_size * (ask <= ask.shift(1)).astype(int)
        + ask_size.shift(1) * (ask > ask.shift(1)).astype(int)
    )

    ofi = bid_ofi + ask_ofi
    ofi.name = "ofi"

    return ofi


def compute_midprice(df: pd.DataFrame) -> pd.Series:
    mid = (df["ask"] + df["bid"]) / 2
    mid.name = "mid"
    return mid


def compute_forward_returns(
    mid: pd.Series,
    horizons: list[int]
) -> pd.DataFrame:

    return pd.DataFrame(
        {
            f"ret_{h}m": np.log(
                mid.shift(-h) / mid
            )
            for h in horizons
        },
        index=mid.index
    )


def information_coefficient(
    signal: pd.Series,
    forward_ret: pd.Series
) -> float:

    mask = signal.notna() & forward_ret.notna()

    if mask.sum() < 10:
        return np.nan

    ic, _ = spearmanr(
        signal[mask],
        forward_ret[mask]
    )

    return ic


def extract_date(stem: str) -> str | None:
    """
    Extract YYYY-MM-DD from filename like:
    A_2020-12-31_34200000_57600000_60_1
    """

    parts = stem.split("_")

    for i in range(len(parts) - 2):
        try:
            date_str = f"{parts[i]}-{parts[i+1]}-{parts[i+2]}"
            pd.to_datetime(date_str)
            return date_str
        except Exception:
            pass

    import re

    match = re.search(
        r"(\d{4}-\d{2}-\d{2})",
        stem
    )

    return match.group(1) if match else None


def process_file(
    filepath: str | Path,
    ticker: str = "",
    verbose: bool = True
) -> tuple[pd.DataFrame, dict]:

    df = load_lob(filepath)

    ofi = compute_ofi(df)
    mid = compute_midprice(df)
    ret = compute_forward_returns(mid, HORIZONS)

    out = pd.concat(
        [ofi, mid, ret],
        axis=1
    ).dropna(subset=["ofi"])

    out["ofi_z"] = (
        out["ofi"] - out["ofi"].mean()
    ) / (
        out["ofi"].std() + 1e-8
    )

    spread = (
        (df["ask"] - df["bid"])
        / df["bid"]
        * 10_000
    ).mean()

    date = extract_date(
        Path(filepath).stem
    ) or ""

    if date:
        out.index = pd.to_datetime(date + " " + out.index.strftime("%H:%M:%S"))

    ic_dict = {
        "ticker": ticker,
        "date": date,
        "file": Path(filepath).stem,
        "spread_bps": round(spread, 2),
        "n_rows": len(out),
    }

    for h in HORIZONS:
        ic_dict[f"ic_{h}m"] = information_coefficient(
            out["ofi_z"],
            out[f"ret_{h}m"]
        )

    if verbose:

        print(f"\n{'─'*50}")
        print(f"  {ticker or filepath} [{date}]")
        print(f"{'─'*50}")

        print(f"  {'Horizon':<12} {'IC':>8}")
        print(f"  {'-------':<12} {'--':>8}")

        for h in HORIZONS:
            print(
                f"  {h} min{'':<8} "
                f"{ic_dict[f'ic_{h}m']:>8.4f}"
            )

        print(
            f"\n  Rows: {len(out):,}"
            f" | Avg spread: {spread:.1f} bps"
        )

    return out, ic_dict


def run_single(
    filepath: str,
    ticker: str = ""
) -> None:

    process_file(
        filepath,
        ticker,
        verbose=True
    )


def run_batch(
    ticker: str,
    data_dir: Path | None = None,
    date_from: str | None = None,
    date_to: str | None = None,
) -> None:

    folder = data_dir or (LOB_DIR / ticker)

    files = sorted(
        Path(folder).glob("*.csv")
    )

    if not files:
        raise FileNotFoundError(
            f"No CSV files found in {folder}"
        )

    if date_from or date_to:

        df_from = (
            pd.to_datetime(date_from)
            if date_from
            else pd.Timestamp.min
        )

        df_to = (
            pd.to_datetime(date_to)
            if date_to
            else pd.Timestamp.max
        )

        filtered = []

        for f in files:

            d = extract_date(f.stem)

            if d:

                dt = pd.to_datetime(d)

                if df_from <= dt <= df_to:
                    filtered.append(f)

        files = filtered

        suffix = (
            f"{date_from or 'start'}"
            f"_{date_to or 'end'}"
        )

    else:
        suffix = "all"

    if not files:
        print("No files matched the date filter.")
        return

    OFI_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    print(f"\n{'═'*50}")
    print(
        f"  BATCH: {ticker}"
        f" — {len(files)} files"
        f" [{suffix}]"
    )
    print(f"{'═'*50}")

    all_data = []
    all_ic = []

    for i, f in enumerate(files):

        try:

            data, ic = process_file(
                f,
                ticker=ticker,
                verbose=False
            )

            date_str = extract_date(
                f.stem
            ) or ""

            data["date"] = date_str

            all_data.append(data)
            all_ic.append(ic)

            if (i + 1) % 50 == 0:

                print(
                    f"  Processed "
                    f"{i+1}/{len(files)} files..."
                )

        except Exception as e:

            print(
                f"  [SKIP] "
                f"{f.name}: {e}"
            )

    if not all_data:
        print(
            "No files processed successfully."
        )
        return

    ic_df = pd.DataFrame(all_ic)

    combined = pd.concat(
        all_data
    ).sort_index()

    agg_rows = []

    print(f"\n{'═'*50}")
    print(
        f"  {ticker}"
        f" — AGGREGATE IC "
        f"({len(combined):,} rows,"
        f" {len(files)} days)"
    )

    print(f"{'═'*50}")

    print(
        f"  {'Horizon':<12}"
        f" {'Agg IC':>8}"
        f"  {'Mean daily':>12}"
        f"  {'Std':>8}"
    )

    print(
        f"  {'-------':<12}"
        f" {'------':>8}"
        f"  {'----------':>12}"
        f"  {'---':>8}"
    )

    for h in HORIZONS:

        agg_ic = information_coefficient(
            combined["ofi_z"],
            combined[f"ret_{h}m"]
        )

        daily_ic = ic_df[
            f"ic_{h}m"
        ].dropna()

        agg_rows.append({
            "ticker": ticker,
            "horizon_min": h,
            "aggregate_ic": agg_ic,
            "mean_daily_ic": daily_ic.mean(),
            "std_daily_ic": daily_ic.std(),
            "rows": len(combined),
            "days": len(files),
            "suffix": suffix,
        })

        print(
            f"  {h} min{'':<8}"
            f" {agg_ic:>8.4f}"
            f"  {daily_ic.mean():>12.4f}"
            f"  {daily_ic.std():>8.4f}"
        )

    data_path = (
        OFI_DIR
        / f"ofi_data_{ticker}_{suffix}.csv"
    )

    combined.to_csv(data_path)

    agg_path = (
        OFI_DIR
        / f"ofi_aggregate_{ticker}_{suffix}.csv"
    )

    pd.DataFrame(
        agg_rows
    ).to_csv(
        agg_path,
        index=False
    )

    print(
        f"\n  Combined data saved → "
        f"{data_path}"
    )

    print(
        f"  Aggregate IC saved → "
        f"{agg_path}"
    )

    print("\n  Done.")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="OFI computation from 1-min LOB data"
    )

    parser.add_argument(
        "path_or_ticker",
        help="File path (single) or ticker (batch)"
    )

    parser.add_argument(
        "ticker",
        nargs="?",
        default="",
        help="Ticker label for single mode"
    )

    parser.add_argument(
        "--batch",
        action="store_true",
        help="Batch mode"
    )

    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="Data directory for batch mode"
    )

    parser.add_argument(
        "--date-from",
        type=str,
        default=None,
        help="Start date YYYY-MM-DD"
    )

    parser.add_argument(
        "--date-to",
        type=str,
        default=None,
        help="End date YYYY-MM-DD"
    )

    args = parser.parse_args()

    if args.batch:

        run_batch(
            args.path_or_ticker,
            data_dir=(
                Path(args.data_dir)
                if args.data_dir
                else None
            ),
            date_from=args.date_from,
            date_to=args.date_to,
        )

    else:

        run_single(
            args.path_or_ticker,
            args.ticker
        )