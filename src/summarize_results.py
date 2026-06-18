"""
summarize_results.py — Cross-ticker IC summary from saved agent results.

Loads pre-computed CSVs from results/agent/ and prints a clean summary.
Makes no API calls and loads no raw market data.

Usage:
  python src/summarize_results.py --suffix 2016-01-01_2020-12-31
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path

AGENT_RESULTS = Path("results") / "agent"
RESULTS       = Path("results")

TICKERS    = ["AAPL", "JPM", "AMD", "QCOM", "NFLX"]
STRATEGIES = ["Agent", "LM", "LLM", "OFI"]
HORIZONS   = [1, 5, 15]


def _load_ic(ticker: str, suffix: str) -> pd.DataFrame | None:
    path = AGENT_RESULTS / f"agent_ic_{ticker}_{suffix}_llm.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


def _fmt_ic(comp: pd.DataFrame, strat: str, h: int) -> str:
    sub = comp[(comp["strategy"] == strat) & (comp["horizon"] == h)]
    if sub.empty or pd.isna(sub.iloc[0]["ic"]):
        return f"{'NaN':>12}"
    v   = sub.iloc[0]["ic"]
    pv  = sub.iloc[0]["pval"]
    sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else " "
    return f"{v:>+11.4f}{sig}"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize LNA-Agent IC results")
    parser.add_argument("--suffix", required=True,
                        help="Date suffix, e.g. 2016-01-01_2020-12-31")
    args   = parser.parse_args()
    suffix = args.suffix

    print(f"\n{'═' * 90}")
    print(f"  LNA-Agent Results Summary — {suffix}")
    print(f"{'═' * 90}")

    # ── Cross-ticker IC table ─────────────────────────────────────────────────
    col_labels = [f"{s} {h}m" for s in STRATEGIES for h in HORIZONS]
    print(f"\n  {'Ticker':<8}  " + "  ".join(f"{lbl:>12}" for lbl in col_labels))
    print(f"  {'-'*8}  " + "  ".join(f"{'-'*12}" for _ in col_labels))

    all_rows: list[pd.DataFrame] = []

    for ticker in TICKERS:
        comp = _load_ic(ticker, suffix)
        if comp is None:
            continue
        all_rows.append(comp.assign(ticker=ticker))

        parts = [f"  {ticker:<8}"]
        for strat in STRATEGIES:
            for h in HORIZONS:
                parts.append(f"  {_fmt_ic(comp, strat, h)}")
        print("".join(parts))

    # ── Chosen-horizon summary ────────────────────────────────────────────────
    if all_rows:
        print(f"\n  {'Ticker':<8}  {'Agent chosen horizon':>22}  {'N':>5}")
        print(f"  {'-'*8}  {'-'*22}  {'-'*5}")
        for df in all_rows:
            ticker = df["ticker"].iloc[0]
            sub    = df[df["strategy"] == "Agent chosen horizon"]
            if sub.empty or pd.isna(sub.iloc[0]["ic"]):
                print(f"  {ticker:<8}  {'NaN':>22}  {0:>5}")
                continue
            v   = sub.iloc[0]["ic"]
            pv  = sub.iloc[0]["pval"]
            n   = int(sub.iloc[0]["n"])
            sig = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else " "
            print(f"  {ticker:<8}  {v:>+21.4f}{sig}  {n:>5}")

    # ── Stratified IC tables ──────────────────────────────────────────────────
    for label, glob_pat in [
        ("Size-stratified IC",       f"size_ic_*_{suffix}_llm.csv"),
        ("Relevance-stratified IC",  f"relevance_ic_*_{suffix}_llm.csv"),
        ("Agreement-stratified IC",  f"agreement_ic_*_{suffix}_llm.csv"),
    ]:
        files = sorted(AGENT_RESULTS.glob(glob_pat))
        if not files:
            continue
        print(f"\n  {'─' * 60}")
        print(f"  {label}")
        print(f"  {'─' * 60}")
        for path in files:
            # Filename pattern: {type}_ic_{TICKER}_{suffix}_{mode}.csv
            parts  = path.stem.split("_")
            ticker = parts[2]
            try:
                df = pd.read_csv(path)
            except Exception as e:
                print(f"  [WARN] Could not read {path.name}: {e}")
                continue
            print(f"\n  {ticker}:")
            print(df.to_string(index=False))

    # ── Save combined CSV ─────────────────────────────────────────────────────
    if all_rows:
        combined  = pd.concat(all_rows, ignore_index=True)
        out_path  = RESULTS / f"summary_{suffix}.csv"
        combined.to_csv(out_path, index=False)
        print(f"\n  Saved summary → {out_path}")

    print()


if __name__ == "__main__":
    main()
