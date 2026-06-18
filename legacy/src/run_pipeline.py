"""
run_pipeline.py — End-to-end LNA-Agent pipeline
LNA-Agent: Lobster News Alpha

Runs all four stages in order:
  1. news.py       — load SentARL headlines, filter to market hours
  2. sentiment.py  — LM score (+ LLM score if --llm flag given)
  3. align.py      — merge_asof with LOBSTER OFI bars
  4. ic_signal.py  — Spearman IC table, regression, IC comparison chart

LOBSTER folder naming convention:
  /Users/dhruvpatel/Downloads/_data_dwn_32_302__{TICKER}_2007-06-27_2021-07-01_10_60

Usage:
  # Full pipeline, LM only (no API cost)
  python src/run_pipeline.py --ticker AAPL --date-from 2016-01-01 --date-to 2020-12-31

  # With LLM sentiment
  python src/run_pipeline.py --ticker AAPL --date-from 2016-01-01 --date-to 2020-12-31 --llm

  # Provide LOB directory to (re)compute OFI from raw LOBSTER files
  python src/run_pipeline.py --ticker AAPL --date-from 2016-01-01 --date-to 2020-12-31 \\
    --lob-dir /Users/dhruvpatel/Downloads/_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60

OFI note: if results/ofi_data_{ticker}_{suffix}.csv already exists it is used as-is.
Pass --lob-dir to recompute OFI from raw LOBSTER CSV files.
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

import sys
import time
import argparse
import pandas as pd
from pathlib import Path

# ── Timing helper ─────────────────────────────────────────────────────────────
class _Timer:
    def __init__(self, label):
        self.label = label

    def __enter__(self):
        print(f"\n{'─' * 60}")
        print(f"  STEP: {self.label}")
        print(f"{'─' * 60}")
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed = time.perf_counter() - self._t
        print(f"  Done in {elapsed:.1f}s")


def _banner(msg: str) -> None:
    print(f"\n{'═' * 60}")
    print(f"  {msg}")
    print(f"{'═' * 60}")


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(
    ticker: str,
    date_from: str,
    date_to: str,
    llm: bool = False,
    lob_dir: str | None = None,
) -> None:
    """
    Execute all four pipeline stages for `ticker` over [date_from, date_to].

    Stage 0 (optional): recompute OFI from LOBSTER data if --lob-dir is given.
    Stage 1: load_headlines — Kaggle Benzinga CSV, market-hours filter
    Stage 2: score_all     — LM scoring, optional LLM scoring
    Stage 3: align         — merge_asof with precomputed OFI bars
    Stage 4: compute_ic_table + run_regression + plot_ic_comparison
    """
    src_dir = Path(__file__).parent
    sys.path.insert(0, str(src_dir))

    suffix = f"{date_from}_{date_to}"
    _banner(f"LNA-Agent Pipeline  |  {ticker}  |  {date_from} → {date_to}")
    print(f"  LLM scoring : {'ON' if llm else 'OFF (use --llm to enable)'}")
    print(f"  Suffix      : {suffix}")

    # ── Stage 0: OFI (optional recompute) ────────────────────────────────────
    ofi_csv = Path("results") / f"ofi_data_{ticker}_{suffix}.csv"
    ofi_par = Path("results") / f"ofi_data_{ticker}_{suffix}.parquet"

    if lob_dir:
        with _Timer("OFI computation (ofi.py)"):
            from ofi import run_batch
            run_batch(
                ticker=ticker,
                data_dir=Path(lob_dir),
                date_from=date_from,
                date_to=date_to,
            )
    elif not ofi_csv.exists() and not ofi_par.exists():
        print(
            f"\n  WARNING: No OFI file found for {ticker}_{suffix}.\n"
            f"  Expected: {ofi_csv}\n"
            f"  Pass --lob-dir to compute OFI from raw LOBSTER data,\n"
            f"  or ensure the precomputed file is in results/."
        )
        raise SystemExit(1)
    else:
        print(f"\n  Using existing OFI: {ofi_csv if ofi_csv.exists() else ofi_par}")

    # ── Stage 1: News ─────────────────────────────────────────────────────────
    with _Timer("Stage 1 — Load headlines (news.py)"):
        from news import load_headlines, save_headlines
        hl_df = load_headlines(ticker, date_from, date_to)
        if hl_df.empty:
            print(f"\n  ERROR: No headlines found for {ticker} in {date_from}→{date_to}.")
            print("  Check SentARL coverage: python src/news.py --scan --ticker {ticker}")
            raise SystemExit(1)
        save_headlines(ticker, hl_df, suffix)
        print(f"  Headlines in market hours : {len(hl_df)}")
        print(f"  Date range                : {hl_df['timestamp'].min().date()} "
              f"→ {hl_df['timestamp'].max().date()}")

    # ── Stage 2: Sentiment ────────────────────────────────────────────────────
    with _Timer("Stage 2 — Sentiment scoring (sentiment.py)"):
        from sentiment import ensure_scored, save_scored
        scored = ensure_scored(ticker, suffix, hl_df, llm=llm, verbose=True)
        save_scored(ticker, scored, suffix)
        lm_nonzero = scored["lm_score"].ne(0).sum()
        print(f"  LM nonzero scores : {lm_nonzero}/{len(scored)}")
        if llm:
            print(f"  LLM mean score    : {scored['llm_score'].mean():+.4f}")

    # ── Stage 3: Align ────────────────────────────────────────────────────────
    with _Timer("Stage 3 — Align to LOB bars (align.py)"):
        from align import load_ofi, align, save_aligned
        ofi_df  = load_ofi(ticker, suffix)
        aligned = align(scored, ofi_df)
        if aligned.empty:
            print("\n  ERROR: 0 events aligned. News and OFI windows don't overlap.")
            raise SystemExit(1)
        save_aligned(ticker, aligned, suffix)
        print(f"  Aligned events : {len(aligned)}")
        print(f"  Drop rate      : {(len(scored) - len(aligned)) / len(scored) * 100:.1f}%")

    # ── Stage 4: IC analysis ──────────────────────────────────────────────────
    with _Timer("Stage 4 — IC analysis (ic_signal.py)"):
        from ic_signal import (
            compute_ic_table, run_regression,
            plot_ic_comparison, _print_ic, _print_regression,
        )

        ic_table = compute_ic_table(aligned)
        _print_ic(ic_table)

        ic_path = Path("results") / f"ic_table_{ticker}_{suffix}.csv"
        ic_table.to_csv(ic_path, index=False)
        print(f"  IC table → {ic_path}")

        for col in ["lm_score", "llm_score"]:
            if col in aligned.columns and aligned[col].notna().any():
                reg = run_regression(aligned, sentiment_col=col)
                if reg:
                    _print_regression(reg, col)
                    reg_path = Path("results") / f"regression_{ticker}_{suffix}_{col}.csv"
                    pd.DataFrame(reg.values()).to_csv(reg_path, index=False)
                    print(f"  Regression → {reg_path}")

        if not ic_table.empty:
            plot_ic_comparison(ic_table, ticker, suffix)

    # ── Final summary ─────────────────────────────────────────────────────────
    _banner("Pipeline complete")
    print(f"  Ticker     : {ticker}")
    print(f"  Period     : {date_from} → {date_to}")
    print(f"  Headlines  : {len(hl_df)}")
    print(f"  Aligned    : {len(aligned)}")
    print(f"  Outputs    :")
    for name in [
        f"data/news/processed/{ticker}_headlines_{suffix}.csv",
        f"results/sentiment/sentiment_{ticker}_{suffix}.csv",
        f"results/aligned_{ticker}_{suffix}.parquet",
        f"results/ic_table_{ticker}_{suffix}.csv",
        f"results/ic_comparison_{ticker}_{suffix}.png",
    ]:
        status = "✓" if Path(name).exists() else "✗ MISSING"
        print(f"    {status}  {name}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="LNA-Agent end-to-end pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--ticker",    type=str, required=True)
    parser.add_argument("--date-from", type=str, required=True)
    parser.add_argument("--date-to",   type=str, required=True)
    parser.add_argument("--llm",       action="store_true",
                        help="Enable LLM sentiment scoring (requires API access)")
    parser.add_argument("--lob-dir",   type=str, default=None,
                        help="Path to LOBSTER CSV folder to (re)compute OFI")
    args = parser.parse_args()

    run(
        ticker    = args.ticker,
        date_from = args.date_from,
        date_to   = args.date_to,
        llm       = args.llm,
        lob_dir   = args.lob_dir,
    )
