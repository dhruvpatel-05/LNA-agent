"""
run_all.py — Multi-ticker LNA-Agent pipeline
LNA-Agent: Lobster News Alpha

Runs the full pipeline (news → sentiment → ofi → align → ic_signal) for each
of the five tickers defined in TICKERS, then pools the per-ticker IC tables
into a cross-ticker aggregate.

Per ticker:
  Stage 0 — OFI computation from LOBSTER CSVs (skipped if parquet/CSV exists)
  Stage 1 — Load Benzinga headlines, market-hours filter
  Stage 2 — LM + LLM sentiment scoring
  Stage 3 — merge_asof alignment to 1-min LOB bars
  Stage 4 — Spearman IC table + per-ticker chart

Aggregate (after all tickers):
  - Mean IC / mean p-value / total n across tickers
  - Aggregate bar chart with ±1σ error bars
  - Printed aggregate table

Flags:
  --skip-existing  Skip stages 1-2 if sentiment_{ticker}_{suffix}.csv exists.
  --no-llm         Skip LLM scoring (LM only); useful for a quick first pass.

Usage:
  python src/run_all.py
  python src/run_all.py --skip-existing
  python src/run_all.py --no-llm
  python src/run_all.py --skip-existing --no-llm
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
import traceback
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Config ─────────────────────────────────────────────────────────────────────
DATE_FROM = "2016-01-01"
DATE_TO   = "2020-12-31"

# LOB folders follow: /Users/dhruvpatel/Downloads/_data_dwn_32_302__{TICKER}_2007-06-27_2021-07-01_10_60
_LOB_BASE = "/Users/dhruvpatel/Downloads"
_LOB_STEM = "_data_dwn_32_302__{ticker}_2007-06-27_2021-07-01_10_60"

def _lob(ticker: str) -> str:
    return f"{_LOB_BASE}/{_LOB_STEM.format(ticker=ticker)}"

TICKERS: dict[str, str] = {
    "AAPL": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60",
    "JPM":  "/Users/dhruvpatel/Downloads/_data_dwn_32_302__JPM_2007-06-27_2021-07-01_10_60",
    "AMD":  "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AMD_2007-06-27_2022-01-01_1_60",
    "QCOM": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__QCOM_2007-06-27_2021-07-01_10_60",
    "NFLX": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__NFLX_2007-06-27_2021-07-01_10_60",
}

RESULTS    = Path("results")
HORIZONS   = [1, 5, 15]
STRATEGIES = ["LM", "LLM", "OFI", "OFI x LM", "OFI x LLM"]
# ───────────────────────────────────────────────────────────────────────────────

sys.path.insert(0, str(Path(__file__).parent))


# ── Helpers ───────────────────────────────────────────────────────────────────

class _Timer:
    def __init__(self, label: str) -> None:
        self.label = label

    def __enter__(self):
        print(f"  ┌ {self.label}")
        self._t = time.perf_counter()
        return self

    def __exit__(self, *_):
        print(f"  └ done in {time.perf_counter() - self._t:.1f}s")


def _banner(msg: str, char: str = "═") -> None:
    width = 62
    print(f"\n{char * width}")
    print(f"  {msg}")
    print(f"{char * width}")


def _ofi_exists(ticker: str, suffix: str) -> bool:
    return (
        (RESULTS / f"ofi_data_{ticker}_{suffix}.csv").exists() or
        (RESULTS / f"ofi_data_{ticker}_{suffix}.parquet").exists()
    )


# ── Per-ticker pipeline ────────────────────────────────────────────────────────

def run_ticker(
    ticker: str,
    lob_dir: str,
    suffix: str,
    llm: bool,
    skip_existing: bool,
) -> pd.DataFrame:
    """
    Run the full pipeline for one ticker.

    Returns the IC table DataFrame (columns: strategy, horizon, ic, pval, n).
    Raises on any unrecoverable error so the caller can catch and continue.
    """
    date_from, date_to = suffix.split("_", 1)
    # suffix is "2016-01-01_2020-06-10" — split at first underscore after date
    # Actually DATE_FROM and DATE_TO are the canonical values; derive from them.
    date_from = DATE_FROM
    date_to   = DATE_TO

    sentiment_path = RESULTS / f"sentiment_{ticker}_{suffix}.csv"

    # ── Stage 0: OFI ─────────────────────────────────────────────────────────
    if not _ofi_exists(ticker, suffix):
        with _Timer("OFI computation"):
            from ofi import run_batch
            run_batch(
                ticker    = ticker,
                data_dir  = Path(lob_dir),
                date_from = date_from,
                date_to   = date_to,
            )
    else:
        ofi_path = (RESULTS / f"ofi_data_{ticker}_{suffix}.parquet"
                    if (RESULTS / f"ofi_data_{ticker}_{suffix}.parquet").exists()
                    else RESULTS / f"ofi_data_{ticker}_{suffix}.csv")
        print(f"  OFI exists: {ofi_path.name}")

    # ── Stages 1+2: News + Sentiment (skippable) ─────────────────────────────
    if skip_existing and sentiment_path.exists():
        from sentiment import load_scored
        scored = load_scored(ticker, suffix)
        print(f"  [skip] Loaded existing sentiment — {len(scored)} headlines")
    else:
        with _Timer("Stage 1 — headlines"):
            from news import load_headlines, save_headlines
            hl_df = load_headlines(ticker, date_from, date_to)
            if hl_df.empty:
                raise ValueError(
                    f"No headlines found for {ticker} in {date_from}→{date_to}. "
                    f"Check that this ticker exists in the Kaggle Benzinga archive."
                )
            save_headlines(ticker, hl_df, suffix)
            print(f"  {len(hl_df)} headlines  "
                  f"({hl_df['timestamp'].min().date()} → {hl_df['timestamp'].max().date()})")

        with _Timer("Stage 2 — sentiment"):
            from sentiment import score_all, save_scored
            scored = score_all(hl_df, llm=llm, verbose=True)
            save_scored(ticker, scored, suffix)
            nonzero = scored["lm_score"].ne(0).sum()
            print(f"  LM nonzero: {nonzero}/{len(scored)}", end="")
            if llm and scored["llm_score"].notna().any():
                print(f"  |  LLM mean: {scored['llm_score'].mean():+.3f}", end="")
            print()

    # ── Stage 3: Align ────────────────────────────────────────────────────────
    with _Timer("Stage 3 — align"):
        from align import load_ofi, align, save_aligned
        ofi_df  = load_ofi(ticker, suffix)
        aligned = align(scored, ofi_df)
        if aligned.empty:
            raise ValueError(
                f"0 events aligned for {ticker}. "
                f"News window ({scored['timestamp'].min().date()} → "
                f"{scored['timestamp'].max().date()}) may not overlap "
                f"with OFI window ({ofi_df.index.min().date()} → "
                f"{ofi_df.index.max().date()})."
            )
        save_aligned(ticker, aligned, suffix)
        drop_rate = (len(scored) - len(aligned)) / len(scored) * 100
        print(f"  {len(aligned)} aligned events  ({drop_rate:.1f}% dropped)")

    # ── Stage 4: IC + per-ticker chart ────────────────────────────────────────
    with _Timer("Stage 4 — IC analysis"):
        from ic_signal import compute_ic_table, plot_ic_comparison, _print_ic
        ic = compute_ic_table(aligned)
        _print_ic(ic)

        # Save per-ticker IC with the name the user specified
        ic_path = RESULTS / f"ic_{ticker}_{suffix}.csv"
        ic.to_csv(ic_path, index=False)
        print(f"  IC table → {ic_path}")

        # Also save under the standard ic_table_ name for ic_signal.py compatibility
        ic.to_csv(RESULTS / f"ic_table_{ticker}_{suffix}.csv", index=False)

        plot_ic_comparison(ic, ticker, suffix)

    return ic


# ── Aggregate IC ──────────────────────────────────────────────────────────────

def compute_aggregate_ic(ic_tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Pool per-ticker IC tables into a cross-ticker aggregate.

    For each (strategy, horizon):
      ic       — mean Spearman IC across tickers with non-NaN values
      pval     — mean p-value (proxy; not a combined-test statistic)
      n        — sum of aligned events across all tickers
      n_tickers — number of tickers with non-NaN IC for this cell
    """
    all_rows = pd.concat(ic_tables.values(), ignore_index=True)

    def _nanmean(s):  return s.mean(skipna=True)
    def _nonnull(s):  return s.notna().sum()

    agg = (
        all_rows.groupby(["strategy", "horizon"], sort=False)
        .agg(
            ic        = ("ic",   _nanmean),
            pval      = ("pval", _nanmean),
            n         = ("n",    "sum"),
            n_tickers = ("ic",   _nonnull),
        )
        .reset_index()
    )
    return agg


def _print_aggregate_ic(
    agg: pd.DataFrame,
    ic_tables: dict[str, pd.DataFrame],
) -> None:
    tickers = list(ic_tables.keys())
    print(f"\n{'═' * 72}")
    print(f"  AGGREGATE IC TABLE  ({len(tickers)} tickers: {', '.join(tickers)})")
    print(f"{'═' * 72}")
    print(f"  {'Strategy':<18}  {'1-min':>10}  {'5-min':>10}  {'15-min':>10}  "
          f"{'N':>7}  {'k':>3}")
    print(f"  {'-' * 18}  {'-' * 10}  {'-' * 10}  {'-' * 10}  "
          f"{'-' * 7}  {'-' * 3}")

    def _fmt(strat: str, h: int) -> str:
        sub = agg[(agg["strategy"] == strat) & (agg["horizon"] == h)]
        if sub.empty:
            return f"{'—':>10}"
        ic_v = sub.iloc[0]["ic"]
        pv   = sub.iloc[0]["pval"]
        if np.isnan(ic_v):
            return f"{'NaN':>10}"
        sig = ("***" if pv < 0.01 else "**" if pv < 0.05
               else "*" if pv < 0.10 else " ")
        return f"{ic_v:>+9.4f}{sig}"

    for strat in STRATEGIES:
        sub = agg[agg["strategy"] == strat]
        if sub.empty:
            continue
        total_n   = int(sub["n"].sum())
        n_tickers = int(sub["n_tickers"].max())
        row = f"  {strat:<18}  " + "  ".join(_fmt(strat, h) for h in HORIZONS)
        print(f"{row}  {total_n:>7}  {n_tickers:>3}")
    print()


def plot_aggregate_ic(
    agg: pd.DataFrame,
    ic_tables: dict[str, pd.DataFrame],
    out_path: Path,
) -> None:
    """
    Grouped bar chart of mean IC across tickers, with ±1σ error bars.

    Each subplot is one forward-return horizon.
    Bars show mean IC; error bars show ±1 standard deviation across tickers,
    indicating cross-ticker consistency of the signal.
    Stars are based on the mean p-value (indicative, not a combined test).
    """
    all_rows = pd.concat(ic_tables.values(), ignore_index=True)
    n_h      = len(HORIZONS)
    colors   = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 5), sharey=False)
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, HORIZONS):
        sub_agg = agg[agg["horizon"] == h].set_index("strategy")
        sub_all = all_rows[all_rows["horizon"] == h]

        for i, strat in enumerate(STRATEGIES):
            mean_ic = sub_agg.at[strat, "ic"]   if strat in sub_agg.index else np.nan
            pval    = sub_agg.at[strat, "pval"] if strat in sub_agg.index else np.nan
            n_tick  = int(sub_agg.at[strat, "n_tickers"]) if strat in sub_agg.index else 0

            # Standard deviation of IC across tickers
            per_ticker_ics = (sub_all[sub_all["strategy"] == strat]["ic"].dropna())
            std_ic = per_ticker_ics.std() if len(per_ticker_ics) > 1 else 0.0

            bar_h = mean_ic if not np.isnan(mean_ic) else 0.0
            ax.bar(
                i, bar_h,
                color=colors[i % len(colors)], alpha=0.82,
                edgecolor="white", linewidth=0.7,
                yerr=std_ic if std_ic > 0 else None,
                error_kw={"ecolor": "#555", "capsize": 4, "elinewidth": 1.2},
            )

            # Significance stars (based on mean pval)
            if not (np.isnan(mean_ic) or np.isnan(pval)):
                stars = ("***" if pval < 0.01 else "**" if pval < 0.05
                         else "*" if pval < 0.10 else "")
                if stars:
                    y_off = (std_ic + 0.004) if bar_h >= 0 else -(std_ic + 0.008)
                    ax.text(i, bar_h + y_off, stars, ha="center",
                            fontsize=8.5, fontweight="bold")

            # n_tickers label below bar
            ax.text(i, min(0.0, bar_h) - 0.002, f"k={n_tick}",
                    ha="center", va="top", fontsize=6.5, color="#555")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(range(len(STRATEGIES)))
        ax.set_xticklabels(STRATEGIES, rotation=28, ha="right", fontsize=9)
        ax.set_title(f"{h}-min forward return", fontsize=11, pad=6)
        ax.set_ylabel("Mean Spearman IC (± 1σ)")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    tickers_str = ", ".join(ic_tables.keys())
    fig.text(
        0.5, -0.04,
        f"* p<0.10   ** p<0.05   *** p<0.01  (mean p-values)  |  "
        f"error bars = ±1σ across {len(ic_tables)} tickers",
        ha="center", fontsize=8, color="#444",
    )
    fig.suptitle(
        f"LNA-Agent Aggregate IC — {tickers_str}  [{DATE_FROM} → {DATE_TO}]",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Aggregate chart → {out_path}")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Multi-ticker LNA-Agent pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip news+sentiment stages if sentiment_{ticker}_{suffix}.csv exists",
    )
    parser.add_argument(
        "--no-llm", action="store_true",
        help="LM scoring only; skip Anthropic API calls",
    )
    args = parser.parse_args()

    llm           = not args.no_llm
    skip_existing = args.skip_existing
    suffix        = f"{DATE_FROM}_{DATE_TO}"

    RESULTS.mkdir(exist_ok=True)

    _banner(f"LNA-Agent Multi-Ticker Run  |  {DATE_FROM} → {DATE_TO}")
    print(f"  Tickers       : {', '.join(TICKERS)}")
    print(f"  LLM scoring   : {'ON' if llm else 'OFF'}")
    print(f"  Skip existing : {skip_existing}")

    # ── Per-ticker loop ───────────────────────────────────────────────────────
    ic_tables:   dict[str, pd.DataFrame] = {}
    errors:      dict[str, str]          = {}
    t_total_start = time.perf_counter()

    for ticker, lob_dir in TICKERS.items():
        _banner(f"{ticker}  |  {lob_dir.split('/')[-1]}", char="─")

        t0 = time.perf_counter()
        try:
            ic = run_ticker(
                ticker        = ticker,
                lob_dir       = lob_dir,
                suffix        = suffix,
                llm           = llm,
                skip_existing = skip_existing,
            )
            ic_tables[ticker] = ic
            elapsed = time.perf_counter() - t0
            print(f"\n  ✓ {ticker} complete in {elapsed:.1f}s")

        except Exception as exc:
            elapsed = time.perf_counter() - t0
            errors[ticker] = str(exc)
            print(f"\n  ✗ {ticker} FAILED after {elapsed:.1f}s: {exc}")
            print("  Traceback:")
            for line in traceback.format_exc().splitlines():
                print(f"    {line}")

    # ── Aggregate ─────────────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - t_total_start
    _banner(f"Aggregate Results  ({len(ic_tables)}/{len(TICKERS)} tickers succeeded)")
    print(f"  Total wall time: {total_elapsed:.0f}s")

    if errors:
        print(f"\n  Failed tickers:")
        for t, msg in errors.items():
            print(f"    {t}: {msg}")

    if not ic_tables:
        print("\n  No tickers succeeded — nothing to aggregate.")
        raise SystemExit(1)

    if len(ic_tables) == 1:
        print("\n  Only one ticker succeeded; skipping cross-ticker aggregate.")
    else:
        agg = compute_aggregate_ic(ic_tables)

        # Save
        agg_path = RESULTS / "ic_aggregate_all_tickers.csv"
        agg.to_csv(agg_path, index=False)
        print(f"  Aggregate IC → {agg_path}")

        # Print
        _print_aggregate_ic(agg, ic_tables)

        # Plot
        plot_aggregate_ic(
            agg,
            ic_tables,
            out_path=RESULTS / "ic_comparison_all_tickers.png",
        )

    # ── Final file manifest ───────────────────────────────────────────────────
    print(f"\n  Output files:")
    for ticker in ic_tables:
        for name in [
            f"results/ic_{ticker}_{suffix}.csv",
            f"results/ic_comparison_{ticker}_{suffix}.png",
        ]:
            status = "✓" if Path(name).exists() else "✗"
            print(f"    {status}  {name}")
    for name in [
        "results/ic_aggregate_all_tickers.csv",
        "results/ic_comparison_all_tickers.png",
    ]:
        if Path(name).exists():
            print(f"    ✓  {name}")
