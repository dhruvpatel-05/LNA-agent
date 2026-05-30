"""
signal.py — IC analysis and sentiment-conditional OFI regression
LNA-Agent: Lobster News Alpha

Tests five predictors for 1/5/15-min forward returns:
  1. LM alone       — bag-of-words finance lexicon baseline
  2. LLM alone      — Anthropic claude-haiku-4-5-20251001 context-aware scores
  3. OFI × LM       — order flow × rule-based sentiment interaction
  4. OFI × LLM      — primary hypothesis: order flow × LLM sentiment
  5. RavenPack      — professional NLP benchmark (Koudijs & Voth style)

Core hypothesis: if LLM captures richer context than LM, then:
  IC(OFI × LLM) > IC(OFI × LM) > IC(LM) ≈ IC(LLM) > IC(OFI alone)

Regression model (sentiment-conditional OFI):
  ret_t = β⁺ · (OFI_t · sent⁺_t) + β⁻ · (OFI_t · sent⁻_t) + ε_t
  sent⁺ = max(sent, 0),  sent⁻ = min(sent, 0)

  β⁺ ≠ β⁻ means OFI predicts returns differently under positive vs negative news —
  consistent with informed trading clustering around sentiment-driven events.

Usage:
  python src/signal.py --ticker AAPL --suffix 2019-01-01_2020-12-31
  python src/signal.py --test   # synthetic smoke-test, no files needed
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

import sys
import argparse
import warnings
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import spearmanr
from scipy.stats import t as t_dist
from statsmodels.regression.linear_model import OLS
from statsmodels.tools.tools import add_constant as sm_add_constant
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS  = Path("results")
HORIZONS = [1, 5, 15]
# ──────────────────────────────────────────────────────────────────────────────

# Strategy definitions: name → function that extracts the predictor Series
_STRATEGIES = {
    "LM":        lambda df: df["lm_score"],
    "LLM":       lambda df: df["llm_score"],
    "OFI x LM":  lambda df: df["ofi_z"] * df["lm_score"],
    "OFI x LLM": lambda df: df["ofi_z"] * df["llm_score"],
    "RavenPack": lambda df: df["ravenpack_score"],
}


def compute_ic_table(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman IC at 1/5/15 min for all five strategies.

    Spearman IC = rank correlation between predictor and forward return.
    Robust to outliers and doesn't assume linear relationships.

    Under H0 (no signal): IC ~ N(0, 1/√n), so |IC| > 2/√n is significant.
    Under H1 (LNA hypothesis): IC(OFI×LLM) > IC(OFI×LM) > IC(LM).

    Strategies with fewer than 10 valid observations return IC=NaN.
    Returns DataFrame with columns: strategy, horizon, ic, pval, n.
    """
    rows = []
    for name, fn in _STRATEGIES.items():
        try:
            signal = fn(aligned_df)
        except KeyError:
            continue

        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in aligned_df.columns:
                continue
            mask = signal.notna() & aligned_df[ret_col].notna()
            n    = int(mask.sum())
            if n < 10:
                rows.append(
                    {"strategy": name, "horizon": h, "ic": np.nan, "pval": np.nan, "n": n}
                )
                continue
            ic, pval = spearmanr(signal[mask], aligned_df.loc[mask, ret_col])
            rows.append({"strategy": name, "horizon": h, "ic": float(ic), "pval": float(pval), "n": n})

    return pd.DataFrame(rows)


def run_regression(
    aligned_df: pd.DataFrame,
    sentiment_col: str = "llm_score",
) -> dict[int, dict]:
    """
    Sentiment-conditional OFI regression at each forward return horizon.

      ret_t = β⁺ · (OFI_t · sent⁺_t) + β⁻ · (OFI_t · sent⁻_t) + ε_t

    where sent⁺ = max(sent, 0), sent⁻ = min(sent, 0).

    HC3 heteroskedasticity-robust standard errors.
    Tests β⁺ = β⁻ using a Wald t-test on the coefficient difference.

    Returns dict keyed by horizon (int) with fields:
      beta_pos, beta_neg, tstat_diff, pval_diff, r2, n
    """
    df = aligned_df.dropna(subset=["ofi_z", sentiment_col]).copy()
    if df.empty:
        return {}

    sent     = df[sentiment_col]
    sent_pos = sent.clip(lower=0)
    sent_neg = sent.clip(upper=0)

    X_base = pd.DataFrame(
        {
            "ofi_pos": df["ofi_z"] * sent_pos,
            "ofi_neg": df["ofi_z"] * sent_neg,
        },
        index=df.index,
    )
    X_base = sm_add_constant(X_base)

    results = {}
    for h in HORIZONS:
        ret_col = f"ret_{h}m"
        if ret_col not in df.columns:
            continue
        y   = df[ret_col]
        idx = X_base.index.intersection(y.dropna().index)
        if len(idx) < 10:
            continue
        try:
            ols   = OLS(y.loc[idx], X_base.loc[idx]).fit(cov_type="HC3")
            b_pos = float(ols.params.get("ofi_pos", np.nan))
            b_neg = float(ols.params.get("ofi_neg", np.nan))
            se_pos = float(ols.bse.get("ofi_pos", np.nan))
            se_neg = float(ols.bse.get("ofi_neg", np.nan))

            # Wald test: β⁺ − β⁻ = 0
            denom  = np.sqrt(se_pos**2 + se_neg**2 + 1e-14)
            tstat  = (b_pos - b_neg) / denom
            pval   = float(2 * t_dist.sf(abs(tstat), df=ols.df_resid))

            results[h] = {
                "horizon":    h,
                "beta_pos":   b_pos,
                "beta_neg":   b_neg,
                "tstat_diff": float(tstat),
                "pval_diff":  pval,
                "r2":         float(ols.rsquared),
                "n":          int(ols.nobs),
            }
        except Exception as e:
            print(f"  [WARN] Regression failed at {h}m horizon: {e}")

    return results


def plot_ic_comparison(
    ic_table: pd.DataFrame,
    ticker: str,
    suffix: str,
) -> Path:
    """
    Bar chart comparing Spearman IC across strategies and horizons.

    Layout: one subplot per horizon, strategies on x-axis.
    Stars mark statistical significance (*, **, ***).
    Saves to results/ic_comparison_{ticker}_{suffix}.png.
    """
    RESULTS.mkdir(exist_ok=True)

    strategies = list(_STRATEGIES.keys())
    # Keep only strategies that appear in ic_table
    strategies = [s for s in strategies if s in ic_table["strategy"].values]
    n_strats   = len(strategies)
    n_h        = len(HORIZONS)

    colors = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 5), sharey=False)
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, HORIZONS):
        sub   = ic_table[ic_table["horizon"] == h].set_index("strategy")
        ics   = [sub.at[s, "ic"]   if s in sub.index else np.nan for s in strategies]
        pvals = [sub.at[s, "pval"] if s in sub.index else np.nan for s in strategies]
        ns    = [sub.at[s, "n"]    if s in sub.index else 0       for s in strategies]

        xs = np.arange(n_strats)
        for i, (ic_v, pv, n) in enumerate(zip(ics, pvals, ns)):
            color = colors[i % len(colors)]
            ax.bar(i, ic_v if not np.isnan(ic_v) else 0,
                   color=color, alpha=0.82, edgecolor="white", linewidth=0.7)

            # Significance annotation
            if not (np.isnan(ic_v) or np.isnan(pv)):
                stars = "***" if pv < 0.01 else "**" if pv < 0.05 else "*" if pv < 0.10 else ""
                if stars:
                    y_off = 0.003 if ic_v >= 0 else -0.007
                    ax.text(i, ic_v + y_off, stars, ha="center", va="bottom",
                            fontsize=8, fontweight="bold")
            # Sample size below bar
            ax.text(i, ax.get_ylim()[0] if ic_v >= 0 else ic_v,
                    f"n={n}", ha="center", va="top", fontsize=6.5, color="#555555")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(xs)
        ax.set_xticklabels(strategies, rotation=28, ha="right", fontsize=8.5)
        ax.set_title(f"{h}-min return", fontsize=11, pad=6)
        ax.set_ylabel("Spearman IC", fontsize=9)
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", alpha=0.25, linewidth=0.5)

    # Legend for significance
    fig.text(
        0.5, -0.04,
        "Stars: * p<0.10  ** p<0.05  *** p<0.01",
        ha="center", fontsize=8.5, color="#444444",
    )
    fig.suptitle(
        f"LNA-Agent: IC Comparison — {ticker}  [{suffix}]",
        fontsize=12, y=1.01,
    )
    plt.tight_layout()

    out = RESULTS / f"ic_comparison_{ticker}_{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart saved → {out}")
    return out


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _print_ic_table(ic_table: pd.DataFrame) -> None:
    print(f"\n{'═' * 68}")
    print(f"  IC TABLE  (Spearman rank correlation with forward return)")
    print(f"{'═' * 68}")
    print(f"  {'Strategy':<18}  {'1-min':>8}  {'5-min':>8}  {'15-min':>8}  {'N':>6}")
    print(f"  {'-' * 18}  {'-' * 8}  {'-' * 8}  {'-' * 8}  {'-' * 6}")
    for strat in _STRATEGIES:
        sub  = ic_table[ic_table["strategy"] == strat]
        if sub.empty:
            continue
        ics  = {int(r["horizon"]): r["ic"]   for _, r in sub.iterrows()}
        pvs  = {int(r["horizon"]): r["pval"] for _, r in sub.iterrows()}
        n_   = int(sub[sub["horizon"] == HORIZONS[0]]["n"].values[0]) if HORIZONS[0] in ics else 0

        def fmt(h):
            v = ics.get(h, np.nan)
            p = pvs.get(h, np.nan)
            if np.isnan(v):
                return f"{'N/A':>8}"
            sig = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            return f"{v:>+7.4f}{sig:<1}"

        vals = "  ".join(fmt(h) for h in HORIZONS)
        print(f"  {strat:<18}  {vals}  {n_:>6}")
    print()


def _print_regression(results: dict[int, dict], sentiment_col: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  REGRESSION  [{sentiment_col}]")
    print(f"  ret = β⁺·(OFI·sent⁺) + β⁻·(OFI·sent⁻) + ε")
    print(f"{'═' * 68}")
    print(
        f"  {'Horizon':<10}  {'β⁺':>10}  {'β⁻':>10}  "
        f"{'t(β⁺-β⁻)':>12}  {'p-val':>8}  {'R²':>7}"
    )
    print(
        f"  {'-' * 10}  {'-' * 10}  {'-' * 10}  "
        f"{'-' * 12}  {'-' * 8}  {'-' * 7}"
    )
    for h, r in sorted(results.items()):
        sig = "***" if r["pval_diff"] < 0.01 else "**" if r["pval_diff"] < 0.05 else "*" if r["pval_diff"] < 0.10 else ""
        print(
            f"  {h} min{'':<6}  "
            f"{r['beta_pos']:>10.5f}  {r['beta_neg']:>10.5f}  "
            f"{r['tstat_diff']:>12.3f}  {r['pval_diff']:>8.4f}{sig:>3}  "
            f"{r['r2']:>7.5f}"
        )
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IC and regression analysis — LNA-Agent")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--test",   action="store_true",
                        help="Run on synthetic aligned data (no files needed)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))

    if args.test:
        print("\n[TEST MODE] Building synthetic aligned panel")
        from align import (
            _make_test_ofi_df,
            _make_test_headlines_df,
            align as do_align,
            filter_market_hours,
            filter_relevance,
        )
        ofi_df       = _make_test_ofi_df()
        headlines_df = _make_test_headlines_df()
        headlines_df = filter_market_hours(headlines_df)
        headlines_df = filter_relevance(headlines_df, min_relevance=0.0)
        aligned = do_align(headlines_df, ofi_df)
        suffix  = "test"
    else:
        from align import load_aligned
        aligned = load_aligned(args.ticker, args.suffix)
        suffix  = args.suffix

    ticker = args.ticker
    print(f"\n  Aligned rows: {len(aligned)}")

    # ── IC table ─────────────────────────────────────────────────────────────
    ic_table = compute_ic_table(aligned)
    _print_ic_table(ic_table)

    ic_path = RESULTS / f"ic_table_{ticker}_{suffix}.csv"
    ic_table.to_csv(ic_path, index=False)
    print(f"  IC table saved → {ic_path}")

    # ── Regression ───────────────────────────────────────────────────────────
    for sent_col in ["llm_score", "lm_score"]:
        if sent_col in aligned.columns and aligned[sent_col].notna().any():
            reg = run_regression(aligned, sentiment_col=sent_col)
            if reg:
                _print_regression(reg, sent_col)
                reg_df   = pd.DataFrame(reg.values())
                reg_path = RESULTS / f"regression_{ticker}_{suffix}_{sent_col}.csv"
                reg_df.to_csv(reg_path, index=False)
                print(f"  Regression saved → {reg_path}")

    # ── IC comparison chart ───────────────────────────────────────────────────
    if not ic_table.empty:
        plot_ic_comparison(ic_table, ticker, suffix)
