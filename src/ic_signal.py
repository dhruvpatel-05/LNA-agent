"""
ic_signal.py — IC analysis and sentiment-conditional OFI regression
LNA-Agent: Lobster News Alpha

Strategies evaluated at 1/5/15-min forward return horizons:
  1. LM alone       — Loughran-McDonald Polarity score
  2. LLM alone      — claude-haiku-4-5-20251001 score (NaN if --no-llm)
  3. OFI alone      — normalised order flow imbalance (already computed)
  4. OFI × LM       — interaction: ofi_z * lm_score
  5. OFI × LLM      — interaction: ofi_z * llm_score (primary hypothesis)

Core hypothesis: OFI × LLM captures directional information not available
from either signal alone, producing higher Spearman IC than OFI × LM.

Regression model (sentiment-conditional OFI):
  ret_t = β⁺ · (OFI_t · sent⁺_t) + β⁻ · (OFI_t · sent⁻_t) + ε_t
  sent⁺ = max(sent, 0),  sent⁻ = min(sent, 0)
  β⁺ ≠ β⁻ ⟹ OFI is more informative under one news tone

Usage:
  python src/ic_signal.py --ticker AAPL --suffix 2019-01-01_2020-12-31
  python src/ic_signal.py --test   # synthetic data smoke-test
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
import warnings
import argparse
import numpy as np
import pandas as pd
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

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS  = Path("results/ic_signal")
HORIZONS = [1, 5, 15]
# ─────────────────────────────────────────────────────────────────────────────

# Strategy registry — name → function(aligned_df) → pd.Series
_STRATEGIES: dict = {
    "LM":        lambda df: df["lm_score"],
    "LLM":       lambda df: df["llm_score"],
    "OFI":       lambda df: df["ofi_z"],
    "OFI x LM":  lambda df: df["ofi_z"] * df["lm_score"],
    "OFI x LLM": lambda df: df["ofi_z"] * df["llm_score"],
}


def compute_ic_table(aligned_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Spearman IC at 1/5/15-min horizons for all strategies.

    Spearman IC = rank correlation between predictor and forward return.
    Robust to outliers, distribution-free. |IC| > 2/√n ≈ significant at 5%.

    Strategies with insufficient non-null data (< 10 rows) return IC = NaN.
    LLM/OFI×LLM rows are NaN when --no-llm was used in sentiment.py.

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
                rows.append({"strategy": name, "horizon": h,
                             "ic": np.nan, "pval": np.nan, "n": n})
                continue
            ic, pval = spearmanr(signal[mask], aligned_df.loc[mask, ret_col])
            rows.append({"strategy": name, "horizon": h,
                         "ic": float(ic), "pval": float(pval), "n": n})

    return pd.DataFrame(rows)


def agreement_ic(aligned_df: pd.DataFrame, sentiment_col: str, horizon: int = 1) -> dict:
    """
    Spearman IC of `sentiment_col` vs ret_{horizon}m, split by whether the
    sentiment sign agrees ("confirmed") or disagrees ("conflicted") with
    sign(ofi_z). Rows where either sign is zero are excluded from both.

    Returns dict with keys "unconditional", "confirmed", "conflicted", each
    mapping to {"ic", "pval", "n"}.
    """
    ret_col = f"ret_{horizon}m"
    df = aligned_df[[sentiment_col, "ofi_z", ret_col]].dropna()

    out = {}
    mask_uncond = df[sentiment_col].notna() & df[ret_col].notna()
    out["unconditional"] = _spearman_or_nan(df.loc[mask_uncond, sentiment_col],
                                             df.loc[mask_uncond, ret_col])

    sign_sent = np.sign(df[sentiment_col])
    sign_ofi  = np.sign(df["ofi_z"])
    agree     = (sign_sent != 0) & (sign_ofi != 0) & (sign_sent == sign_ofi)
    conflict  = (sign_sent != 0) & (sign_ofi != 0) & (sign_sent != sign_ofi)

    out["confirmed"]  = _spearman_or_nan(df.loc[agree, sentiment_col],   df.loc[agree, ret_col])
    out["conflicted"] = _spearman_or_nan(df.loc[conflict, sentiment_col], df.loc[conflict, ret_col])

    return out


def _spearman_or_nan(x: pd.Series, y: pd.Series) -> dict:
    n = int(len(x))
    if n < 10:
        return {"ic": np.nan, "pval": np.nan, "n": n}
    ic, pval = spearmanr(x, y)
    return {"ic": float(ic), "pval": float(pval), "n": n}


def run_regression(
    aligned_df: pd.DataFrame,
    sentiment_col: str = "lm_score",
) -> dict:
    """
    Sentiment-conditional OFI regression at each forward return horizon.

      ret_t = β⁺ · (OFI_t · sent⁺_t) + β⁻ · (OFI_t · sent⁻_t) + ε_t

    sent⁺ = max(sent, 0),  sent⁻ = min(sent, 0)
    HC3 heteroskedasticity-robust standard errors.
    Wald test: H₀: β⁺ = β⁻  (OFI signal is tone-symmetric)
               H₁: β⁺ ≠ β⁻  (OFI more informative under one tone)

    Returns dict keyed by horizon with fields:
      beta_pos, beta_neg, tstat_diff, pval_diff, r2, n
    """
    df = aligned_df.dropna(subset=["ofi_z", sentiment_col]).copy()
    if len(df) < 10:
        return {}

    sent     = df[sentiment_col]
    sent_pos = sent.clip(lower=0)
    sent_neg = sent.clip(upper=0)

    X_base = sm_add_constant(pd.DataFrame({
        "ofi_pos": df["ofi_z"] * sent_pos,
        "ofi_neg": df["ofi_z"] * sent_neg,
    }, index=df.index))

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
            ols    = OLS(y.loc[idx], X_base.loc[idx]).fit(cov_type="HC3")
            b_pos  = float(ols.params.get("ofi_pos", np.nan))
            b_neg  = float(ols.params.get("ofi_neg", np.nan))
            se_pos = float(ols.bse.get("ofi_pos",   np.nan))
            se_neg = float(ols.bse.get("ofi_neg",   np.nan))
            tstat  = (b_pos - b_neg) / np.sqrt(se_pos**2 + se_neg**2 + 1e-14)
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
            print(f"  [WARN] Regression failed at {h}m: {e}")

    return results


def plot_ic_comparison(
    ic_table: pd.DataFrame,
    ticker: str,
    suffix: str,
) -> Path:
    """
    Grouped bar chart: IC by strategy and horizon.
    Bars with p < 0.10 annotated with *, **, ***.
    Saves to results/ic_comparison_{ticker}_{suffix}.png.
    """
    RESULTS.mkdir(exist_ok=True)

    strategies = [s for s in _STRATEGIES if s in ic_table["strategy"].values]
    n_h        = len(HORIZONS)
    colors     = ["#2E86AB", "#A23B72", "#F18F01", "#C73E1D", "#3B1F2B"]

    fig, axes = plt.subplots(1, n_h, figsize=(5 * n_h, 5), sharey=False)
    if n_h == 1:
        axes = [axes]

    for ax, h in zip(axes, HORIZONS):
        sub  = ic_table[ic_table["horizon"] == h].set_index("strategy")
        ics  = [sub.at[s, "ic"]   if s in sub.index else np.nan for s in strategies]
        pvs  = [sub.at[s, "pval"] if s in sub.index else np.nan for s in strategies]
        ns   = [int(sub.at[s, "n"]) if s in sub.index else 0    for s in strategies]

        for i, (ic_v, pv, n) in enumerate(zip(ics, pvs, ns)):
            bar_h = ic_v if not np.isnan(ic_v) else 0
            ax.bar(i, bar_h, color=colors[i % len(colors)], alpha=0.82,
                   edgecolor="white", linewidth=0.7)
            if not (np.isnan(ic_v) or np.isnan(pv)):
                stars = ("***" if pv < 0.01 else "**" if pv < 0.05
                         else "*" if pv < 0.10 else "")
                if stars:
                    ax.text(i, ic_v + (0.003 if ic_v >= 0 else -0.006),
                            stars, ha="center", va="bottom",
                            fontsize=8.5, fontweight="bold")
            ax.text(i, min(0, bar_h) - 0.002, f"n={n}",
                    ha="center", va="top", fontsize=6.5, color="#555")

        ax.axhline(0, color="black", lw=0.8, ls="--", alpha=0.5)
        ax.set_xticks(range(len(strategies)))
        ax.set_xticklabels(strategies, rotation=28, ha="right", fontsize=9)
        ax.set_title(f"{h}-min forward return", fontsize=11, pad=6)
        ax.set_ylabel("Spearman IC")
        ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))
        ax.grid(axis="y", alpha=0.25, lw=0.5)

    fig.text(0.5, -0.04, "* p<0.10   ** p<0.05   *** p<0.01",
             ha="center", fontsize=8.5, color="#444")
    fig.suptitle(f"LNA-Agent IC — {ticker}  [{suffix}]", fontsize=12, y=1.01)
    plt.tight_layout()

    out = RESULTS / f"ic_comparison_{ticker}_{suffix}.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Chart → {out}")
    return out


# ── Pretty-print helpers ──────────────────────────────────────────────────────

def _print_ic(ic_table: pd.DataFrame) -> None:
    print(f"\n{'═' * 68}")
    print(f"  SPEARMAN IC TABLE")
    print(f"{'═' * 68}")
    print(f"  {'Strategy':<18}  {'1-min':>9}  {'5-min':>9}  {'15-min':>9}  {'N':>5}")
    print(f"  {'-' * 18}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 5}")

    def _fmt(strat, h):
        sub = ic_table[(ic_table["strategy"] == strat) & (ic_table["horizon"] == h)]
        if sub.empty:
            return f"{'—':>9}"
        v = sub.iloc[0]["ic"]
        p = sub.iloc[0]["pval"]
        if np.isnan(v):
            return f"{'NaN':>9}"
        sig = ("***" if p < 0.01 else "**" if p < 0.05
               else "*" if p < 0.10 else " ")
        return f"{v:>+8.4f}{sig}"

    for strat in _STRATEGIES:
        sub = ic_table[ic_table["strategy"] == strat]
        if sub.empty:
            continue
        n_ = int(sub.iloc[0]["n"]) if not sub.empty else 0
        row = f"  {strat:<18}  " + "  ".join(_fmt(strat, h) for h in HORIZONS)
        print(f"{row}  {n_:>5}")
    print()


def _print_regression(results: dict, col: str) -> None:
    print(f"\n{'═' * 68}")
    print(f"  REGRESSION [{col}]  ret = β⁺·(OFI·sent⁺) + β⁻·(OFI·sent⁻)")
    print(f"{'═' * 68}")
    print(f"  {'Horiz':<8} {'β⁺':>11} {'β⁻':>11} "
          f"{'t(β⁺-β⁻)':>12} {'p':>8} {'R²':>8}")
    print(f"  {'-'*8} {'-'*11} {'-'*11} {'-'*12} {'-'*8} {'-'*8}")
    for h, r in sorted(results.items()):
        sig = ("***" if r["pval_diff"] < 0.01 else "**" if r["pval_diff"] < 0.05
               else "*" if r["pval_diff"] < 0.10 else "")
        print(f"  {h} min{'':<4} "
              f"{r['beta_pos']:>11.5f} {r['beta_neg']:>11.5f} "
              f"{r['tstat_diff']:>12.3f} {r['pval_diff']:>8.4f}{sig:>3} "
              f"{r['r2']:>8.5f}")
    print()


# ── Synthetic test data ───────────────────────────────────────────────────────

def _make_test_aligned() -> pd.DataFrame:
    np.random.seed(42)
    n        = 80
    ofi_z    = np.random.randn(n)
    lm_score = np.random.choice([-1, 0, 0, 1], size=n).astype(float)
    ret_1m   = (0.001 * ofi_z * lm_score.clip(min=0)
                - 0.0005 * ofi_z * lm_score.clip(max=0)
                + np.random.randn(n) * 0.001)
    return pd.DataFrame({
        "timestamp":  pd.date_range("2020-03-09 09:31", periods=n, freq="5min"),
        "lm_score":   lm_score,
        "llm_score":  np.clip(lm_score + np.random.normal(0, 0.1, n), -1, 1),
        "ofi_z":      ofi_z,
        "ret_1m":     ret_1m,
        "ret_5m":     ret_1m * 2.5 + np.random.randn(n) * 0.002,
        "ret_15m":    ret_1m * 5.0 + np.random.randn(n) * 0.004,
    })


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="IC & regression — LNA-Agent")
    parser.add_argument("--ticker", type=str, default="AAPL")
    parser.add_argument("--suffix", type=str, default="")
    parser.add_argument("--test",   action="store_true",
                        help="Run on synthetic data (no files needed)")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))

    if args.test:
        print("\n[TEST MODE] Using synthetic aligned data")
        aligned = _make_test_aligned()
        suffix  = "test"
        ticker  = args.ticker
    else:
        from align import load_aligned
        aligned = load_aligned(args.ticker, args.suffix)
        suffix  = args.suffix
        ticker  = args.ticker

    print(f"\n  Aligned rows : {len(aligned)}")

    # ── IC table ─────────────────────────────────────────────────────────────
    ic_table = compute_ic_table(aligned)
    _print_ic(ic_table)

    ic_path = RESULTS / f"ic_table_{ticker}_{suffix}.csv"
    ic_table.to_csv(ic_path, index=False)
    print(f"  IC table → {ic_path}")

    # ── Regression ───────────────────────────────────────────────────────────
    for col in ["lm_score", "llm_score"]:
        if col in aligned.columns and aligned[col].notna().any():
            reg = run_regression(aligned, sentiment_col=col)
            if reg:
                _print_regression(reg, col)
                reg_path = RESULTS / f"regression_{ticker}_{suffix}_{col}.csv"
                pd.DataFrame(reg.values()).to_csv(reg_path, index=False)
                print(f"  Regression → {reg_path}")

    # ── Plot ─────────────────────────────────────────────────────────────────
    if not ic_table.empty:
        plot_ic_comparison(ic_table, ticker, suffix)
