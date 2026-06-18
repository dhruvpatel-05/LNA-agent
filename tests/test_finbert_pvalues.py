"""
test_finbert_pvalues.py — FinBERT IC significance: p-values and bootstrap CIs.

Same clean, point-in-time evaluation window and confirmed/conflicted OFI-sign
split as test_finbert_conditional_eval.py (2019-01-01 .. 2020-12-31, a 6+
month buffer past FinBERT's ~mid-2018 training cutoff).

For each (horizon, split) pair, computes:
  - Spearman IC and two-sided p-value (scipy.stats.spearmanr)
  - 95% bootstrap CI on the IC (n_boot=1000, seed=42, resampling events
    with replacement and recomputing Spearman IC each draw)

Read-only on all existing data files. No plots, no model training.

Usage:
  python tests/test_finbert_pvalues.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentiment_finbert import score_headlines

ALIGNED_CSV = (Path(__file__).resolve().parent.parent / "results" / "align"
               / "aligned_AAPL_2016-01-01_2020-12-31.csv")
CLEAN_START = "2019-01-01"
CLEAN_END   = "2020-12-31"
HORIZONS    = [1, 5, 15]
MIN_EVENTS  = 100
N_BOOT      = 1000
SEED        = 42


def _ic(signal: pd.Series, returns: pd.Series) -> tuple[float, float, int]:
    mask = signal.notna() & returns.notna()
    n = int(mask.sum())
    if n < 10:
        return float("nan"), float("nan"), n
    ic, pval = spearmanr(signal[mask], returns[mask])
    return float(ic), float(pval), n


def _bootstrap_ci(signal: pd.Series, returns: pd.Series, rng: np.random.Generator,
                  n_boot: int = N_BOOT) -> tuple[float, float]:
    mask = signal.notna() & returns.notna()
    sig  = signal[mask].to_numpy()
    ret  = returns[mask].to_numpy()
    n    = len(sig)
    if n < 10:
        return float("nan"), float("nan")

    boot_ics = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        ic, _ = spearmanr(sig[idx], ret[idx])
        if not np.isnan(ic):
            boot_ics.append(ic)

    if not boot_ics:
        return float("nan"), float("nan")
    lo, hi = np.percentile(boot_ics, [2.5, 97.5])
    return float(lo), float(hi)


def test_finbert_pvalues():
    df = pd.read_csv(ALIGNED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Clean, point-in-time evaluation window ───────────────────────────────
    clean = df[(df["timestamp"] >= CLEAN_START) & (df["timestamp"] <= CLEAN_END)].copy()
    n_total = len(clean)

    print(f"\n{'═' * 80}")
    print(f"  FinBERT IC significance — p-values and bootstrap CIs")
    print(f"  Clean window: {CLEAN_START} .. {CLEAN_END}")
    print(f"{'═' * 80}")
    print(f"  Overall n before split: {n_total}")

    assert n_total > MIN_EVENTS, (
        f"Clean window filter left only {n_total} events (need > {MIN_EVENTS})"
    )

    # ── Score headlines with FinBERT ──────────────────────────────────────────
    scored = score_headlines(clean["headline"].tolist(), batch_size=32)
    clean["finbert_score"] = scored["finbert_score"].values

    # ── Confirmed / conflicted split by sign agreement with OFI ──────────────
    finbert_sign = np.sign(clean["finbert_score"])
    ofi_sign     = np.sign(clean["ofi_z"])
    agree        = finbert_sign == ofi_sign

    splits = {
        "confirmed":  clean[agree].copy(),
        "conflicted": clean[~agree].copy(),
    }

    print(f"  confirmed  (sign agreement):    n={len(splits['confirmed'])}")
    print(f"  conflicted (sign disagreement): n={len(splits['conflicted'])}")

    # ── IC, p-value, bootstrap CI per (horizon, split) ───────────────────────
    rng  = np.random.default_rng(SEED)
    rows = []
    for h in HORIZONS:
        ret_col = f"ret_{h}m"
        for split_name, sub in splits.items():
            ic, pval, n = _ic(sub["finbert_score"], sub[ret_col])
            ci_lo, ci_hi = _bootstrap_ci(sub["finbert_score"], sub[ret_col], rng)
            rows.append({
                "horizon":  f"{h}min",
                "split":    split_name,
                "ic":       ic,
                "p_value":  pval,
                "ci_lower": ci_lo,
                "ci_upper": ci_hi,
                "n":        n,
            })

    table = pd.DataFrame(rows)

    # ── Print clearly labeled table with significance flags ──────────────────
    print(f"\n  {'horizon':<8} {'split':<11} {'ic':>9} {'p_value':>9} "
          f"{'ci_lower':>10} {'ci_upper':>10} {'n':>6}  sig")
    print(f"  {'-'*8} {'-'*11} {'-'*9} {'-'*9} {'-'*10} {'-'*10} {'-'*6}  ---")
    for _, r in table.iterrows():
        flag = "*" if pd.notna(r["p_value"]) and r["p_value"] < 0.05 else ""
        print(f"  {r['horizon']:<8} {r['split']:<11} {r['ic']:>+9.4f} "
              f"{r['p_value']:>9.4f} {r['ci_lower']:>+10.4f} {r['ci_upper']:>+10.4f} "
              f"{r['n']:>6}  {flag}")
    print()

    # ── Sanity assertions ──────────────────────────────────────────────────────
    assert n_total > MIN_EVENTS

    ic_values = table["ic"].dropna()
    assert ic_values.between(-1.0, 1.0).all(), "IC values out of [-1, 1] range"

    pvals = table["p_value"].dropna()
    assert pvals.between(0.0, 1.0).all(), "p-values out of [0, 1] range"

    return table


if __name__ == "__main__":
    test_finbert_pvalues()
