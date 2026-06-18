"""
test_finbert_conditional_eval.py — FinBERT IC conditioned on OFI agreement.

Same clean, point-in-time evaluation window as test_finbert_clean_eval.py
(2019-01-01 .. 2020-12-31, a 6+ month buffer past FinBERT's ~mid-2018
training cutoff) and the same FinBERT scores.

Splits events into:
  confirmed  — sign(finbert_score) == sign(ofi_z)
  conflicted — sign(finbert_score) != sign(ofi_z)

and computes Spearman IC of finbert_score vs forward returns at 1/5/15-minute
horizons within each split.

Read-only on all existing data files. No plots, no model training.

Usage:
  python tests/test_finbert_conditional_eval.py
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


def _ic(signal: pd.Series, returns: pd.Series) -> tuple[float, float]:
    mask = signal.notna() & returns.notna()
    if mask.sum() < 10:
        return float("nan"), float("nan")
    ic, pval = spearmanr(signal[mask], returns[mask])
    return float(ic), float(pval)


def test_finbert_conditional_eval():
    df = pd.read_csv(ALIGNED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Clean, point-in-time evaluation window ───────────────────────────────
    clean = df[(df["timestamp"] >= CLEAN_START) & (df["timestamp"] <= CLEAN_END)].copy()
    n_total = len(clean)

    print(f"\n{'═' * 70}")
    print(f"  FinBERT conditional evaluation — OFI agreement split")
    print(f"  Clean window: {CLEAN_START} .. {CLEAN_END}")
    print(f"{'═' * 70}")
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

    agree     = finbert_sign == ofi_sign
    confirmed = clean[agree].copy()
    conflicted = clean[~agree].copy()

    print(f"  confirmed (sign agreement):  n={len(confirmed)}")
    print(f"  conflicted (sign disagreement): n={len(conflicted)}")

    # ── IC per split, per horizon ─────────────────────────────────────────────
    rows = []
    for h in HORIZONS:
        ret_col = f"ret_{h}m"
        conf_ic,  _ = _ic(confirmed["finbert_score"],  confirmed[ret_col])
        confl_ic, _ = _ic(conflicted["finbert_score"], conflicted[ret_col])
        n_confirmed  = int((confirmed["finbert_score"].notna()  & confirmed[ret_col].notna()).sum())
        n_conflicted = int((conflicted["finbert_score"].notna() & conflicted[ret_col].notna()).sum())
        rows.append({
            "horizon":       f"{h}min",
            "confirmed_ic":  conf_ic,
            "conflicted_ic": confl_ic,
            "n_confirmed":   n_confirmed,
            "n_conflicted":  n_conflicted,
        })

    table = pd.DataFrame(rows)

    print(f"\n  IC comparison — confirmed vs conflicted (FinBERT × OFI sign agreement)")
    print(f"  {'-' * 60}")
    print(table.to_string(index=False))
    print()

    # ── Sanity assertions ──────────────────────────────────────────────────────
    assert n_total > MIN_EVENTS

    ic_values = pd.concat([table["confirmed_ic"], table["conflicted_ic"]]).dropna()
    assert ic_values.between(-1.0, 1.0).all(), "IC values out of [-1, 1] range"

    return table


if __name__ == "__main__":
    test_finbert_conditional_eval()
