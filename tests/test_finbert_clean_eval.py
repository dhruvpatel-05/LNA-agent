"""
test_finbert_clean_eval.py — FinBERT IC on a point-in-time clean evaluation window.

ProsusAI/finbert's training cutoff is approximately mid-2018, so any IC
evaluation on headlines from before that date risks lookahead bias (the
model may have seen those headlines, or headlines describing those events,
during pretraining/finetuning). To avoid this, we restrict evaluation to
2019-01-01 .. 2020-12-31 — a 6+ month buffer after the cutoff.

Loads the existing aligned event panel (headline + lm_score + forward
returns), filters to the clean window, scores headlines with FinBERT, and
compares Spearman IC of finbert_score vs lm_score (read from data, not
recomputed) against forward returns at 1/5/15-minute horizons.

Read-only on all existing data files. No plots, no model training.

Usage:
  python tests/test_finbert_clean_eval.py
"""

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentiment_finbert import score_headlines

ALIGNED_CSV   = (Path(__file__).resolve().parent.parent / "results" / "align"
                 / "aligned_AAPL_2016-01-01_2020-12-31.csv")
CLEAN_START   = "2019-01-01"
CLEAN_END     = "2020-12-31"
HORIZONS      = [1, 5, 15]
MIN_EVENTS    = 100


def _ic(signal: pd.Series, returns: pd.Series) -> tuple[float, float]:
    mask = signal.notna() & returns.notna()
    if mask.sum() < 10:
        return float("nan"), float("nan")
    ic, pval = spearmanr(signal[mask], returns[mask])
    return float(ic), float(pval)


def test_finbert_clean_eval():
    df = pd.read_csv(ALIGNED_CSV)
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # ── Step 1: filter to clean, point-in-time evaluation window ─────────────
    clean = df[(df["timestamp"] >= CLEAN_START) & (df["timestamp"] <= CLEAN_END)].copy()
    n_events = len(clean)

    print(f"\n{'═' * 70}")
    print(f"  FinBERT clean evaluation window: {CLEAN_START} .. {CLEAN_END}")
    print(f"  (FinBERT training cutoff ≈ mid-2018; 6+ month buffer applied)")
    print(f"{'═' * 70}")
    print(f"  Events in clean window: {n_events}  (of {len(df)} total)")

    assert n_events > MIN_EVENTS, (
        f"Clean window filter left only {n_events} events (need > {MIN_EVENTS})"
    )

    # ── Step 2: score headlines with FinBERT ─────────────────────────────────
    scored = score_headlines(clean["headline"].tolist(), batch_size=32)
    clean["finbert_score"] = scored["finbert_score"].values

    # ── Step 3 & 4: Spearman IC of finbert_score and lm_score vs returns ─────
    rows = []
    for h in HORIZONS:
        ret_col = f"ret_{h}m"
        finbert_ic, finbert_pval = _ic(clean["finbert_score"], clean[ret_col])
        lm_ic,      lm_pval      = _ic(clean["lm_score"],      clean[ret_col])
        n = int((clean["finbert_score"].notna() & clean[ret_col].notna()).sum())
        rows.append({
            "horizon":    f"{h}min",
            "finbert_ic": finbert_ic,
            "lm_ic":      lm_ic,
            "n_events":   n,
        })

    table = pd.DataFrame(rows)

    # ── Step 5: print clearly labeled IC table ───────────────────────────────
    print(f"\n  IC comparison — FinBERT vs LM (clean window, n={n_events} events)")
    print(f"  {'-' * 50}")
    print(table.to_string(index=False))
    print()

    # ── Step 6 & 7: sanity assertions ────────────────────────────────────────
    assert n_events > MIN_EVENTS

    ic_values = pd.concat([table["finbert_ic"], table["lm_ic"]]).dropna()
    assert ic_values.between(-1.0, 1.0).all(), "IC values out of [-1, 1] range"

    return table


if __name__ == "__main__":
    test_finbert_clean_eval()
