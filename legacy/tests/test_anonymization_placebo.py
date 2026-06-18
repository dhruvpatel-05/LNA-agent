"""
test_anonymization_placebo.py — Named vs anonymized agent IC comparison
(live API).

Loads the same clean, point-in-time evaluation window as the FinBERT tests
(2019-01-01 .. 2020-12-31), draws a fixed sample of 10 events per ticker
(50 total, seed=42) across all 5 pipeline tickers, anonymizes headlines with
src/sentiment_anonymizer.py, and runs EVERY event through src/decisions.py's
decide_llm() TWICE — once with the original headline, once with the
anonymized version — using the live Anthropic API (100 calls total).

Compares Spearman IC of agent_signal vs ret_1m for the named run vs the
anonymized run. A large ic_drop suggests the agent leans on entity
recognition/anchoring rather than headline content.

Usage:
  python tests/test_anonymization_placebo.py
"""
from dotenv import load_dotenv
load_dotenv()

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import decisions as _decisions
from decisions import decide_llm, decision_to_signal
from sentiment_anonymizer import anonymize_batch

ALIGN_DIR   = Path(__file__).resolve().parent.parent / "results" / "align"
SUFFIX      = "2016-01-01_2020-12-31"
CLEAN_START = "2019-01-01"
CLEAN_END   = "2020-12-31"
TICKER_LIST = ["AAPL", "AMD", "QCOM", "NFLX", "JPM"]
N_PER_TICKER = 10
SEED         = 42


def _ic(signal: pd.Series, returns: pd.Series) -> tuple[float, int]:
    mask = signal.notna() & returns.notna()
    n = int(mask.sum())
    if n < 10:
        return float("nan"), n
    ic, _ = spearmanr(signal[mask], returns[mask])
    return float(ic), n


def _load_sample() -> pd.DataFrame:
    """Draw 10 events per ticker (seed=42) from the clean window, concatenated."""
    samples = []
    for ticker in TICKER_LIST:
        path = ALIGN_DIR / f"aligned_{ticker}_{SUFFIX}.csv"
        df = pd.read_csv(path)
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        clean = df[(df["timestamp"] >= CLEAN_START) & (df["timestamp"] <= CLEAN_END)].copy()
        sub = clean.sample(n=N_PER_TICKER, random_state=SEED).copy()
        sub["ticker"] = ticker
        samples.append(sub)
    return pd.concat(samples, ignore_index=True)


def test_anonymization_placebo():
    sample = _load_sample()
    n_events = len(sample)

    print(f"\n{'═' * 80}")
    print(f"  Named vs anonymized agent IC — live API")
    print(f"  Clean window: {CLEAN_START} .. {CLEAN_END}")
    print(f"  Sample: {N_PER_TICKER} events/ticker × {len(TICKER_LIST)} tickers "
          f"= {n_events} events (seed={SEED})")
    print(f"{'═' * 80}")

    originals  = sample["headline"].tolist()
    anonymized = anonymize_batch(originals, TICKER_LIST)

    # ── Run every event through decide_llm twice: original, then anonymized ──
    named_scores = []
    anon_scores  = []
    for i, row in sample.iterrows():
        ticker    = row["ticker"]
        ofi_z     = float(row.get("ofi_z",     0.0))
        lm_score  = float(row.get("lm_score",  0.0))
        llm_score = float(row.get("llm_score", 0.0) or 0.0)

        d_named = decide_llm(ticker, originals[i],  ofi_z, lm_score, llm_score)
        time.sleep(_decisions.API_DELAY)
        d_anon  = decide_llm(ticker, anonymized[i], ofi_z, lm_score, llm_score)
        time.sleep(_decisions.API_DELAY)

        named_scores.append(decision_to_signal(d_named))
        anon_scores.append(decision_to_signal(d_anon))

        if (i + 1) % 10 == 0:
            print(f"  ... scored {i + 1}/{n_events} events")

    sample["named_score"] = named_scores
    sample["anon_score"]  = anon_scores

    # ── IC table: named vs anonymized vs ret_1m ──────────────────────────────
    named_ic, n_named = _ic(sample["named_score"], sample["ret_1m"])
    anon_ic,  n_anon  = _ic(sample["anon_score"],  sample["ret_1m"])
    ic_drop = named_ic - anon_ic

    print(f"\n  IC table (Spearman vs ret_1m)")
    print(f"  {'-' * 32}")
    print(f"  {'version':<11} | {'ic':>6} | {'n':>5}")
    print(f"  {'-'*11} | {'-'*6} | {'-'*5}")
    print(f"  {'named':<11} | {named_ic:>+6.3f} | {n_named:>5}")
    print(f"  {'anonymized':<11} | {anon_ic:>+6.3f} | {n_anon:>5}")
    print(f"  {'ic_drop':<11} | {ic_drop:>+6.3f} | {'':>5}")

    # ── Full row-by-row inspection table ─────────────────────────────────────
    print(f"\n  Per-event inspection ({n_events} rows)")
    print(f"  {'-' * 100}")
    for i, row in sample.iterrows():
        print(f"\n  [{i:>2}] ticker={row['ticker']}  "
              f"named={row['named_score']:+.2f}  anon={row['anon_score']:+.2f}  "
              f"fwd_ret_1m={row['ret_1m']:+.5f}")
        print(f"        original  : {originals[i]}")
        print(f"        anonymized: {anonymized[i]}")

    print(f"\n  Total estimated API cost: ${_decisions.get_total_cost():.4f}")
    print()

    # ── Sanity assertions ──────────────────────────────────────────────────────
    assert n_events == N_PER_TICKER * len(TICKER_LIST)
    assert len(sample["named_score"]) == len(sample["anon_score"]) == n_events

    return sample


if __name__ == "__main__":
    test_anonymization_placebo()
