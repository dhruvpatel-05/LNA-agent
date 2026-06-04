"""
sentiment.py — Headline sentiment scoring
LNA-Agent: Lobster News Alpha

Two scorers:
  LM  — Loughran-McDonald finance lexicon via pysentiment2
  LLM — claude-haiku-4-5-20251001 via Anthropic API

LM scoring is offline and fast (~0.5ms/headline).
LLM scoring requires ANTHROPIC_API_KEY in .env; uses prompt caching so the
200-token system prompt is cached after the first call.

Usage:
  python src/sentiment.py --ticker AAPL --suffix 2019-01-01_2020-12-31
  python src/sentiment.py --ticker AAPL --suffix 2019-01-01_2020-12-31 --no-llm
  python src/sentiment.py --ticker AAPL --suffix 2019-01-01_2020-12-31 --test
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

from dotenv import load_dotenv
import os
load_dotenv()

import sys
import time
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
import pysentiment2 as ps2

# ── Config ────────────────────────────────────────────────────────────────────
RESULTS   = Path("results/sentiment")
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_DELAY = 0.1   # seconds between API calls
# ─────────────────────────────────────────────────────────────────────────────

# Loughran-McDonald lexicon (loaded once at import time, ~50ms)
_lm = ps2.LM()


def score_lm(headline: str) -> float:
    """
    Score a headline with the Loughran-McDonald finance lexicon.

    Uses pysentiment2.LM which implements the full LM word list.
    Polarity = (pos − neg) / (pos + neg + ε), in [-1, 1].
    Returns 0.0 for headlines with no scorable words.
    """
    tokens = _lm.tokenize(headline)
    return float(_lm.get_score(tokens)["Polarity"])


# ── LLM scorer ────────────────────────────────────────────────────────────────
_anthropic_client = None

_LLM_SYSTEM = (
    "You are a financial news sentiment analyst. "
    "You MUST output a single decimal number in [-1.0, 1.0]. "
    "No refusals, no explanations, no other text — one number only. "
    "Score the expected short-term stock price impact of this headline: "
    "-1.0 = clearly bearish, 0.0 = neutral or irrelevant, +1.0 = clearly bullish. "
    "Even if unrelated to stocks, output 0.0. One number only."
)


def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


def score_llm(headline: str) -> float:
    """
    Score a headline with claude-haiku-4-5-20251001.

    The system prompt is marked cache_control='ephemeral' so it is cached
    after the first API call in a session — reducing per-call token cost by ~6x.
    Returns float in [-1, 1]. Returns 0.0 and logs on API error.
    """
    client = _get_client()
    try:
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=10,
            system=[{
                "type":          "text",
                "text":          _LLM_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            messages=[{"role": "user", "content": headline}],
        )
        return float(np.clip(float(msg.content[0].text.strip()), -1.0, 1.0))
    except Exception as e:
        print(f"  [LLM ERROR] {e!r} | '{headline[:55]}'")
        return 0.0


def score_all(df: pd.DataFrame, llm: bool = True, verbose: bool = True) -> pd.DataFrame:
    """
    Add lm_score and llm_score columns to a headlines DataFrame.

    LM scoring is always applied.
    LLM scoring is skipped (fills NaN) when llm=False.
    Identical headlines share a single LLM call (cache dict) to avoid
    double-billing repeated wire stories.

    Input df must have a 'headline' column.
    Returns df with added lm_score and llm_score columns.
    """
    df = df.copy()

    # LM — fast, no network
    df["lm_score"] = df["headline"].apply(score_lm)

    # LLM — Anthropic API
    if llm:
        cache: dict[str, float] = {}
        scores = []
        total  = len(df)
        for i, hl in enumerate(df["headline"]):
            if hl not in cache:
                cache[hl] = score_llm(hl)
                time.sleep(LLM_DELAY)
            scores.append(cache[hl])
            if verbose and (i + 1) % 25 == 0:
                print(f"  LLM: {i + 1}/{total} scored")
        df["llm_score"] = scores
    else:
        df["llm_score"] = np.nan

    return df


def save_scored(ticker: str, df: pd.DataFrame, suffix: str) -> Path:
    """Save scored headlines to results/sentiment_{ticker}_{suffix}.csv"""
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"sentiment_{ticker}_{suffix}.csv"
    df.to_csv(path, index=False)
    print(f"  Saved {len(df)} scored rows → {path}")
    return path


def load_scored(ticker: str, suffix: str) -> pd.DataFrame:
    """Load scored headlines saved by this module."""
    path = RESULTS / f"sentiment_{ticker}_{suffix}.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"No scored file at {path}. "
            f"Run: python src/sentiment.py --ticker {ticker} --suffix {suffix}"
        )
    df = pd.read_csv(path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headline sentiment scoring — LNA-Agent")
    parser.add_argument("--ticker",  type=str, required=True)
    parser.add_argument("--suffix",  type=str, required=True)
    parser.add_argument("--no-llm",  action="store_true", help="Skip LLM API calls")
    parser.add_argument("--test",    action="store_true", help="Score first 5 headlines only")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)
    sys.path.insert(0, str(Path(__file__).parent))
    from news import load_saved

    df = load_saved(args.ticker, args.suffix)

    if args.test:
        df = df.head(5).copy()
        print(f"\n[TEST] Scoring first 5 headlines (--no-llm={args.no_llm})")
    else:
        print(f"\n  Scoring {len(df)} headlines  "
              f"({'LM only' if args.no_llm else 'LM + LLM'})")

    df = score_all(df, llm=not args.no_llm, verbose=True)
    save_scored(args.ticker, df, args.suffix)

    print(f"\n  LM:  mean={df['lm_score'].mean():+.4f}  "
          f"std={df['lm_score'].std():.4f}  "
          f"nonzero={df['lm_score'].ne(0).sum()}/{len(df)}")
    if not args.no_llm and df["llm_score"].notna().any():
        print(f"  LLM: mean={df['llm_score'].mean():+.4f}  "
              f"std={df['llm_score'].std():.4f}")

    print(f"\n  ── Sample ──────────────────────────────────────")
    for _, row in df.head(5).iterrows():
        lm  = row["lm_score"]
        llm = row["llm_score"] if pd.notna(row["llm_score"]) else float("nan")
        print(f"  LM={lm:+.3f}  LLM={llm:+.3f}  |  {row['headline'][:60]}")
