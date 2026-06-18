"""
placebo.py — Anonymization and shuffled-headline placebos.
LNA-Agent: Lobster News Alpha

Tests whether the LLM signal is anchored to entity names vs. underlying
news content. Runs three conditions over ~300 events:
  named      — original headline
  anonymized — company/ticker replaced with neutral tokens
  shuffled   — wrong date + wrong company name variant

Calls stats_rigor.paired_ic_diff for each IC drop with bootstrap CI.
"""

import sys
import random
import numpy as np
import pandas as pd
from pathlib import Path

# Pull the existing LLM scorer; do not reimplement the API call
_LEGACY = Path(__file__).parent.parent / "legacy" / "src"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from sentiment import score_llm          # type: ignore  — legacy scorer
from sentiment_anonymizer import anonymize_headline  # type: ignore

from stats_rigor import paired_ic_diff, spearman_ic

# ── Anonymization ──────────────────────────────────────────────────────────────

# Additional replacement map for shuffle — wrong-company substitutions
_WRONG_COMPANY: dict[str, str] = {
    "AAPL": "Caterpillar",
    "AMD":  "Walgreens",
    "QCOM": "Delta Air Lines",
    "NFLX": "ExxonMobil",
    "JPM":  "Kroger",
    "NVDA": "Ford Motor",
    "META": "General Mills",
    "TSLA": "Nordstrom",
}

_WRONG_DATE_PHRASES = [
    "three years ago", "in 2009", "during Q1 2015", "back in 2003",
]


def anonymize(headline: str, ticker: str = "") -> str:
    """
    Replace company names and ticker symbols with neutral tokens.

    Delegates to the existing sentiment_anonymizer logic so the substitution
    map stays in one place.
    """
    tickers = [ticker] if ticker else list(_WRONG_COMPANY.keys())
    return anonymize_headline(headline, tickers)


def shuffle_headline(
    headline: str,
    ticker: str,
    seed: int | None = None,
) -> str:
    """
    Produce a semantically misleading variant of the headline.

    Substitutes the real company name with a wrong-sector company and
    prepends a false date reference.  The text structure is preserved so
    the LLM sees a plausible-looking headline.
    """
    rng = random.Random(seed)
    wrong = _WRONG_COMPANY.get(ticker, "Acme Corp")
    date_phrase = rng.choice(_WRONG_DATE_PHRASES)

    # Anonymize first, then replace COMPANY/TICKER tokens with the wrong name
    anon = anonymize(headline, ticker)
    anon = anon.replace("COMPANY", wrong).replace("TICKER", wrong)

    return f"{date_phrase.capitalize()}, {anon}"


# ── Batch scoring ──────────────────────────────────────────────────────────────

def score_three_conditions(
    panel: pd.DataFrame,
    ticker: str,
    n_events: int = 300,
    seed: int = 42,
) -> pd.DataFrame:
    """
    Score up to n_events rows under named / anonymized / shuffled conditions.

    Returns DataFrame with columns:
      timestamp, headline, named_score, anon_score, shuffled_score,
      ofi_z, ret_1m, ret_5m, ret_15m
    """
    df = panel.dropna(subset=["headline", "ofi_z", "ret_1m"]).copy()
    if len(df) > n_events:
        df = df.sample(n=n_events, random_state=seed).reset_index(drop=True)

    named_scores    = []
    anon_scores     = []
    shuffled_scores = []

    total = len(df)
    for i, row in df.iterrows():
        hl = str(row["headline"])
        named_scores.append(score_llm(hl))
        anon_scores.append(score_llm(anonymize(hl, ticker)))
        shuffled_scores.append(score_llm(shuffle_headline(hl, ticker, seed=i)))
        if (len(named_scores)) % 50 == 0:
            print(f"  placebo: {len(named_scores)}/{total}")

    df["named_score"]    = named_scores
    df["anon_score"]     = anon_scores
    df["shuffled_score"] = shuffled_scores

    keep = [c for c in ["timestamp", "headline", "named_score", "anon_score",
                         "shuffled_score", "ofi_z", "ret_1m", "ret_5m", "ret_15m"]
            if c in df.columns]
    return df[keep].reset_index(drop=True)


def run_placebo_analysis(
    panel: pd.DataFrame,
    ticker: str,
    horizon: int = 1,
    n_events: int = 300,
    n_boot: int = 5000,
    seed: int = 42,
) -> dict:
    """
    Full placebo pipeline: score three conditions, compute paired IC drops.

    Returns dict with keys:
      named_vs_anon    : paired_ic_diff result
      named_vs_shuffled: paired_ic_diff result
      scored_df        : the scored DataFrame
    """
    scored = score_three_conditions(panel, ticker, n_events=n_events, seed=seed)

    ret = scored[f"ret_{horizon}m"]
    named_vs_anon     = paired_ic_diff(scored["named_score"],
                                       scored["anon_score"], ret,
                                       n_boot=n_boot, seed=seed)
    named_vs_shuffled = paired_ic_diff(scored["named_score"],
                                       scored["shuffled_score"], ret,
                                       n_boot=n_boot, seed=seed)

    print(f"\n  Placebo results ({ticker}, ret_{horizon}m, n={named_vs_anon['n']}):")
    print(f"  named IC    : {named_vs_anon['ic_a']:+.4f}")
    print(f"  anon IC     : {named_vs_anon['ic_b']:+.4f}  "
          f"drop={named_vs_anon['diff']:+.4f}  "
          f"95%CI=[{named_vs_anon['ci_lo']:+.4f}, {named_vs_anon['ci_hi']:+.4f}]  "
          f"p_boot={named_vs_anon['p_boot']:.3f}")
    print(f"  shuffled IC : {named_vs_shuffled['ic_b']:+.4f}  "
          f"drop={named_vs_shuffled['diff']:+.4f}  "
          f"95%CI=[{named_vs_shuffled['ci_lo']:+.4f}, {named_vs_shuffled['ci_hi']:+.4f}]  "
          f"p_boot={named_vs_shuffled['p_boot']:.3f}")

    return {
        "named_vs_anon":     named_vs_anon,
        "named_vs_shuffled": named_vs_shuffled,
        "scored_df":         scored,
    }
