"""
sentiment.py — Multi-source headline sentiment scoring
LNA-Agent: Lobster News Alpha

Three sentiment sources are computed for each headline:
  1. Loughran-McDonald (LM) — rule-based finance lexicon, no external API
  2. LLM (claude-haiku-4-5-20251001) — context-aware scoring via Anthropic API
  3. RavenPack — professional NLP score loaded from vendor CSV (passthrough)

Research hypothesis: LLM scores capture contextual nuance that LM misses
(e.g., "Apple *misses* expectations" vs "Apple misses *nothing*"), producing
a stronger signal when interacted with OFI in signal.py.

Usage:
  # Score headlines for a ticker (requires sentiment_{ticker}_{suffix}.csv or RavenPack CSV)
  python src/sentiment.py --ticker AAPL --suffix 2019-01-01_2020-12-31

  # Dry-run without LLM API calls
  python src/sentiment.py --ticker AAPL --suffix 2019-01-01_2020-12-31 --no-llm

  # Smoke-test on synthetic data before real data arrives
  python src/sentiment.py --test
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
import os
load_dotenv()

import re
import sys
import time
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

# ── Config ─────────────────────────────────────────────────────────────────────
NEWS_DIR  = Path("data/news/raw")
RESULTS   = Path("results")
LLM_MODEL = "claude-haiku-4-5-20251001"
LLM_DELAY = 0.1        # seconds between API calls to respect rate limits
# ──────────────────────────────────────────────────────────────────────────────


# ── Loughran-McDonald Finance Lexicon ─────────────────────────────────────────
# Representative subset of the LM Master Dictionary (Loughran & McDonald 2011,
# J. Finance). Full dictionary: https://sraf.nd.edu/loughranmcdonald/resources/
# Matching is whole-word, case-insensitive. Words stored in uppercase.

_LM_NEGATIVE = {
    # Solvency / credit
    "BANKRUPT", "BANKRUPTCY", "DEFAULT", "DEFAULTED", "DEFAULTING",
    "INSOLVENT", "INSOLVENCY", "LIQUIDATE", "LIQUIDATION", "LIQUIDATING",
    "RESTRUCTURE", "RESTRUCTURING", "RECEIVERSHIP",
    # Financial distress
    "IMPAIR", "IMPAIRED", "IMPAIRMENT", "WRITEDOWN", "WRITEOFF",
    "WRITEDOWNS", "WRITEOFFS", "RESTATE", "RESTATED", "RESTATEMENT",
    "DEFICIT", "DEFICITS", "DELINQUENT", "DELINQUENCY",
    # Performance / outlook
    "DECLINE", "DECLINED", "DECLINING", "MISS", "MISSED", "MISSING",
    "DISAPPOINT", "DISAPPOINTING", "DISAPPOINTED", "DISAPPOINTMENT",
    "SHORTFALL", "SHORTFALLS", "DOWNGRADE", "DOWNGRADED",
    "LOSS", "LOSSES", "LOSING", "UNPROFITABLE", "UNFAVORABLE",
    # Legal / regulatory
    "LAWSUIT", "LAWSUITS", "LITIGATION", "LITIGATIONS", "FRAUD",
    "FRAUDULENT", "INVESTIGATION", "INVESTIGATE", "INVESTIGATED",
    "PROBE", "PENALTY", "PENALTIES", "SANCTION", "SANCTIONS",
    "VIOLATION", "VIOLATIONS", "IRREGULARITY", "IRREGULARITIES",
    "ALLEGATION", "ALLEGATIONS", "ALLEGE", "ALLEGED", "ACCUSE",
    "ACCUSATION", "NONCOMPLIANCE",
    # Risk / uncertainty
    "RISK", "RISKY", "UNCERTAIN", "UNCERTAINTY", "VOLATILE", "VOLATILITY",
    "ADVERSE", "ADVERSELY", "ADVERSITY", "DETERIORATE", "DETERIORATING",
    "DETERIORATION", "CONCERN", "CONCERNS", "CHALLENGING", "CHALLENGE",
    "DIFFICULT", "DIFFICULTY", "DIFFICULTIES",
    # Damage / harm
    "DAMAGE", "DAMAGED", "DAMAGES", "HARM", "HARMFUL", "HARMED",
    "HAZARD", "HAZARDOUS", "BURDEN", "BURDENSOME", "CEASE", "CLOSURE",
    "CLOSED", "SUSPEND", "SUSPENDED", "SUSPENSION", "TERMINATE",
    "TERMINATED", "TERMINATION", "ABANDON", "ABANDONED", "ABANDONMENT",
    # Macro / sector
    "RECESSION", "RECESSIONS", "DOWNTURN", "CRISIS", "CRISES",
    "DISTRESS", "WEAK", "WEAKENED", "WEAKNESS", "POOR", "NEGATIVE",
    "OVERSTATE", "OVERSTATED", "MISLEAD", "MISLEADING",
}

_LM_POSITIVE = {
    # Performance
    "ACHIEVE", "ACHIEVED", "ACHIEVEMENT", "EXCEED", "EXCEEDED", "EXCEEDS",
    "BEAT", "BEATS", "SURPASS", "SURPASSED", "OUTPERFORM", "OUTPERFORMED",
    "OUTPERFORMANCE", "RECORD", "RECORDS",
    # Growth / improvement
    "GROWTH", "GROW", "GROWING", "GREW", "GAIN", "GAINS", "INCREASE",
    "INCREASED", "INCREASES", "IMPROVING", "IMPROVED", "IMPROVEMENT",
    "EXPAND", "EXPANDED", "EXPANSION", "RISE", "RISING", "RECOVER",
    "RECOVERED", "RECOVERY",
    # Positive sentiment
    "STRONG", "STRENGTH", "STRONGEST", "SUCCEED", "SUCCEEDED", "SUCCESS",
    "SUCCESSFUL", "SUCCESSFULLY", "BEST", "BETTER", "SUPERIOR",
    "FAVORABLE", "POSITIVE", "CONFIDENT", "CONFIDENCE",
    "PROFITABLE", "PROFITABILITY", "BENEFIT", "BENEFICIAL",
    # Innovation / leadership
    "INNOVATIVE", "INNOVATION", "LEAD", "LEADER", "LEADING",
    "OPPORTUNITY", "OPPORTUNITIES", "ADVANTAGE", "ADVANTAGES",
    "BREAKTHROUGH", "BREAKTHROUGHS",
    # Capital returns
    "DIVIDEND", "DIVIDENDS", "BUYBACK", "REPURCHASE", "REPURCHASES",
    "UPGRADE", "UPGRADED",
}

# ──────────────────────────────────────────────────────────────────────────────


def load_ravenpack(ticker: str, suffix: str = "") -> pd.DataFrame:
    """
    Load RavenPack headlines CSV from data/news/raw/.

    Expected columns: timestamp, headline, ticker, sentiment_score, relevance_score
    sentiment_score in [-1, 1] (RavenPack's composite sentiment score).
    relevance_score in [0, 1] (how relevant the article is to this ticker).

    The RavenPack score serves as professional-grade NLP baseline for comparison
    against our LM and LLM scores in compute_ic_table().
    """
    fname = f"{ticker}_ravenpack{'_' + suffix if suffix else ''}.csv"
    path  = NEWS_DIR / fname
    if not path.exists():
        raise FileNotFoundError(f"RavenPack file not found: {path}")

    df = pd.read_csv(path)
    df.columns = df.columns.str.strip().str.lower()
    df["timestamp"] = pd.to_datetime(df["timestamp"])

    # Normalise vendor column name to our internal convention
    if "sentiment_score" in df.columns:
        df = df.rename(columns={"sentiment_score": "ravenpack_score"})

    required = {"timestamp", "headline", "ravenpack_score"}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(f"RavenPack CSV missing columns: {missing}")

    return df.sort_values("timestamp").reset_index(drop=True)


def score_lm(headline: str) -> float:
    """
    Score a single headline using the Loughran-McDonald finance lexicon.

    Returns float in [-1, 1]:
      score = (positive_word_count - negative_word_count) / total_word_count

    Returns 0.0 for empty headlines or headlines with no scorable words.
    The LM lexicon is specifically designed for financial text — avoids
    mis-scoring words like "liability" that are neutral in general English
    but strongly negative in finance.
    """
    words = re.findall(r"[A-Za-z]+", headline.upper())
    if not words:
        return 0.0
    pos   = sum(1 for w in words if w in _LM_POSITIVE)
    neg   = sum(1 for w in words if w in _LM_NEGATIVE)
    total = len(words)
    return (pos - neg) / total


# ── LLM Scoring ───────────────────────────────────────────────────────────────
_anthropic_client = None

_LLM_SYSTEM = (
    "You are a financial news sentiment analyst. "
    "You will receive a stock market news headline. "
    "Reply with a single decimal number between -1.0 and 1.0:\n"
    "  -1.0 = very bearish / strongly negative for the company\n"
    "   0.0 = neutral or ambiguous\n"
    "  +1.0 = very bullish / strongly positive for the company\n"
    "Consider only the directional implication for the company's stock. "
    "Output ONLY the number, no explanation, no units."
)


def _get_anthropic():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set. Add it to .env")
        _anthropic_client = anthropic.Anthropic(api_key=api_key)
    return _anthropic_client


def score_llm(headline: str) -> float:
    """
    Score a headline using claude-haiku-4-5-20251001 via the Anthropic API.

    The system prompt uses prompt caching (cache_control="ephemeral") so that
    the 200-token system prompt is cached after the first call — subsequent
    calls in a batch hit the cache rather than paying input tokens again.

    Returns float in [-1, 1], clipped to the valid range.
    Returns 0.0 on any API error (logged to stdout).
    """
    client = _get_anthropic()
    try:
        msg = client.messages.create(
            model=LLM_MODEL,
            max_tokens=10,
            system=[
                {
                    "type": "text",
                    "text": _LLM_SYSTEM,
                    "cache_control": {"type": "ephemeral"},
                }
            ],
            messages=[{"role": "user", "content": headline}],
        )
        raw   = msg.content[0].text.strip()
        score = float(raw)
        return float(np.clip(score, -1.0, 1.0))
    except Exception as e:
        print(f"  [LLM ERROR] {e!r} | '{headline[:60]}'")
        return 0.0


def score_all(
    df: pd.DataFrame,
    llm: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Score every headline in df, adding columns: lm_score, llm_score.

    If df already has a 'ravenpack_score' column it is preserved unchanged.
    If llm=False, llm_score is filled with NaN (useful for offline testing).

    LLM results are cached by headline text to avoid double-billing duplicate
    headlines — common in news feeds where the same wire story appears twice.

    Returns a new DataFrame with the original columns plus lm_score, llm_score,
    and ravenpack_score (NaN if not already present).
    """
    df = df.copy()

    # ── LM (fast, no network) ────────────────────────────────────────────────
    df["lm_score"] = df["headline"].apply(score_lm)

    # ── LLM (Anthropic API) ──────────────────────────────────────────────────
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
                print(f"  LLM: {i + 1}/{total} headlines scored")
        df["llm_score"] = scores
    else:
        df["llm_score"] = np.nan

    # ── RavenPack passthrough ────────────────────────────────────────────────
    if "ravenpack_score" not in df.columns:
        df["ravenpack_score"] = np.nan

    return df


# ── Synthetic data for --test mode ────────────────────────────────────────────

def _make_test_headlines() -> pd.DataFrame:
    """
    Generate 10 synthetic headlines with known sentiment polarity.
    Used for offline smoke-testing before real data arrives.
    """
    np.random.seed(42)
    rows = [
        ("AAPL beats earnings expectations, raises guidance",       0.8),
        ("Apple faces lawsuit over privacy violations",             -0.7),
        ("Apple announces record quarterly revenue",                 0.9),
        ("CEO steps down amid regulatory investigation",            -0.8),
        ("Analyst note: AAPL in-line with estimates",               0.0),
        ("Apple launches innovative new product line",               0.7),
        ("Apple loses major patent dispute, faces damages",         -0.6),
        ("Strong iPhone demand drives AAPL gains",                   0.75),
        ("Supply chain disruptions weigh on Apple outlook",         -0.5),
        ("Apple declares increased dividend and buyback program",    0.6),
    ]
    base = pd.Timestamp("2020-01-02 09:30:00")
    records = []
    for i, (hl, true_sent) in enumerate(rows):
        records.append({
            "timestamp":       base + pd.Timedelta(minutes=i * 47),
            "headline":        hl,
            "ticker":          "AAPL",
            "ravenpack_score": float(np.clip(true_sent + np.random.normal(0, 0.05), -1, 1)),
            "relevance_score": float(np.random.uniform(0.5, 1.0)),
        })
    return pd.DataFrame(records)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Headline sentiment scoring — LNA-Agent")
    parser.add_argument("--ticker",  type=str, default="AAPL")
    parser.add_argument("--suffix",  type=str, default="")
    parser.add_argument("--no-llm",  action="store_true", help="Skip LLM scoring")
    parser.add_argument("--test",    action="store_true", help="Run on synthetic data")
    args = parser.parse_args()

    RESULTS.mkdir(exist_ok=True)

    if args.test:
        print("\n[TEST MODE] Using synthetic headlines")
        df      = _make_test_headlines()
        out_tag = "test"
    else:
        out_tag = args.suffix
        # Try RavenPack first; fall back to Alpha Vantage headlines
        try:
            df = load_ravenpack(args.ticker, args.suffix)
            print(f"  Loaded {len(df)} RavenPack headlines")
        except FileNotFoundError:
            sys.path.insert(0, str(Path(__file__).parent))
            from news import load_headlines
            df = load_headlines(args.ticker, args.suffix)
            print(f"  Loaded {len(df)} Alpha Vantage headlines (no RavenPack file found)")

    df = score_all(df, llm=not args.no_llm, verbose=True)

    out_path = RESULTS / f"sentiment_{args.ticker}_{out_tag}.csv"
    df.to_csv(out_path, index=False)

    print(f"\n  Saved {len(df)} scored headlines → {out_path}")
    print(f"\n  Score summary:")
    for col in ["lm_score", "llm_score", "ravenpack_score"]:
        if col in df.columns and df[col].notna().any():
            s = df[col].dropna()
            print(f"    {col:<22} mean={s.mean():+.3f}  std={s.std():.3f}  n={len(s)}")

    print(f"\n  Sample (first 3):")
    for _, row in df.head(3).iterrows():
        lm  = row.get("lm_score",        float("nan"))
        llm = row.get("llm_score",       float("nan"))
        rp  = row.get("ravenpack_score", float("nan"))
        print(f"    [{row['timestamp']}]")
        print(f"    {row['headline'][:70]}")
        print(f"    LM={lm:+.3f}  LLM={llm:+.3f}  RP={rp:+.3f}\n")
