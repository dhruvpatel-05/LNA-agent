"""
test_finbert_sentiment.py — Smoke test for FinBERT headline scoring.

Loads ~20 sample headlines from an existing processed-news CSV, runs
score_headlines(), and checks shape/columns/score range/no nulls.

Usage:
  python tests/test_finbert_sentiment.py
"""

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from sentiment_finbert import score_headlines

HEADLINES_CSV = (Path(__file__).resolve().parent.parent / "data" / "news" / "processed"
                 / "AAPL_headlines_2016-01-01_2020-12-31.csv")
N_SAMPLES = 20


def test_score_headlines():
    df = pd.read_csv(HEADLINES_CSV)
    headlines = df["headline"].head(N_SAMPLES).tolist()

    out = score_headlines(headlines, batch_size=32)

    assert len(out) == len(headlines)
    assert list(out.columns) == [
        "headline", "finbert_pos", "finbert_neg", "finbert_neutral", "finbert_score",
    ]
    assert out["finbert_score"].between(-1.0, 1.0).all()
    assert not out[["finbert_pos", "finbert_neg", "finbert_neutral", "finbert_score"]].isnull().values.any()

    print(f"\nScored {len(out)} headlines from {HEADLINES_CSV.name}")
    print("\nSample rows:")
    print(out.head(5).to_string(index=False))


if __name__ == "__main__":
    test_score_headlines()
