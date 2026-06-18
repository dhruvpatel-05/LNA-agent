"""
sentiment_finbert.py — FinBERT headline sentiment scoring
LNA-Agent: Lobster News Alpha

Batched inference with ProsusAI/finbert (transformers), auto-detecting
CPU/GPU. Produces per-headline positive/negative/neutral probabilities and
a signed score matching the existing lm_score / llm_score sign convention
(positive = bullish, negative = bearish, in [-1, 1]).

Usage:
  from sentiment_finbert import score_headlines
  out = score_headlines(df["headline"].tolist())
"""

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

MODEL_NAME = "ProsusAI/finbert"

_tokenizer = None
_model     = None
_device    = None


def _get_model():
    global _tokenizer, _model, _device
    if _model is None:
        _device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        _model     = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
        _model.to(_device)
        _model.eval()
    return _tokenizer, _model, _device


def score_headlines(headlines: list[str], batch_size: int = 32) -> pd.DataFrame:
    """
    Score a list of headlines with FinBERT.

    Returns a DataFrame with columns:
      headline, finbert_pos, finbert_neg, finbert_neutral, finbert_score

    finbert_score = finbert_pos - finbert_neg, in [-1, 1], matching the
    sign convention of lm_score / llm_score (positive = bullish).
    """
    tokenizer, model, device = _get_model()

    # FinBERT label order: 0=positive, 1=negative, 2=neutral
    pos, neg, neutral = [], [], []

    for start in range(0, len(headlines), batch_size):
        batch = [str(h) for h in headlines[start:start + batch_size]]
        inputs = tokenizer(batch, padding=True, truncation=True,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            logits = model(**inputs).logits
            probs  = torch.softmax(logits, dim=-1).cpu().numpy()

        pos.extend(probs[:, 0].tolist())
        neg.extend(probs[:, 1].tolist())
        neutral.extend(probs[:, 2].tolist())

    pos     = np.array(pos)
    neg     = np.array(neg)
    neutral = np.array(neutral)

    return pd.DataFrame({
        "headline":        [str(h) for h in headlines],
        "finbert_pos":     pos,
        "finbert_neg":     neg,
        "finbert_neutral": neutral,
        "finbert_score":   pos - neg,
    })
