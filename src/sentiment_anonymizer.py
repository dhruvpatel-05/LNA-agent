"""
sentiment_anonymizer.py — Entity anonymization for headline sentiment scoring.
LNA-Agent: Lobster News Alpha

Replaces company names with "COMPANY" and ticker symbols with "TICKER" using
case-insensitive, whole-word matching (\\b boundaries). Used to test whether
sentiment/agent signals are sensitive to entity anchoring rather than the
underlying news content.

Usage:
  from sentiment_anonymizer import anonymize_headline, anonymize_batch
  anonymize_headline("Apple Inc shares rise on iPhone demand", ["AAPL"])
  # -> "COMPANY shares rise on iPhone demand"
"""

import re

# Company-name variants per ticker — replaced with "COMPANY"
_COMPANY_TERMS: dict[str, list[str]] = {
    "AAPL": ["Apple Inc", "Apple"],
    "AMD":  ["Advanced Micro Devices"],
    "QCOM": ["Qualcomm"],
    "NFLX": ["Netflix"],
    "JPM":  ["JPMorgan", "JP Morgan", "J.P. Morgan"],
}

# Ticker-symbol variants per ticker — replaced with "TICKER"
_TICKER_TERMS: dict[str, list[str]] = {
    "AAPL": ["AAPL"],
    "AMD":  ["AMD"],
    "QCOM": ["QCOM"],
    "NFLX": ["NFLX"],
    "JPM":  ["JPM"],
}


def anonymize_headline(headline: str, ticker_list: list[str]) -> str:
    """
    Replace company names with "COMPANY" and ticker symbols with "TICKER".

    Case-insensitive, whole-word matching only (regex \\b boundaries).
    Longer terms are substituted first to avoid partial-match conflicts
    (e.g. "Apple Inc" before "Apple", "JPMorgan" before "JPM").
    """
    replacements: list[tuple[str, str]] = []
    for ticker in ticker_list:
        for term in _COMPANY_TERMS.get(ticker, []):
            replacements.append((term, "COMPANY"))
        for term in _TICKER_TERMS.get(ticker, [ticker]):
            replacements.append((term, "TICKER"))

    replacements.sort(key=lambda pair: len(pair[0]), reverse=True)

    out = headline
    for term, repl in replacements:
        pattern = re.compile(r"\b" + re.escape(term) + r"\b", re.IGNORECASE)
        out = pattern.sub(repl, out)
    return out


def anonymize_batch(headlines: list[str], ticker_list: list[str]) -> list[str]:
    """Anonymize a list of headlines. See anonymize_headline."""
    return [anonymize_headline(str(h), ticker_list) for h in headlines]
