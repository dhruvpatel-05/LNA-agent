"""
src/analysis/earnings_flag.py -- Fast keyword-based earnings event classifier.

Adds is_earnings and kw_event_type columns to any DataFrame that has a 'headline'
column. No API calls. Works on existing panels immediately.

Usage:
    from src.analysis.earnings_flag import add_earnings_flag
    panel = add_earnings_flag(panel)

Then stratify:
    earnings = panel[panel["is_earnings"]]
    non_earnings = panel[~panel["is_earnings"]]
"""

from __future__ import annotations
import re
import pandas as pd

# Earnings/results keywords — company reported actual numbers
_EARNINGS_KW = re.compile(
    r"\bearnings\b",
    re.IGNORECASE,
)

# Guidance keywords — forward-looking company statements
_GUIDANCE_KW = re.compile(
    r"\b(guidance|outlook|forecast|raises?\s+guidance|cuts?\s+guidance|"
    r"reaffirms?\s+guidance|full.year\s+(expects?|target)|"
    r"fiscal.year\s+(outlook|guidance)|sees?\s+(revenue|eps|earnings))\b",
    re.IGNORECASE,
)

# Analyst action keywords
_ANALYST_KW = re.compile(
    r"\b(upgrade|downgrade|initiat|price.target|raises?\s+pt|cuts?\s+pt|"
    r"outperform|underperform|buy.rating|sell.rating|overweight|underweight|"
    r"neutral.rating|hold.rating)\b",
    re.IGNORECASE,
)

# Legal/regulatory
_LEGAL_KW = re.compile(
    r"\b(lawsuit|sued|settlement|fine|penalty|investigation|antitrust|"
    r"regulatory|sec.charges?|doj|ftc|fda.approv|fda.reject|"
    r"class.action|probe)\b",
    re.IGNORECASE,
)

# M&A
_MA_KW = re.compile(
    r"\b(acqui|merger|takeover|buyout|bid.for|deal|acquires?|acquiring|"
    r"merges?\s+with|spun?\s+off|spin-off|divestitur)\b",
    re.IGNORECASE,
)

_KW_MAP = [
    ("earnings",  _EARNINGS_KW),
    ("guidance",  _GUIDANCE_KW),
    ("analyst",   _ANALYST_KW),
    ("legal",     _LEGAL_KW),
    ("ma",        _MA_KW),
]


def classify_headline(headline: str) -> str:
    """Return the first matching keyword event type, or 'other'."""
    for label, pattern in _KW_MAP:
        if pattern.search(headline):
            return label
    return "other"


def add_earnings_flag(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add two columns to df:
      kw_event_type : keyword-classified event type string
      is_earnings   : True if kw_event_type is 'earnings'

    Requires a 'headline' column. Returns a copy.
    """
    out = df.copy()
    out["kw_event_type"] = out["headline"].fillna("").apply(classify_headline)
    out["is_earnings"]   = out["kw_event_type"] == "earnings"
    return out


if __name__ == "__main__":
    # Quick smoke test on a few representative headlines
    test = [
        ("NVDA Q3 earnings beat: EPS $0.81 vs $0.74 estimate",         "earnings"),
        ("Apple raises guidance for fiscal year 2025",                   "guidance"),
        ("Goldman Sachs upgrades TSLA to buy, raises price target",     "analyst"),
        ("Meta faces antitrust investigation from FTC",                  "legal"),
        ("Microsoft acquires AI startup for $2B",                        "ma"),
        ("Fed holds rates steady at November meeting",                   "other"),
        ("Tesla deliveries miss analyst expectations",                   "earnings"),
    ]
    print(f"{'Headline':<55} {'Expected':<10} {'Got':<10} {'OK'}")
    print("-" * 85)
    all_ok = True
    for headline, expected in test:
        got = classify_headline(headline)
        ok = got == expected
        if not ok:
            all_ok = False
        print(f"{headline:<55} {expected:<10} {got:<10} {'YES' if ok else 'FAIL'}")
    print(f"\n{'All tests passed.' if all_ok else 'Some tests FAILED.'}")
