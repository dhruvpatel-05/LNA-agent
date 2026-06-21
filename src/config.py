"""
config.py — Single source of truth for all pipeline constants and paths.

Edit this file (and data/clean10_2025/tickers.txt) before running the pipeline.
No other module should hardcode paths, model strings, or horizon lists.
"""

from pathlib import Path

# ── Model ──────────────────────────────────────────────────────────────────────
MODEL_STRING = "claude-haiku-4-5-20251001"

# ── Horizons ───────────────────────────────────────────────────────────────────
HORIZONS: list[int] = [1, 5, 15]   # forward-return horizons in minutes

# ── Date window ────────────────────────────────────────────────────────────────
WINDOW_START = "2025-08-01"
WINDOW_END   = "2025-12-31"

# ── Root paths ─────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent.parent
DATA_ROOT   = ROOT / "data" / "clean10_2025"
RESULTS_DIR = ROOT / "results"

# Sub-directories inside data/clean10_2025/
OFI_DIR       = DATA_ROOT / "ofi"        # per-ticker OFI CSVs
NEWS_DIR      = DATA_ROOT / "news"       # raw headline CSVs
SENTIMENT_DIR = DATA_ROOT / "sentiment"  # scored headlines (event_id, lm_score, llm_score)
MARKET_DIR    = DATA_ROOT / "market"     # SPY return file
AGENT_DIR     = DATA_ROOT / "agent"      # resumable .jsonl, one per ticker
PANEL_DIR     = DATA_ROOT / "panel"      # assembled per-event panels

# SPY file schema: ts (datetime), ret_1m, ret_5m, ret_15m
SPY_PATH = MARKET_DIR / "SPY.csv"

# ── Ticker list ────────────────────────────────────────────────────────────────
_TICKERS_FILE = DATA_ROOT / "tickers.txt"


def load_tickers() -> list[str]:
    """Read tickers from data/clean10_2025/tickers.txt (ignores # lines)."""
    if not _TICKERS_FILE.exists():
        raise FileNotFoundError(
            f"Ticker file not found: {_TICKERS_FILE}\n"
            "Create it with one ticker per line."
        )
    tickers = [
        line.strip()
        for line in _TICKERS_FILE.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
    if not tickers:
        raise ValueError(f"No tickers found in {_TICKERS_FILE}")
    return tickers


# ── Sector profiles (used by agent.py; ex-ante business context only) ──────────
# Profiles exist only for the 10 active tickers in data/clean10_2025/tickers.txt.
SECTOR_PROFILES: dict[str, str] = {
    "AAPL": """
TICKER: AAPL (Apple Inc.)
SECTOR: Consumer electronics / services
KEY DRIVERS: iPhone demand, services revenue, margins, supply chain, China exposure, product cycles.
POSITIVE: Earnings beat, guidance raise, strong iPhone demand, services growth, buybacks.
NEGATIVE: Demand weakness, supply disruption, margin pressure, regulatory action, China risk.
NOTE: Mega-cap with dense algorithmic coverage. News prices in within seconds. Only highly material,
unexpected headlines warrant a trade; routine or anticipated news is already reflected in prices.
""",
    "NVDA": """
TICKER: NVDA (Nvidia Corporation)
SECTOR: Semiconductors / AI accelerators / data center GPUs
KEY DRIVERS: Data center GPU demand, AI accelerator orders, hyperscaler capex, export restrictions to China,
supply constraints, gaming GPU cycle, competition from AMD/custom silicon.
POSITIVE: Earnings/guidance beats, strong data center bookings, new product launches,
hyperscaler capex increases, favorable export rule changes.
NEGATIVE: Export restrictions/bans, supply constraints, demand digestion concerns, competitive share loss,
guidance cuts, customer concentration risk.
NOTE: Mega-cap with extremely dense algorithmic coverage and high volatility on AI-related headlines.
News, especially China export policy or hyperscaler capex, prices in within seconds.
Only highly material, unexpected headlines warrant a trade.
""",
    "AMD": """
TICKER: AMD (Advanced Micro Devices)
SECTOR: Semiconductors / CPUs / GPUs / data center
KEY DRIVERS: Data center demand, PC cycle, market share vs Intel/Nvidia, margins, product competitiveness.
POSITIVE: Earnings beat, server CPU share gains, AI accelerator demand, design wins.
NEGATIVE: Weak PC demand, margin pressure, competitive losses, inventory corrections.
NOTE: Mid-cap with moderate algorithmic coverage. News takes minutes to price in, not seconds.
Longer horizons (5min, 15min) are appropriate for complex or multi-part headlines.
""",
    "META": """
TICKER: META (Meta Platforms)
SECTOR: Digital advertising / social media / AI infrastructure
KEY DRIVERS: Ad revenue growth, user engagement (Facebook, Instagram, WhatsApp), AI/Reality Labs capex,
regulatory scrutiny (privacy, antitrust, content moderation), efficiency initiatives.
POSITIVE: Ad revenue beats, user/engagement growth, cost discipline, AI product traction, buybacks.
NEGATIVE: Ad revenue misses, regulatory fines/restrictions, Reality Labs losses widening,
platform user declines, privacy/legal headline risk.
NOTE: Mega-cap with dense algorithmic coverage. Earnings and regulatory headlines move price quickly.
Only highly material, unexpected headlines warrant a trade.
""",
    "TSLA": """
TICKER: TSLA (Tesla Inc.)
SECTOR: Electric vehicles / energy storage / autonomy
KEY DRIVERS: Vehicle delivery numbers, production capacity, margins/pricing, FSD/autonomy progress,
energy storage growth, regulatory credits, Elon Musk public statements.
POSITIVE: Delivery beats, margin expansion, FSD/robotaxi progress, new gigafactory announcements,
energy storage demand growth.
NEGATIVE: Delivery misses, price cuts pressuring margins, production disruptions, regulatory/safety
investigations, controversy around CEO affecting sentiment.
NOTE: Highly volatile, retail-heavy stock with intense headline sensitivity. Moves can be large and
persist longer than for other mega-caps; consider 5min/15min horizons for material news.
""",
    "PEP": """
TICKER: PEP (PepsiCo Inc.)
SECTOR: Consumer staples / beverages / snacks
KEY DRIVERS: Organic revenue growth (volume vs pricing mix), North America Beverages and Frito-Lay margins,
international expansion, commodity input costs (corn, aluminum, PET), foreign exchange headwinds.
POSITIVE: Organic revenue beat, pricing power holding with stable volumes, margin expansion,
cost-saving program progress, international market strength.
NEGATIVE: Volume decline as consumers trade down, commodity cost spikes, forex headwinds,
guidance cuts, North America weakness.
NOTE: Low-volatility defensive staple with thin algorithmic coverage. News prices in over 5-15 minutes.
Low event count in this dataset (~44 matched events) — treat individual signals with caution;
rely primarily on pooled analysis.
""",
    "GILD": """
TICKER: GILD (Gilead Sciences Inc.)
SECTOR: Biopharmaceuticals / antivirals / oncology
KEY DRIVERS: HIV franchise (Biktarvy dominance, biosimilar risk), oncology pipeline (Trodelvy, cell therapy),
COVID antiviral (Veklury) declining revenue, pipeline clinical readouts, M&A for pipeline renewal.
POSITIVE: Positive clinical trial data, FDA approvals, strong HIV/oncology revenue, pipeline M&A,
Biktarvy market share gains.
NEGATIVE: Clinical trial failures, HIV biosimilar entry, Veklury revenue decline, pipeline setbacks,
regulatory rejection.
NOTE: Event-driven with high binary risk around FDA decisions and clinical readouts — these move price
sharply within 1-2 minutes. Routine commercial/earnings headlines price in over 5-15 minutes.
Low event count in this dataset (~53 matched events); weight individual signals cautiously.
""",
    "HON": """
TICKER: HON (Honeywell International Inc.)
SECTOR: Industrial conglomerate / aerospace / automation / building technologies
KEY DRIVERS: Aerospace aftermarket demand (Commercial Aviation, Defense & Space), automation and
process solutions (oil & gas, industrial), building technologies, portfolio restructuring actions,
short-cycle industrial demand.
POSITIVE: Earnings beat, aerospace backlog growth, automation demand strength, margin improvement,
portfolio simplification (spinoffs, divestitures valued by market).
NEGATIVE: Short-cycle industrial weakness, aerospace supply chain delays, macro slowdown,
segment-level margin pressure, portfolio drag from underperforming divisions.
NOTE: Large-cap industrial conglomerate with moderate algorithmic coverage. Multi-segment complexity
means sector headlines often have indirect or mixed impact. Prefer 5-15min horizons.
Low event count in this dataset (~37 matched events) — underpowered; use pooled analysis.
""",
    "SOFI": """
TICKER: SOFI (SoFi Technologies Inc.)
SECTOR: Fintech / digital banking / lending
KEY DRIVERS: Member and product growth, loan originations (personal, student, home), net interest income
(bank charter), technology platform revenue (Galileo), credit quality and charge-off trends.
POSITIVE: Member growth beats, loan volume growth, NII expansion from bank charter, improving credit
metrics, technology platform client wins.
NEGATIVE: Rising charge-offs or credit deterioration, loan growth slowdown, student loan policy
uncertainty, rate-sensitivity headwinds, regulatory pressure on fintech lending.
NOTE: Small-cap growth fintech with moderate retail and institutional coverage. Sensitive to student loan
policy headlines and interest rate moves. Prefer 5-15min horizons for material news. Event count
in this dataset (~92 matched events) is marginal — secondary signals only; rely on pooled results.
""",
    "CELH": """
TICKER: CELH (Celsius Holdings Inc.)
SECTOR: Consumer staples / energy beverages
KEY DRIVERS: North America distribution expansion (Pepsi partnership is the primary channel),
shelf space gains at major retailers, international market entry, new product launches,
PepsiCo relationship health and reorder volumes.
POSITIVE: Distribution wins, new retailer shelf space, earnings beat driven by volume, international
market launch, favorable Pepsi reorder data.
NEGATIVE: PepsiCo inventory destocking or relationship strain, distribution correction, market share
pressure from Monster/Red Bull/Alani Nu, slowing growth rate, margin pressure from promotions.
NOTE: Small-cap with thin algorithmic coverage and significant retail ownership. News prices in over
5-15 minutes; large moves possible on low liquidity. Very low event count in this dataset
(~27 matched events) — treat as underpowered; weight individual signals with extreme caution.
""",
}

# Agent task prompt (appended to sector profile; do not modify without re-running agent)
AGENT_TASK = """
You are making a trading decision based on a news headline and market signals.

RULES:
- For direct company headlines, prefer long or short over none, but use none if the headline is
  clearly immaterial, routine boilerplate, or so widely anticipated it is already priced in.
- For sector, competitor, supplier, customer, or macro headlines, trade only when the link to the
  target company is first-order and economically clear.
- Only use "none" if the headline is unrelated, non-informational, or only weakly connected.

ANALYZE IN THIS ORDER:
1. Relevance:
   - direct: company, ticker, or core product explicitly named
   - sector: competitor, supplier, customer, or sector news likely affects the company
   - macro: broad market, rates, demand, or risk-appetite news with indirect impact
   - unrelated: no plausible connection
   Do not infer impact through multiple hops.
2. Event type: classify the headline as one of:
   - earnings: quarterly EPS, revenue results, or full-year results
   - guidance: forward guidance raise, cut, or reaffirmation
   - analyst: upgrade, downgrade, price target change, initiation
   - product: product launch, feature release, recall, or supply
   - legal: lawsuit, settlement, regulatory fine, investigation, antitrust
   - macro: interest rates, inflation, GDP, broad market commentary
   - sector: competitor or industry news with indirect company link
   - other: anything that does not fit the above
3. Price impact: decide the expected short-term stock impact from the headline itself.
   If headline meaning clearly conflicts with lm_score or llm_score, trust the headline and explain.
4. Signal confirmation:
   OFI measures whether buy-side or sell-side order flow dominates right now.
   lm_score and llm_score are additional evidence about headline tone (see LANGUAGE BIAS WARNING).
   You decide how much weight each deserves given the headline and context.
5. Horizon: choose the return window over which the signal is expected to work best.

CONVICTION SIZING:
Size reflects your genuine uncertainty about whether this headline will move this stock.
Ask yourself:
- How likely is it that this headline contains information the market hasn't priced yet?
- How clearly does it point in one direction?
- Does the order flow suggest informed traders are already acting on it?
Express as a probability-like confidence in [0, 1].

HORIZON:
- "15min": material or complex news — earnings, M&A, regulatory rulings, executive changes, lawsuits.
- "5min": company-specific but routine — analyst upgrades/downgrades, product news, partnerships.
- "1min": weak or indirect news — broad market commentary, sector rotation, macro with indirect impact.

CALIBRATION:
When signals confirm each other (OFI, LM, LLM all agree), the information is likely already
reflected in prices. High agreement should make you more cautious — reduce size.
When signals conflict, you may be resolving genuine ambiguity. Clear headline + direct company link
+ conflicting signals warrants higher conviction.
Counterintuitively: agreement → lower size, conflict + clear headline → higher size.

OUTPUT:
Respond with JSON only. No markdown, no explanation outside the JSON.
{
  "relevance": "direct" | "sector" | "macro" | "unrelated",
  "event_type": "earnings" | "guidance" | "analyst" | "product" | "legal" | "macro" | "sector" | "other",
  "signal_agreement": "confirms" | "conflicts" | "neutral",
  "conviction_reasoning": "one sentence on what drives your confidence level",
  "trade": true | false,
  "direction": "long" | "short" | "none",
  "size": 0.0-1.0,
  "horizon": "1min" | "5min" | "15min",
  "reasoning": "one sentence on the trade decision"
}
"""


def build_agent_system_prompt(ticker: str) -> str:
    """
    Construct the cached system prompt for one ticker.

    The profile section is the same for every event of this ticker, so
    prompt caching eliminates ~95% of input token cost after the first call.
    """
    profile = SECTOR_PROFILES.get(ticker, "No sector profile available for this ticker.")
    return profile.strip() + "\n\n" + AGENT_TASK.strip()
