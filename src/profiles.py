"""
profiles.py — Sector profiles and agent task prompt for LNA-Agent.
"""

_SECTOR_PROFILES = {
    "AAPL": """
TICKER: AAPL (Apple Inc.)
SECTOR: Consumer electronics / services
KEY DRIVERS: iPhone demand, services revenue, margins, supply chain, China exposure, product cycles.
POSITIVE: Earnings beat, guidance raise, strong iPhone demand, services growth, buybacks.
NEGATIVE: Demand weakness, supply disruption, margin pressure, regulatory action, China risk.
NOTE: Mega-cap with dense algorithmic coverage. News prices in within seconds. Only highly material,
unexpected headlines warrant a trade; routine or anticipated news is already reflected in prices.
""",
    "JPM": """
TICKER: JPM (JPMorgan Chase)
SECTOR: Banking / investment banking
KEY DRIVERS: Net interest income, yield curve, credit quality, trading revenue, regulation.
POSITIVE: Earnings beat, strong trading revenue, low credit losses, favorable rate dynamics.
NEGATIVE: Loan loss provisions, credit deterioration, regulatory penalties, recession risk.
NOTE: Large-cap, highly liquid, macro-sensitive. Fed and rates news prices in faster than
company-specific headlines. Prefer short horizons; macro events are often priced before the headline lands.
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
    "QCOM": """
TICKER: QCOM (Qualcomm)
SECTOR: Wireless semiconductors / licensing
KEY DRIVERS: Smartphone demand, 5G modem share, licensing revenue, Apple relationship, automotive wins.
POSITIVE: Earnings beat, handset demand strength, licensing settlements, automotive design wins.
NEGATIVE: Handset weakness, Apple modem substitution, licensing disputes, China demand drop.
NOTE: Event-driven stock with high institutional coverage. Binary events (patent rulings, licensing
settlements) cause sharp moves that price in within 1-2 minutes. Prefer 1min horizon for most
headlines; use 5min only for earnings or complex regulatory events requiring digestion.
""",
    "NVDA": """
TICKER: NVDA (Nvidia Corporation)
SECTOR: Semiconductors / AI accelerators / data center GPUs
KEY DRIVERS: Data center GPU demand, AI accelerator orders, hyperscaler capex, export restrictions to China,
supply (TSMC) constraints, gaming GPU cycle, competition from AMD/custom silicon.
POSITIVE: Earnings/guidance beats, strong data center bookings, new product launches (Blackwell etc.),
hyperscaler capex increases, favorable export rule changes.
NEGATIVE: Export restrictions/bans, supply constraints, demand digestion concerns, competitive share loss,
guidance cuts, customer concentration risk headlines.
NOTE: Mega-cap with extremely dense algorithmic coverage and high volatility on AI-related headlines.
News, especially anything touching China export policy or hyperscaler capex, prices in within seconds.
Only highly material, unexpected headlines warrant a trade.
""",
    "META": """
TICKER: META (Meta Platforms)
SECTOR: Digital advertising / social media / AI infrastructure
KEY DRIVERS: Ad revenue growth, user engagement (Facebook, Instagram, WhatsApp), AI/Reality Labs capex,
regulatory scrutiny (privacy, antitrust, content moderation), efficiency initiatives.
POSITIVE: Ad revenue beats, user/engagement growth, cost discipline, AI product traction, buybacks.
NEGATIVE: Ad revenue misses, regulatory fines/restrictions (EU/US), Reality Labs losses widening,
platform user declines, privacy/legal headline risk.
NOTE: Mega-cap with dense algorithmic coverage. Earnings and regulatory headlines move price quickly.
Only highly material, unexpected headlines warrant a trade; routine news is largely priced in within seconds.
""",
    "TSLA": """
TICKER: TSLA (Tesla Inc.)
SECTOR: Electric vehicles / energy storage / autonomy
KEY DRIVERS: Vehicle delivery numbers, production capacity, margins/pricing, FSD/autonomy progress,
energy storage growth, regulatory credits, Elon Musk public statements and other ventures.
POSITIVE: Delivery beats, margin expansion, FSD/robotaxi progress, new gigafactory or capacity announcements,
energy storage demand growth.
NEGATIVE: Delivery misses, price cuts pressuring margins, production disruptions, regulatory/safety
investigations, controversy around Elon Musk affecting sentiment.
NOTE: Highly volatile, retail-heavy stock with intense headline sensitivity, including news unrelated to
core business (CEO statements, other ventures). Moves can be large and persist longer than for other
mega-caps; consider 5min/15min horizons for material news.
""",
    "NFLX": """
TICKER: NFLX (Netflix)
SECTOR: Streaming entertainment
KEY DRIVERS: Subscriber growth, retention, content spending, pricing power, ad-tier adoption, competition.
POSITIVE: Subscriber beats, pricing increases, hit content, ad revenue growth, international expansion.
NEGATIVE: Subscriber misses, content cost overruns, competition gains, password crackdown backfires, guidance cuts.
NOTE: Mid-to-large cap with event-driven volatility. Earnings and subscriber data cause large sustained
moves; use 15min for these. Routine content or competition headlines price in over 5min.
""",
}

_AGENT_TASK = """
You are making a trading decision based on a news headline and market signals.

RULES:
- For direct company headlines, prefer long or short over none, but use none if the headline is clearly immaterial, routine boilerplate, or so widely anticipated it has already been priced in.
- For sector, competitor, supplier, customer, or macro headlines, trade only when the link to the target company is first-order and economically clear.
- Only use "none" if the headline is unrelated, non-informational, or only weakly connected through vague sector association.
- Direction should follow the expected price impact of the headline.

ANALYZE IN THIS ORDER:
1. Relevance:
   - direct: company, ticker, or core product explicitly named
   - sector: competitor, supplier, customer, or sector news likely affects the company
   - macro: broad market, rates, demand, or risk-appetite news with indirect impact
   - unrelated: no plausible connection
   Do not infer impact through multiple hops. A supplier/customer/competitor headline must have a clear first-order link to the target company.
2. Event type:
   earnings, guidance, analyst, product, legal/regulatory, M&A, macro, sector, other
3. Price impact:
   Decide the expected short-term stock impact from the headline itself.
   If headline meaning clearly conflicts with lm_score or llm_score, trust the headline meaning and explain the conflict.
4. Signal confirmation:
   OFI measures whether buy-side or sell-side order flow dominates right now.
   Treat it as evidence about what informed traders may already know.
   lm_score and llm_score are additional evidence about headline tone.
   You decide how much weight each deserves given the headline and context.
   These are inputs to your reasoning, not rules to follow mechanically.
5. Horizon:
   Choose the return window over which the same signed conviction signal is expected to work best.

CONVICTION SIZING:
There are no fixed rules. Size reflects your genuine uncertainty about
whether this headline will move this stock in the next few minutes.

Ask yourself:
- How likely is it that this headline contains information the market
  hasn't priced yet?
- How clearly does it point in one direction?
- Does the order flow suggest informed traders are already acting on it?

Express your answer as a probability-like confidence in [0, 1].
A size of 0.8 means you are quite confident. A size of 0.3 means you
see a signal but have real doubt. Size 0.0 only when direction is none.

HORIZON:
- The output creates one alpha signal: direction × size.
- Horizon does not create a separate alpha formula; it identifies the return window for evaluating that same signal.
- Use "15min" when material or complex news should take longer to digest: earnings, M&A, regulatory rulings, executive changes, lawsuits, guidance.
- Use "5min" for company-specific but routine news: analyst upgrades/downgrades, product news, partnerships.
- Use "1min" for weak or indirect news: broad market commentary, sector rotation, macro with indirect impact.

RELEVANCE:
- direct: company/ticker named or core product named
- sector: sector/competitor/customer news likely affects company
- broad: macro/market commentary only weakly related

Use lower size for sector or broad relevance. Use "none" for unrelated or vague multi-hop links.
Sector and macro headlines rarely move individual stocks at intraday horizons.
Unless the first-order link to the target company is unusually direct and
specific, size sector events at 0.1-0.25 and macro events at 0.0-0.15.
Do not mechanically multiply signals. The fixed OFI×LLM baseline does that;
your job is to make a structured judgment.

CALIBRATION:
When signals confirm each other (OFI, LM, LLM all agree on direction),
the information is likely already reflected in prices. High agreement
should make you more cautious, not more confident — reduce size, as the
market has probably already acted on this signal.
When signals conflict (e.g. OFI opposes sentiment direction), you may be
resolving genuine ambiguity the market has not yet priced. If your
reading of the headline is clear and the company link is direct, this
warrants higher conviction than confirmed signals would.
Counterintuitively: agreement → lower size, conflict + clear headline → higher size.

OUTPUT:
Respond with JSON only. No markdown, no explanation outside the JSON.
{
  "relevance": "direct" | "sector" | "macro" | "unrelated",
  "signal_agreement": "confirms" | "conflicts" | "neutral",
  "conviction_reasoning": "one sentence on what drives your confidence level",
  "trade": true | false,
  "direction": "long" | "short" | "none",
  "size": 0.0-1.0,
  "horizon": "1min" | "5min" | "15min",
  "reasoning": "one sentence on the trade decision"
}

relevance: how directly does this headline connect to the ticker?
  direct: company, ticker, or core product explicitly named
  sector: competitor, supplier, customer, or sector news with clear first-order link
  macro: broad market/rates/demand with indirect impact
  unrelated: no plausible connection
signal_agreement: do lm_score, llm_score, and ofi_z point the same direction as your trade?
  confirms: majority of signals agree with your direction
  conflicts: majority of signals oppose your direction
  neutral: signals are mixed or near zero
conviction_reasoning: one sentence on the main factor raising or lowering your size
trade: true when direction is long or short, false only when direction is none
direction: "long", "short", or "none"
size: 0.0-1.0 (0.0 only when direction is none)
horizon: "1min", "5min", or "15min"
reasoning: one sentence explaining the trade decision
"""


def _build_system_prompt(ticker: str) -> str:
    """
    Construct the cached system prompt for a ticker.

    The profile section is the same for every event of this ticker, so
    prompt caching eliminates ~95% of input token cost after the first call.
    """
    profile = _SECTOR_PROFILES.get(ticker, "No sector profile available.")
    return profile.strip() + "\n\n" + _AGENT_TASK.strip()
