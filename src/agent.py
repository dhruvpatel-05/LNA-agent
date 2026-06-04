"""
agent.py — Agentic trade decision layer
LNA-Agent: Lobster News Alpha

Replaces the fixed OFI × LLM rule with a per-ticker LLM agent that reasons
about sector context, LOB state, and headline together before committing to a
direction, size, and horizon.

Each event feeds the agent three signals:
  lm_score   — Loughran-McDonald polarity [-1, 1]
  llm_score  — claude-haiku sentiment score [-1, 1]
  ofi_z      — normalised order-flow imbalance

The agent receives a cached sector profile (cache_control: ephemeral) with
ex-ante business context only. Profiles describe what kinds of news matter for
each company, not historical IC findings or fitted trading rules.

Agent output (structured JSON):
  trade      — bool
  direction  — "long" | "short" | "none"
  size       — float [0, 1]
  horizon    — "1min" | "5min" | "15min"
  reasoning  — one sentence

Evaluation:
  agent_signal = direction_sign * size   (+1 long, -1 short, 0 no trade)
  Spearman IC of agent_signal vs ret_1m / ret_5m / ret_15m
  Compared to fixed rule: ofi_z * llm_score

Flags:
  --no-llm   Ablation: skip API, use deterministic sector-aware rule instead.
             Isolates the value of LLM reasoning over hand-coded sector heuristics.

Usage:
  python src/agent.py --ticker NVDA --suffix 2016-01-01_2020-06-10
  python src/agent.py --suffix 2016-01-01_2020-06-10        # all tickers
  python src/agent.py --suffix 2016-01-01_2020-06-10 --no-llm
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
import json
import time
import re
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).parent))

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS   = Path("results")
AGENT_RESULTS = RESULTS / "agent"
MODEL     = "claude-haiku-4-5-20251001"
API_DELAY = 0.02      # seconds between calls
HORIZONS  = [1, 5, 15]
TICKERS = {
    "AAPL": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60",
    "JPM":  "/Users/dhruvpatel/Downloads/_data_dwn_32_302__JPM_2007-06-27_2021-07-01_10_60",
    "AMD":  "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AMD_2007-06-27_2022-01-01_1_60",
    "QCOM": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__QCOM_2007-06-27_2021-07-01_10_60",
    "NFLX": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__NFLX_2007-06-27_2021-07-01_10_60",
}
# ───────────────────────────────────────────────────────────────────────────────


# ── Sector profiles ────────────────────────────────────────────────────────────
# Each profile is injected into the system prompt (prompt-cached per ticker).
# Profiles must be ex-ante only: business model, common news types, and economic
# interpretation. Do not encode historical IC results or fitted ticker effects.

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


# ── Cost tracker ──────────────────────────────────────────────────────────────

_total_cost_usd: float = 0.0

# ── Anthropic client ───────────────────────────────────────────────────────────

_client = None

def _get_client():
    global _client
    if _client is None:
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        _client = anthropic.Anthropic(api_key=key)
    return _client


# ── Decision: LLM agent ────────────────────────────────────────────────────────

def _build_system_prompt(ticker: str) -> str:
    """
    Construct the cached system prompt for a ticker.

    The profile section is the same for every event of this ticker, so
    prompt caching eliminates ~95% of input token cost after the first call.
    """
    profile = _SECTOR_PROFILES.get(ticker, "No sector profile available.")
    return profile.strip() + "\n\n" + _AGENT_TASK.strip()


def _parse_decision(text: str) -> dict:
    """
    Extract JSON from the model response.

    Handles responses that wrap JSON in markdown code blocks or include
    a sentence before/after the object.
    Returns a safe no-trade default on any parse failure.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()

    # Extract first {...} block
    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {"trade": False, "direction": "none", "size": 0.0,
                "horizon": "1min", "reasoning": "parse error"}
    try:
        d = json.loads(match.group())
    except json.JSONDecodeError:
        return {"trade": False, "direction": "none", "size": 0.0,
                "horizon": "1min", "reasoning": "json decode error"}

    # Sanitise fields
    d.setdefault("trade",                False)
    d.setdefault("direction",            "none")
    d.setdefault("size",                 0.0)
    d.setdefault("horizon",              "1min")
    d.setdefault("reasoning",            "")
    d.setdefault("relevance",            "unrelated")
    d.setdefault("signal_agreement",     "neutral")
    d.setdefault("conviction_reasoning", "")

    if d["direction"] not in ("long", "short", "none"):
        d["direction"] = "none"
    if d["horizon"] not in ("1min", "5min", "15min"):
        d["horizon"] = "1min"
    d["size"] = float(np.clip(float(d["size"]), 0.0, 1.0))
    if not d["trade"]:
        d["direction"] = "none"
        d["size"]      = 0.0
    if d["relevance"] not in ("direct", "sector", "macro", "unrelated"):
        d["relevance"] = "unrelated"
    if d["signal_agreement"] not in ("confirms", "conflicts", "neutral"):
        d["signal_agreement"] = "neutral"

    return d


def decide_llm(
    ticker: str,
    headline: str,
    ofi_z: float,
    lm_score: float,
    llm_score: float,
) -> dict:
    """
    Call the LLM agent for one event.

    System prompt (sector profile) is marked cache_control='ephemeral'.
    All events for the same ticker share the same system prompt content,
    so all but the first call in each 5-min window are cache hits.

    Returns parsed decision dict on success; no-trade default on API error.
    """
    client = _get_client()

    user_msg = (
        f'Headline: "{headline}"\n\n'
        f"LM sentiment : {lm_score:+.3f}\n"
        f"LLM sentiment: {llm_score:+.3f}\n"
        f"OFI (z-score): {ofi_z:+.3f}"
    )

    try:
        resp = client.messages.create(
            model      = MODEL,
            max_tokens = 200,
            system     = [{
                "type":          "text",
                "text":          _build_system_prompt(ticker),
                "cache_control": {"type": "ephemeral"},
            }],
            messages   = [{"role": "user", "content": user_msg}],
        )
        decision = _parse_decision(resp.content[0].text)

        global _total_cost_usd
        usage = resp.usage
        cache_read  = getattr(usage, "cache_read_input_tokens",      0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens",  0) or 0
        input_tok   = getattr(usage, "input_tokens",                 0) or 0
        output_tok  = getattr(usage, "output_tokens",                0) or 0
        _total_cost_usd += (
            cache_read  * 0.08  / 1e6
            + cache_write * 0.80  / 1e6
            + (input_tok - cache_read - cache_write) * 0.80 / 1e6
            + output_tok  * 4.0   / 1e6
        )

        return decision
    except Exception as e:
        print(f"  [AGENT ERROR] {ticker}: {e!r} | '{headline[:50]}'")
        return {"trade": False, "direction": "none", "size": 0.0,
                "horizon": "1min", "reasoning": f"api error: {e}"}


# ── Decision: deterministic ablation (--no-llm) ────────────────────────────────

def decide_rule(
    ticker: str,
    ofi_z: float,
    lm_score: float,
) -> dict:
    """
    Generic deterministic fallback used when --no-llm is set.

    This deliberately avoids ticker-specific backtest findings. It uses only
    ex-ante information available at the event time: LM sentiment and current
    OFI. The rule is a bias-controlled baseline, not a fitted sector strategy.
    """
    direction = "none"
    size      = 0.0
    horizon   = "1min"

    sentiment_threshold = 0.15
    ofi_threshold       = 0.75

    if abs(lm_score) >= sentiment_threshold:
        direction = "long" if lm_score > 0 else "short"
        agrees = (ofi_z > 0 and lm_score > 0) or (ofi_z < 0 and lm_score < 0)
        conflict = (ofi_z > ofi_threshold and lm_score < 0) or (ofi_z < -ofi_threshold and lm_score > 0)
        conviction = abs(lm_score)
        if agrees:
            conviction += min(abs(ofi_z) / 3.0, 0.4)
        elif conflict:
            conviction *= 0.5
        size = float(np.clip(conviction, 0.0, 1.0))
        horizon = "5min" if abs(lm_score) >= 0.6 else "1min"
    elif abs(ofi_z) >= ofi_threshold:
        direction = "long" if ofi_z > 0 else "short"
        size = float(np.clip(abs(ofi_z) / 3.0, 0.1, 0.8))
        horizon = "1min"

    trade = direction != "none"
    reasoning = (
        f"generic ex-ante rule: lm={lm_score:+.2f}, "
        f"ofi={ofi_z:+.2f}, no ticker-specific fit"
    )

    if not trade:
        size = 0.0

    return {
        "trade":     trade,
        "direction": direction,
        "size":      round(size, 4),
        "horizon":   horizon,
        "reasoning": reasoning,
    }


# ── Signal conversion ─────────────────────────────────────────────────────────

def decision_to_signal(d: dict) -> float:
    """
    Convert a trade decision to a scalar signal for IC computation.

    signal =  size   if direction == "long"
           = -size   if direction == "short"
           =  0.0    if no trade
    """
    if not d.get("trade") or d.get("direction") == "none":
        return 0.0
    return d["size"] if d["direction"] == "long" else -d["size"]


# ── Per-ticker run ─────────────────────────────────────────────────────────────

def run_ticker(
    ticker: str,
    aligned: pd.DataFrame,
    llm: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the agent on every event in `aligned` for one ticker.

    For each row the agent receives:
      headline, lm_score, llm_score (NaN-safe), ofi_z

    Returns a copy of `aligned` with additional columns:
      agent_direction, agent_size, agent_horizon, agent_reasoning,
      agent_signal
    """
    results = []
    total   = len(aligned)

    for i, row in aligned.iterrows():
        hl        = str(row.get("headline",  ""))
        ofi_z     = float(row.get("ofi_z",    0.0))
        lm_score  = float(row.get("lm_score", 0.0))
        llm_score = float(row.get("llm_score", 0.0) or 0.0)  # NaN → 0

        if llm:
            d = decide_llm(ticker, hl, ofi_z, lm_score, llm_score)
            time.sleep(API_DELAY)
        else:
            d = decide_rule(ticker, ofi_z, lm_score)

        d["agent_signal"] = decision_to_signal(d)
        results.append(d)

        if verbose and (i + 1) % 25 == 0:
            pct = (i + 1) / total * 100
            print(f"  {ticker}: {i + 1}/{total} ({pct:.0f}%)  "
                  f"last→ {d['direction']} {d['size']:.2f} @ {d['horizon']}")

    out = aligned.copy()
    out["agent_direction"] = [r["direction"] for r in results]
    out["agent_size"]      = [r["size"]      for r in results]
    out["agent_horizon"]   = [r["horizon"]   for r in results]
    out["agent_reasoning"] = [r["reasoning"] for r in results]
    out["agent_signal"]    = [r["agent_signal"] for r in results]
    out["agent_relevance"]            = [r.get("relevance",            "unrelated") for r in results]
    out["agent_signal_agreement"]     = [r.get("signal_agreement",     "neutral")   for r in results]
    out["agent_conviction_reasoning"] = [r.get("conviction_reasoning", "")          for r in results]

    # ── Direction distribution summary ───────────────────────────────────────
    dirs   = out["agent_direction"].value_counts().to_dict()
    trades = out[out["agent_direction"] != "none"]
    mean_s = trades["agent_size"].mean() if len(trades) else 0.0
    print(f"\n  Decision summary for {ticker}:")
    print(f"    long={dirs.get('long',0)}  short={dirs.get('short',0)}  "
          f"none={dirs.get('none',0)}  "
          f"mean_size={mean_s:.3f}  trade_rate={len(trades)/len(out)*100:.0f}%")

    # ── 5 sample reasoning strings (evenly spaced) ───────────────────────────
    print(f"\n  Sample decisions ({ticker}):")
    indices = np.linspace(0, len(out) - 1, 5, dtype=int)
    for idx in indices:
        row = out.iloc[idx]
        hl  = str(row.get("headline", ""))[:55]
        lm  = row.get("lm_score", float("nan"))
        oz  = row.get("ofi_z",    float("nan"))
        print(f"  [{idx:>3}] {row['agent_direction']:>5} {row['agent_size']:.2f} "
              f"@ {row['agent_horizon']}  "
              f"lm={lm:+.2f} ofi={oz:+.2f}")
        print(f"        headline : {hl}")
        print(f"        reasoning: {row['agent_reasoning']}")
        print(f"        relevance: {row['agent_relevance']}  agreement: {row['agent_signal_agreement']}")
        print(f"        conviction: {row['agent_conviction_reasoning'][:100]}")

    return out


# ── IC evaluation ─────────────────────────────────────────────────────────────

def _ic(signal: pd.Series, returns: pd.Series) -> tuple[float, float]:
    """Spearman IC and p-value; NaN if insufficient data."""
    mask = signal.notna() & returns.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    ic, pval = spearmanr(signal[mask], returns[mask])
    return float(ic), float(pval)


def compute_comparison(ticked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build IC comparison table for one ticker.

    Compares agent_signal IC against the fixed rule (ofi_z * llm_score)
    and the constituent signals (LM alone, LLM alone, OFI alone) at each
    horizon.  The fixed rule is the pre-agent baseline.

    Returns DataFrame with columns:
      strategy, horizon, ic, pval, n
    """
    # agent_signal may be all-zero if llm_score was all NaN; guard
    fixed_rule = ticked_df["ofi_z"] * ticked_df["llm_score"].fillna(0)

    signals = {
        "Agent": ticked_df["agent_signal"],
        "LM":    ticked_df["lm_score"],
        "LLM":   ticked_df["llm_score"],
        "OFI":   ticked_df["ofi_z"],
    }

    rows = []
    for name, sig in signals.items():
        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in ticked_df.columns:
                continue
            mask = sig.notna() & ticked_df[ret_col].notna()
            n    = int(mask.sum())
            ic, pval = _ic(sig, ticked_df[ret_col])
            rows.append({"strategy": name, "horizon": h,
                         "ic": ic, "pval": pval, "n": n})

    # Fair agent-specific metric: evaluate each event against the horizon the
    # agent selected. Fixed baselines remain evaluated separately at 1/5/15 min.
    chosen_returns = []
    chosen_signal = ticked_df["agent_signal"]
    horizon_to_return = {
        "1min": "ret_1m",
        "5min": "ret_5m",
        "15min": "ret_15m",
    }
    for _, row in ticked_df.iterrows():
        h = str(row.get("agent_horizon", "1min"))
        ret_col = horizon_to_return.get(h, "ret_1m")
        chosen_returns.append(row.get(ret_col, np.nan))
    chosen_returns = pd.Series(chosen_returns, index=ticked_df.index)
    mask = chosen_signal.notna() & chosen_returns.notna()
    ic, pval = _ic(chosen_signal, chosen_returns)
    rows.append({
        "strategy": "Agent chosen horizon",
        "horizon": "chosen",
        "ic": ic,
        "pval": pval,
        "n": int(mask.sum()),
    })

    # Bootstrap baseline: randomly assign returns to events, compute IC
    rng = np.random.default_rng(42)
    boot_ics = []
    ret_cols = [f"ret_{h}m" for h in HORIZONS if f"ret_{h}m" in ticked_df.columns]
    for _ in range(1000):
        random_col = ret_cols[rng.integers(len(ret_cols))]
        shuffled = ticked_df[random_col].values.copy()
        rng.shuffle(shuffled)
        shuffled = pd.Series(shuffled, index=ticked_df.index)
        b_mask = chosen_signal.notna() & shuffled.notna()
        if b_mask.sum() >= 10:
            b_ic, _ = spearmanr(chosen_signal[b_mask], shuffled[b_mask])
            boot_ics.append(float(b_ic))
    rows.append({
        "strategy": "Agent chosen horizon (random baseline)",
        "horizon": "chosen_random",
        "ic": float(np.mean(boot_ics)) if boot_ics else np.nan,
        "pval": np.nan,
        "n": int(mask.sum()),
    })

    # Horizon-group evaluation: for each horizon choice, evaluate all return windows
    for h_choice in ["1min", "5min", "15min"]:
        subset = ticked_df[ticked_df["agent_horizon"] == h_choice]
        if len(subset) < 10:
            continue
        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in subset.columns:
                continue
            sig_sub = subset["agent_signal"]
            ret_sub = subset[ret_col]
            h_ic, h_pval = _ic(sig_sub, ret_sub)
            h_mask = sig_sub.notna() & ret_sub.notna()
            rows.append({
                "strategy": f"Agent (chose {h_choice})",
                "horizon": h,
                "ic": h_ic,
                "pval": h_pval,
                "n": int(h_mask.sum()),
            })

    return pd.DataFrame(rows)


# ── Size-stratified IC ────────────────────────────────────────────────────────

def size_stratified_ic(
    ticked_df: pd.DataFrame,
    ticker: str,
    suffix: str,
    mode_tag: str,
) -> pd.DataFrame:
    """
    IC by agent_size quartile × return horizon.

    Only events where agent_direction != 'none' are included.
    Returns DataFrame with columns: quartile, horizon, ic, pval, n, mean_size.
    """
    df = ticked_df[ticked_df["agent_direction"] != "none"].copy()
    if len(df) < 20:
        return pd.DataFrame(columns=["quartile", "horizon", "ic", "pval", "n", "mean_size"])

    df["size_quartile"] = pd.qcut(df["agent_size"], q=4, labels=["Q1", "Q2", "Q3", "Q4"])

    rows = []
    for quartile in ["Q1", "Q2", "Q3", "Q4"]:
        sub = df[df["size_quartile"] == quartile]
        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in sub.columns:
                continue
            sig = sub["agent_signal"]
            ret = sub[ret_col]
            h_mask = sig.notna() & ret.notna()
            h_ic, h_pval = _ic(sig, ret)
            rows.append({
                "quartile":  quartile,
                "horizon":   h,
                "ic":        h_ic,
                "pval":      h_pval,
                "n":         int(h_mask.sum()),
                "mean_size": float(sub["agent_size"].mean()),
            })

    result = pd.DataFrame(rows)
    print(result.to_string(index=False))

    out_path = AGENT_RESULTS / f"size_ic_{ticker}_{suffix}_{mode_tag}.csv"
    result.to_csv(out_path, index=False)
    print(f"  Saved size IC → {out_path}")

    return result


def relevance_stratified_ic(ticked_df: pd.DataFrame, ticker: str, suffix: str, mode_tag: str) -> pd.DataFrame:
    rows = []
    for rel in ["direct", "sector", "macro", "unrelated"]:
        sub = ticked_df[ticked_df["agent_relevance"] == rel]
        if len(sub) < 10:
            continue
        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in sub.columns:
                continue
            ic, pval = _ic(sub["agent_signal"], sub[ret_col])
            rows.append({"relevance": rel, "horizon": h,
                         "ic": ic, "pval": pval, "n": len(sub),
                         "mean_size": sub["agent_size"].mean()})
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  -- Relevance-stratified IC ({ticker}) --")
        print(df.to_string(index=False))
        df.to_csv(AGENT_RESULTS / f"relevance_ic_{ticker}_{suffix}_{mode_tag}.csv", index=False)
    return df


def agreement_stratified_ic(ticked_df: pd.DataFrame, ticker: str, suffix: str, mode_tag: str) -> pd.DataFrame:
    rows = []
    for ag in ["confirms", "conflicts", "neutral"]:
        sub = ticked_df[ticked_df["agent_signal_agreement"] == ag]
        if len(sub) < 10:
            continue
        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in sub.columns:
                continue
            ic, pval = _ic(sub["agent_signal"], sub[ret_col])
            rows.append({"agreement": ag, "horizon": h,
                         "ic": ic, "pval": pval, "n": len(sub),
                         "mean_size": sub["agent_size"].mean()})
    df = pd.DataFrame(rows)
    if not df.empty:
        print(f"\n  -- Agreement-stratified IC ({ticker}) --")
        print(df.to_string(index=False))
        df.to_csv(AGENT_RESULTS / f"agreement_ic_{ticker}_{suffix}_{mode_tag}.csv", index=False)
    return df


# ── Pretty-print ──────────────────────────────────────────────────────────────

def _print_comparison(ticker: str, comp: pd.DataFrame) -> None:
    strategies = ["Agent", "LM", "LLM", "OFI"]

    print(f"\n{'═' * 70}")
    print(f"  IC COMPARISON — {ticker}")
    print(f"{'═' * 70}")
    print(f"  {'Strategy':<22}  {'1-min':>9}  {'5-min':>9}  {'15-min':>9}  {'N':>5}")
    print(f"  {'-' * 22}  {'-' * 9}  {'-' * 9}  {'-' * 9}  {'-' * 5}")

    def _fmt(strat: str, h: int) -> str:
        sub = comp[(comp["strategy"] == strat) & (comp["horizon"] == h)]
        if sub.empty or np.isnan(sub.iloc[0]["ic"]):
            return f"{'NaN':>9}"
        v   = sub.iloc[0]["ic"]
        pv  = sub.iloc[0]["pval"]
        sig = ("***" if pv < 0.01 else "**" if pv < 0.05
               else "*" if pv < 0.10 else " ")
        return f"{v:>+8.4f}{sig}"

    for strat in strategies:
        sub = comp[comp["strategy"] == strat]
        if sub.empty:
            continue
        n_  = int(sub.iloc[0]["n"])
        row = f"  {strat:<22}  " + "  ".join(_fmt(strat, h) for h in HORIZONS)
        print(f"{row}  {n_:>5}")

    chosen = comp[comp["strategy"] == "Agent chosen horizon"]
    if not chosen.empty:
        v = chosen.iloc[0]["ic"]
        pv = chosen.iloc[0]["pval"]
        n = int(chosen.iloc[0]["n"])
        if np.isnan(v):
            shown = "NaN"
        else:
            sig = ("***" if pv < 0.01 else "**" if pv < 0.05
                   else "*" if pv < 0.10 else "")
            shown = f"{v:+.4f}{sig}"
        print(f"  {'Agent chosen horizon':<22}  {shown:>9}  n={n}")

    rand_row = comp[comp["strategy"] == "Agent chosen horizon (random baseline)"]
    if not rand_row.empty:
        r_ic = rand_row.iloc[0]["ic"]
        r_n  = int(rand_row.iloc[0]["n"])
        if not np.isnan(r_ic):
            print(f"  {'Agent chosen (random)':22}  {r_ic:>9.4f}  n={r_n}")
    print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LNA-Agent trade decision layer")
    parser.add_argument("--ticker", type=str, default=None,
                        help="Single ticker (omit for all tickers)")
    parser.add_argument("--suffix", type=str, required=True,
                        help="Date suffix, e.g. 2016-01-01_2020-06-10")
    parser.add_argument("--no-llm", action="store_true",
                        help="Ablation: use deterministic sector-aware rule, skip API")
    parser.add_argument("--max-events", type=int, default=None,
                        help="Stop after N events per ticker (for quick testing)")
    args = parser.parse_args()

    llm        = not args.no_llm
    suffix     = args.suffix
    max_events = args.max_events

    # Determine ticker list
    if args.ticker:
        ticker_list = [args.ticker]
    else:
        ticker_list = list(TICKERS.keys())

    AGENT_RESULTS.mkdir(parents=True, exist_ok=True)

    from align import load_aligned

    all_comps: dict[str, pd.DataFrame] = {}

    for ticker in ticker_list:
        print(f"\n{'─' * 60}")
        print(f"  AGENT: {ticker}  ({'LLM' if llm else 'rule-based'} mode)")
        print(f"{'─' * 60}")

        # Load aligned panel (must already exist from run_all.py or run_pipeline.py)
        try:
            aligned = load_aligned(ticker, suffix)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        if aligned.empty:
            print(f"  [SKIP] Empty aligned panel for {ticker}")
            continue

        if max_events is not None:
            aligned = aligned.head(max_events).copy()
        print(f"  Events: {len(aligned)}"
              + (f"  (capped at {max_events})" if max_events is not None else ""))

        # Run agent
        ticked = run_ticker(ticker, aligned, llm=llm, verbose=True)

        # Evaluate
        comp = compute_comparison(ticked)
        _print_comparison(ticker, comp)

        # Size-stratified IC
        mode_tag = "llm" if llm else "rule"
        size_ic  = size_stratified_ic(ticked, ticker, suffix, mode_tag)
        relevance_stratified_ic(ticked, ticker, suffix, mode_tag)
        agreement_stratified_ic(ticked, ticker, suffix, mode_tag)

        # Save
        ic_path  = AGENT_RESULTS / f"agent_ic_{ticker}_{suffix}_{mode_tag}.csv"
        comp.to_csv(ic_path, index=False)
        print(f"  Saved → {ic_path}")

        # Save full ticked panel (for manual inspection / further analysis)
        panel_path = AGENT_RESULTS / f"agent_panel_{ticker}_{suffix}_{mode_tag}.parquet"
        ticked.to_parquet(panel_path, index=False)

        print(f"  Estimated API cost so far: ${_total_cost_usd:.4f}")

        all_comps[ticker] = comp

    # ── Cross-ticker summary ──────────────────────────────────────────────────
    if len(all_comps) > 1:
        print(f"\n{'═' * 70}")
        print(f"  CROSS-TICKER SUMMARY  ({'LLM agent' if llm else 'rule ablation'})")
        print(f"{'═' * 70}")
        print(f"  {'Ticker':<8}  "
              f"{'Agent 1m':>9}  {'Agent 5m':>9}  {'Agent 15m':>10}")
        print(f"  {'-'*8}  {'-'*9}  {'-'*9}  {'-'*10}")

        for ticker, comp in all_comps.items():
            row_parts = [f"  {ticker:<8}"]
            for h in HORIZONS:
                sub = comp[(comp["strategy"] == "Agent") & (comp["horizon"] == h)]
                if sub.empty or np.isnan(sub.iloc[0]["ic"]):
                    row_parts.append(f"  {'NaN':>9}")
                else:
                    v   = sub.iloc[0]["ic"]
                    pv  = sub.iloc[0]["pval"]
                    sig = ("***" if pv < 0.01 else "**" if pv < 0.05
                           else "*" if pv < 0.10 else " ")
                    row_parts.append(f"  {v:>+8.4f}{sig}")
            print("".join(row_parts))

        print()

        print(f"  {'Ticker':<8}  {'Agent chosen horizon':>22}  {'N':>5}")
        print(f"  {'-'*8}  {'-'*22}  {'-'*5}")
        for ticker, comp in all_comps.items():
            sub = comp[comp["strategy"] == "Agent chosen horizon"]
            if sub.empty or np.isnan(sub.iloc[0]["ic"]):
                print(f"  {ticker:<8}  {'NaN':>22}  {0:>5}")
                continue
            v = sub.iloc[0]["ic"]
            pv = sub.iloc[0]["pval"]
            n = int(sub.iloc[0]["n"])
            sig = ("***" if pv < 0.01 else "**" if pv < 0.05
                   else "*" if pv < 0.10 else " ")
            print(f"  {ticker:<8}  {v:>+21.4f}{sig}  {n:>5}")

        print()

        # Aggregate: mean agent IC and mean fixed IC per horizon
        all_rows = pd.concat(all_comps.values(), ignore_index=True)
        print(f"  Mean IC across {len(all_comps)} tickers:")
        print(f"  {'Strategy':<22}  {'1-min':>9}  {'5-min':>9}  {'15-min':>9}")
        print(f"  {'-'*22}  {'-'*9}  {'-'*9}  {'-'*9}")
        for strat in ["Agent"]:
            parts = [f"  {strat:<22}"]
            for h in HORIZONS:
                sub = all_rows[(all_rows["strategy"] == strat) & (all_rows["horizon"] == h)]
                mean_ic = sub["ic"].mean(skipna=True)
                if np.isnan(mean_ic):
                    parts.append(f"  {'NaN':>9}")
                else:
                    parts.append(f"  {mean_ic:>+9.4f}")
            print("".join(parts))
        chosen = all_rows[all_rows["strategy"] == "Agent chosen horizon"]
        if not chosen.empty:
            chosen_mean = chosen["ic"].mean(skipna=True)
            if np.isnan(chosen_mean):
                chosen_text = "NaN"
            else:
                chosen_text = f"{chosen_mean:+.4f}"
            print(f"  {'Agent chosen horizon':<22}  {chosen_text:>9}")
        print()

    print(f"\n  Total estimated API cost: ${_total_cost_usd:.4f}")
