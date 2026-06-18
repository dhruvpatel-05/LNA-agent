"""
decisions.py — LLM and rule-based trade decision functions for LNA-Agent.
"""

import os
import re
import json
import time
import numpy as np

MODEL     = "claude-haiku-4-5-20251001"
API_DELAY = 0.02

_total_cost_usd: float = 0.0
_client = None


def get_total_cost() -> float:
    """Return the running total of estimated API cost."""
    return _total_cost_usd


def _get_client():
    global _client
    if _client is None:
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set in .env")
        _client = anthropic.Anthropic(api_key=key)
    return _client


def _parse_decision(text: str) -> dict:
    """
    Extract JSON from the model response.

    Handles responses that wrap JSON in markdown code blocks or include
    a sentence before/after the object.
    Returns a safe no-trade default on any parse failure.
    """
    text = re.sub(r"```(?:json)?", "", text).strip()

    match = re.search(r"\{.*?\}", text, re.DOTALL)
    if not match:
        return {"trade": False, "direction": "none", "size": 0.0,
                "horizon": "1min", "reasoning": "parse error",
                "relevance": "unrelated", "signal_agreement": "neutral",
                "conviction_reasoning": ""}
    try:
        d = json.loads(match.group())
    except json.JSONDecodeError:
        return {"trade": False, "direction": "none", "size": 0.0,
                "horizon": "1min", "reasoning": "json decode error",
                "relevance": "unrelated", "signal_agreement": "neutral",
                "conviction_reasoning": ""}

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
    # Lazy import avoids path issues when this module is imported as src.decisions
    from profiles import _build_system_prompt

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
        usage       = resp.usage
        cache_read  = getattr(usage, "cache_read_input_tokens",     0) or 0
        cache_write = getattr(usage, "cache_creation_input_tokens", 0) or 0
        input_tok   = getattr(usage, "input_tokens",                0) or 0
        output_tok  = getattr(usage, "output_tokens",               0) or 0
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
                "horizon": "1min", "reasoning": f"api error: {e}",
                "relevance": "unrelated", "signal_agreement": "neutral",
                "conviction_reasoning": ""}


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
        agrees   = (ofi_z > 0 and lm_score > 0) or (ofi_z < 0 and lm_score < 0)
        conflict = (ofi_z > ofi_threshold and lm_score < 0) or (ofi_z < -ofi_threshold and lm_score > 0)
        conviction = abs(lm_score)
        if agrees:
            conviction += min(abs(ofi_z) / 3.0, 0.4)
        elif conflict:
            conviction *= 0.5
        size    = float(np.clip(conviction, 0.0, 1.0))
        horizon = "5min" if abs(lm_score) >= 0.6 else "1min"
    elif abs(ofi_z) >= ofi_threshold:
        direction = "long" if ofi_z > 0 else "short"
        size      = float(np.clip(abs(ofi_z) / 3.0, 0.1, 0.8))
        horizon   = "1min"

    trade     = direction != "none"
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
