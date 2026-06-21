"""
src/signals/agent.py — LLM-agent trading signal.

One LLM call per event. Reimplements legacy/src/agent.py behavior without importing it.

Output columns added to panel:
  agent_relevance, agent_signal_agreement, agent_conviction_reasoning,
  agent_direction, agent_size, agent_horizon, agent_reasoning, agent_signal

agent_signal = direction_sign * size, where direction_sign is +1 (long), -1 (short), 0 (none).

Resumable cache: AGENT_DIR/{ticker}.jsonl, keyed by event_id.
                 Events already scored are skipped (no re-spend).
Does NOT import from legacy/.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from src.config import AGENT_DIR, MODEL_STRING, build_agent_system_prompt

load_dotenv()

_AGENT_DELAY = 0.15   # seconds between agent API calls

_DIRECTION_SIGN = {"long": 1, "short": -1, "none": 0}

_AGENT_COLS = [
    "agent_relevance",
    "agent_event_type",
    "agent_signal_agreement",
    "agent_conviction_reasoning",
    "agent_direction",
    "agent_size",
    "agent_horizon",
    "agent_reasoning",
    "agent_signal",
]

_anthropic_client = None


def _get_client():
    global _anthropic_client
    if _anthropic_client is None:
        import anthropic
        key = os.getenv("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError("ANTHROPIC_API_KEY not set in environment or .env")
        _anthropic_client = anthropic.Anthropic(api_key=key)
    return _anthropic_client


# ── Cache I/O ──────────────────────────────────────────────────────────────────

def _load_cache(ticker: str) -> dict[str, dict]:
    """Load existing agent results from AGENT_DIR/{ticker}.jsonl."""
    path = AGENT_DIR / f"{ticker}.jsonl"
    cache: dict[str, dict] = {}
    if not path.exists():
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                cache[rec["event_id"]] = rec
            except (json.JSONDecodeError, KeyError):
                pass
    return cache


def _append_cache(ticker: str, record: dict) -> None:
    """Append one scored record to AGENT_DIR/{ticker}.jsonl."""
    AGENT_DIR.mkdir(parents=True, exist_ok=True)
    path = AGENT_DIR / f"{ticker}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")


# ── Single-event scoring ───────────────────────────────────────────────────────

def _make_user_message(row: pd.Series) -> str:
    return (
        f'Headline: "{row["headline"]}"\n\n'
        f'LM sentiment : {float(row["lm_score"]):+.3f}\n'
        f'LLM sentiment: {float(row["llm_score"]):+.3f}\n'
        f'OFI (z-score): {float(row["ofi_z"]):+.3f}'
    )


def _parse_agent_response(text: str) -> dict[str, Any]:
    """Parse JSON from agent response; return defaults on error."""
    defaults: dict[str, Any] = {
        "relevance":            "unrelated",
        "event_type":           "other",
        "signal_agreement":     "neutral",
        "conviction_reasoning": "",
        "trade":                False,
        "direction":            "none",
        "size":                 0.0,
        "horizon":              "1min",
        "reasoning":            "[parse error]",
    }
    try:
        start = text.find("{")
        end   = text.rfind("}") + 1
        if start == -1 or end == 0:
            return defaults
        parsed = json.loads(text[start:end])
        defaults.update(parsed)
    except json.JSONDecodeError:
        pass
    return defaults


def _score_event(
    row: pd.Series,
    system_prompt: str,
) -> dict[str, Any]:
    """Call the agent LLM for one event row."""
    client = _get_client()
    user_msg = _make_user_message(row)
    try:
        msg = client.messages.create(
            model      = MODEL_STRING,
            max_tokens = 200,
            system     = [{
                "type":          "text",
                "text":          system_prompt,
                "cache_control": {"type": "ephemeral"},
            }],
            messages   = [{"role": "user", "content": user_msg}],
        )
        raw    = msg.content[0].text
        parsed = _parse_agent_response(raw)
    except Exception as e:
        print(f"  [agent] ERROR on event {row.get('event_id', '?')}: {e!r}")
        parsed = {
            "relevance":            "unrelated",
            "signal_agreement":     "neutral",
            "conviction_reasoning": "",
            "trade":                False,
            "direction":            "none",
            "size":                 0.0,
            "horizon":              "1min",
            "reasoning":            f"[API error: {e!r}]",
        }

    direction_sign = _DIRECTION_SIGN.get(str(parsed.get("direction", "none")), 0)
    size           = float(parsed.get("size", 0.0))

    return {
        "agent_relevance":            parsed.get("relevance",            "unrelated"),
        "agent_event_type":           parsed.get("event_type",           "other"),
        "agent_signal_agreement":     parsed.get("signal_agreement",     "neutral"),
        "agent_conviction_reasoning": parsed.get("conviction_reasoning", ""),
        "agent_direction":            parsed.get("direction",            "none"),
        "agent_size":                 size,
        "agent_horizon":              parsed.get("horizon",              "1min"),
        "agent_reasoning":            parsed.get("reasoning",            ""),
        "agent_signal":               direction_sign * size,
    }


# ── Batch scoring ──────────────────────────────────────────────────────────────

def score_ticker(
    ticker: str,
    panel: pd.DataFrame,
    limit: int | None = None,
    dry_run: bool = False,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the agent over all events for one ticker, resuming from cache.

    Parameters
    ----------
    ticker  : Stock ticker (used for system prompt and cache file).
    panel   : Per-event panel (must have event_id, headline, lm_score,
              llm_score, ofi_z columns).
    limit   : Process at most this many NEW events (resume skips are free).
    dry_run : If True, print what would be scored but make no API calls.
              Fills agent columns with NaN.

    Returns
    -------
    Copy of panel with agent signal columns merged in.
    """
    cache  = _load_cache(ticker)
    prompt = build_agent_system_prompt(ticker)

    rows    = panel.to_dict("records")
    results: list[dict] = []
    n_skipped = 0
    n_scored  = 0

    for row_dict in rows:
        eid = str(row_dict.get("event_id", ""))
        if eid in cache:
            results.append({**cache[eid], "event_id": eid})
            n_skipped += 1
            continue

        if dry_run:
            stub = {col: np.nan for col in _AGENT_COLS}
            stub["event_id"] = eid
            results.append(stub)
            continue

        if limit is not None and n_scored >= limit:
            stub = {col: np.nan for col in _AGENT_COLS}
            stub["event_id"] = eid
            results.append(stub)
            continue

        row = pd.Series(row_dict)
        agent_out = _score_event(row, prompt)
        record    = {"event_id": eid, **agent_out}
        _append_cache(ticker, record)
        results.append(record)
        n_scored += 1
        time.sleep(_AGENT_DELAY)

        if verbose and n_scored % 10 == 0:
            total = len(rows) - n_skipped
            print(f"  [agent] {ticker}: {n_scored}/{total} new events scored")

    if verbose:
        print(f"  [agent] {ticker}: {n_skipped} resumed, {n_scored} new")

    results_df = pd.DataFrame(results)
    out = panel.copy()
    agent_cols_present = [c for c in _AGENT_COLS if c in results_df.columns]
    out = out.merge(
        results_df[["event_id"] + agent_cols_present],
        on    = "event_id",
        how   = "left",
    )
    return out


def load_agent_scores(ticker: str) -> pd.DataFrame:
    """
    Load cached agent scores as a DataFrame.

    Returns empty DataFrame if no cache exists yet.
    """
    cache = _load_cache(ticker)
    if not cache:
        return pd.DataFrame(columns=["event_id"] + _AGENT_COLS)
    return pd.DataFrame(list(cache.values()))
