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

Flags:
  --no-llm     Ablation: skip API, use deterministic sector-aware rule instead.
  --anonymize  Replace ticker/company names in headlines with "TICKER" before
               passing to the agent (tests sensitivity to entity anchoring).
  --spillover  After per-ticker runs, compute cross-asset IC spillover matrix.

Usage:
  python src/agent.py --ticker NVDA --suffix 2016-01-01_2020-06-10
  python src/agent.py --suffix 2016-01-01_2020-06-10        # all tickers
  python src/agent.py --suffix 2016-01-01_2020-06-10 --no-llm
  python src/agent.py --suffix 2016-01-01_2020-06-10 --anonymize
  python src/agent.py --suffix 2016-01-01_2020-06-10 --spillover
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
load_dotenv()

import sys
import time
import re
import argparse
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)
sys.path.insert(0, str(Path(__file__).parent))

# ── Config ─────────────────────────────────────────────────────────────────────
RESULTS   = Path("results")
AGENT_RESULTS = RESULTS / "agent"
TICKERS = {
    "AAPL": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60",
    "JPM":  "/Users/dhruvpatel/Downloads/_data_dwn_32_302__JPM_2007-06-27_2021-07-01_10_60",
    "AMD":  "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AMD_2007-06-27_2022-01-01_1_60",
    "QCOM": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__QCOM_2007-06-27_2021-07-01_10_60",
    "NFLX": "/Users/dhruvpatel/Downloads/_data_dwn_32_302__NFLX_2007-06-27_2021-07-01_10_60",
    "NVDA": "",
    "META": "",
    "TSLA": "",
}
# ───────────────────────────────────────────────────────────────────────────────

import decisions as _decisions
from decisions import decide_llm, decide_rule, decision_to_signal
from evaluation import (
    compute_comparison,
    size_stratified_ic,
    relevance_stratified_ic,
    agreement_stratified_ic,
    _print_comparison,
    cross_asset_spillover,
    HORIZONS,
)

# ── Entity anonymization ───────────────────────────────────────────────────────

_ANONYMIZE_TERMS: dict[str, list[str]] = {
    "AAPL": ["Apple Inc", "Apple", "AAPL", "iPhone"],
    "AMD":  ["Advanced Micro Devices", "AMD"],
    "QCOM": ["Qualcomm", "QCOM"],
    "NFLX": ["Netflix", "NFLX"],
    "JPM":  ["JP Morgan", "JPMorgan", "JPM", "Chase"],
    "NVDA": ["Nvidia", "NVDA"],
    "META": ["Meta", "Facebook", "Instagram", "META"],
    "TSLA": ["Tesla", "Elon Musk", "TSLA"],
}


def anonymize_headlines(df: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """
    Replace ticker/company name variants in the headline column with "TICKER".

    Applies case-insensitive whole-word matching. Longer terms are replaced
    first to avoid partial-match conflicts (e.g. "Apple Inc" before "Apple").
    Returns a copy of df.
    """
    terms = _ANONYMIZE_TERMS.get(ticker, [ticker])
    out   = df.copy()
    for term in sorted(terms, key=len, reverse=True):
        pattern = re.compile(re.escape(term), re.IGNORECASE)
        out["headline"] = out["headline"].apply(
            lambda h: pattern.sub("TICKER", str(h))
        )
    return out


# ── Per-ticker run ─────────────────────────────────────────────────────────────

def run_ticker(
    ticker: str,
    aligned: pd.DataFrame,
    llm: bool,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Run the agent on every event in `aligned` for one ticker.

    Returns a copy of `aligned` with additional columns:
      agent_direction, agent_size, agent_horizon, agent_reasoning,
      agent_signal, agent_relevance, agent_signal_agreement,
      agent_conviction_reasoning
    """
    results = []
    total   = len(aligned)

    for i, row in aligned.iterrows():
        hl        = str(row.get("headline",   ""))
        ofi_z     = float(row.get("ofi_z",    0.0))
        lm_score  = float(row.get("lm_score", 0.0))
        llm_score = float(row.get("llm_score", 0.0) or 0.0)

        if llm:
            d = decide_llm(ticker, hl, ofi_z, lm_score, llm_score)
            time.sleep(_decisions.API_DELAY)
        else:
            d = decide_rule(ticker, ofi_z, lm_score)

        d["agent_signal"] = decision_to_signal(d)
        results.append(d)

        if verbose and (i + 1) % 25 == 0:
            pct = (i + 1) / total * 100
            print(f"  {ticker}: {i + 1}/{total} ({pct:.0f}%)  "
                  f"last→ {d['direction']} {d['size']:.2f} @ {d['horizon']}")

    out = aligned.copy()
    out["agent_direction"] = [r["direction"]    for r in results]
    out["agent_size"]      = [r["size"]         for r in results]
    out["agent_horizon"]   = [r["horizon"]      for r in results]
    out["agent_reasoning"] = [r["reasoning"]    for r in results]
    out["agent_signal"]    = [r["agent_signal"] for r in results]
    out["agent_relevance"]            = [r.get("relevance",            "unrelated") for r in results]
    out["agent_signal_agreement"]     = [r.get("signal_agreement",     "neutral")   for r in results]
    out["agent_conviction_reasoning"] = [r.get("conviction_reasoning", "")          for r in results]

    dirs   = out["agent_direction"].value_counts().to_dict()
    trades = out[out["agent_direction"] != "none"]
    mean_s = trades["agent_size"].mean() if len(trades) else 0.0
    print(f"\n  Decision summary for {ticker}:")
    print(f"    long={dirs.get('long',0)}  short={dirs.get('short',0)}  "
          f"none={dirs.get('none',0)}  "
          f"mean_size={mean_s:.3f}  trade_rate={len(trades)/len(out)*100:.0f}%")

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


def _validate_agent_inputs(
    ticker: str,
    aligned: pd.DataFrame,
    llm: bool,
    suffix: str,
) -> None:
    """
    Refuse LLM-agent runs when required sentiment inputs are incomplete.

    Running the agent with missing `llm_score` values would feed malformed
    prompts like "LLM sentiment: nan" into the trading decision step.
    """
    required = {"headline", "ofi_z", "lm_score"}
    missing_cols = sorted(required - set(aligned.columns))
    if missing_cols:
        raise ValueError(
            f"{ticker}: aligned panel is missing required columns: {', '.join(missing_cols)}"
        )

    if llm:
        if "llm_score" not in aligned.columns:
            raise ValueError(
                f"{ticker}: aligned panel has no llm_score column. "
                f"Rerun pipeline with --llm first."
            )
        missing_llm = int(aligned["llm_score"].isna().sum())
        if missing_llm:
            date_from, date_to = suffix.split("_", 1)
            raise ValueError(
                f"{ticker}: aligned panel has {missing_llm} missing llm_score values. "
                f"Rerun `python src/run_pipeline.py --ticker {ticker} "
                f"--date-from {date_from} --date-to {date_to} --llm` "
                f"to complete sentiment before running the agent."
            )


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LNA-Agent trade decision layer")
    parser.add_argument("--ticker",     type=str,  default=None,
                        help="Single ticker (omit for all tickers)")
    parser.add_argument("--suffix",     type=str,  required=True,
                        help="Date suffix, e.g. 2016-01-01_2020-06-10")
    parser.add_argument("--no-llm",     action="store_true",
                        help="Ablation: use deterministic rule, skip API")
    parser.add_argument("--max-events", type=int,  default=None,
                        help="Stop after N events per ticker (for quick testing)")
    parser.add_argument("--anonymize",  action="store_true",
                        help="Replace entity names in headlines with TICKER")
    parser.add_argument("--spillover",  action="store_true",
                        help="Compute cross-asset IC spillover after per-ticker runs")
    args = parser.parse_args()

    llm        = not args.no_llm
    suffix     = args.suffix
    max_events = args.max_events
    mode_tag   = ("llm" if llm else "rule") + ("_anon" if args.anonymize else "")

    ticker_list = [args.ticker] if args.ticker else list(TICKERS.keys())

    AGENT_RESULTS.mkdir(parents=True, exist_ok=True)

    from align import load_aligned

    all_comps: dict[str, pd.DataFrame] = {}

    for ticker in ticker_list:
        print(f"\n{'─' * 60}")
        print(f"  AGENT: {ticker}  ({'LLM' if llm else 'rule-based'} mode"
              + ("  [anonymized]" if args.anonymize else "") + ")")
        print(f"{'─' * 60}")

        try:
            aligned = load_aligned(ticker, suffix)
        except FileNotFoundError as e:
            print(f"  [SKIP] {e}")
            continue

        if aligned.empty:
            print(f"  [SKIP] Empty aligned panel for {ticker}")
            continue

        try:
            _validate_agent_inputs(ticker, aligned, llm=llm, suffix=suffix)
        except ValueError as e:
            print(f"  [SKIP] {e}")
            continue

        if max_events is not None:
            aligned = aligned.head(max_events).copy()
        print(f"  Events: {len(aligned)}"
              + (f"  (capped at {max_events})" if max_events is not None else ""))

        if args.anonymize:
            aligned = anonymize_headlines(aligned, ticker)

        ticked = run_ticker(ticker, aligned, llm=llm, verbose=True)

        comp = compute_comparison(ticked)
        _print_comparison(ticker, comp)

        size_stratified_ic(ticked, ticker, suffix, mode_tag)
        relevance_stratified_ic(ticked, ticker, suffix, mode_tag)
        agreement_stratified_ic(ticked, ticker, suffix, mode_tag)

        ic_path = AGENT_RESULTS / f"agent_ic_{ticker}_{suffix}_{mode_tag}.csv"
        comp.to_csv(ic_path, index=False)
        print(f"  Saved → {ic_path}")

        panel_path = AGENT_RESULTS / f"agent_panel_{ticker}_{suffix}_{mode_tag}.parquet"
        ticked.to_parquet(panel_path, index=False)

        print(f"  Estimated API cost so far: ${_decisions.get_total_cost():.4f}")

        all_comps[ticker] = comp

    # ── Spillover ─────────────────────────────────────────────────────────────
    if args.spillover:
        cross_asset_spillover(suffix)

    # ── Cross-ticker summary ──────────────────────────────────────────────────
    if len(all_comps) > 1:
        print(f"\n{'═' * 70}")
        print(f"  CROSS-TICKER SUMMARY  ({'LLM agent' if llm else 'rule ablation'})")
        print(f"{'═' * 70}")

        h_labels = [f"Agent {h}m" for h in HORIZONS]
        print(f"  {'Ticker':<8}  " + "  ".join(f"{lbl:>10}" for lbl in h_labels))
        print(f"  {'-'*8}  " + "  ".join(f"{'-'*10}" for _ in HORIZONS))

        for ticker, comp in all_comps.items():
            row_parts = [f"  {ticker:<8}"]
            for h in HORIZONS:
                sub = comp[(comp["strategy"] == "Agent") & (comp["horizon"] == h)]
                if sub.empty or pd.isna(sub.iloc[0]["ic"]):
                    row_parts.append(f"  {'NaN':>10}")
                else:
                    v   = sub.iloc[0]["ic"]
                    pv  = sub.iloc[0]["pval"]
                    sig = ("***" if pv < 0.01 else "**" if pv < 0.05
                           else "*" if pv < 0.10 else " ")
                    row_parts.append(f"  {v:>+9.4f}{sig}")
            print("".join(row_parts))

        print()

        print(f"  {'Ticker':<8}  {'Agent chosen horizon':>22}  {'N':>5}")
        print(f"  {'-'*8}  {'-'*22}  {'-'*5}")
        for ticker, comp in all_comps.items():
            sub = comp[comp["strategy"] == "Agent chosen horizon"]
            if sub.empty or pd.isna(sub.iloc[0]["ic"]):
                print(f"  {ticker:<8}  {'NaN':>22}  {0:>5}")
                continue
            v   = sub.iloc[0]["ic"]
            pv  = sub.iloc[0]["pval"]
            n   = int(sub.iloc[0]["n"])
            sig = ("***" if pv < 0.01 else "**" if pv < 0.05
                   else "*" if pv < 0.10 else " ")
            print(f"  {ticker:<8}  {v:>+21.4f}{sig}  {n:>5}")

        print()

        all_rows = pd.concat(all_comps.values(), ignore_index=True)
        print(f"  Mean IC across {len(all_comps)} tickers:")
        avail_h = sorted([
            h for h in all_rows["horizon"].unique()
            if isinstance(h, (int, float, np.integer)) and not pd.isna(h)
        ])
        h_hdrs = [f"{int(h)}-min" for h in avail_h]
        print(f"  {'Strategy':<22}  " + "  ".join(f"{lbl:>9}" for lbl in h_hdrs))
        print(f"  {'-'*22}  " + "  ".join(f"{'-'*9}" for _ in avail_h))
        parts = [f"  {'Agent':<22}"]
        for h in avail_h:
            sub     = all_rows[(all_rows["strategy"] == "Agent") & (all_rows["horizon"] == h)]
            mean_ic = sub["ic"].mean(skipna=True)
            parts.append(f"  {'NaN':>9}" if pd.isna(mean_ic) else f"  {mean_ic:>+9.4f}")
        print("".join(parts))

        chosen = all_rows[all_rows["strategy"] == "Agent chosen horizon"]
        if not chosen.empty:
            chosen_mean = chosen["ic"].mean(skipna=True)
            chosen_text = "NaN" if pd.isna(chosen_mean) else f"{chosen_mean:+.4f}"
            print(f"  {'Agent chosen horizon':<22}  {chosen_text:>9}")
        print()

    print(f"\n  Total estimated API cost: ${_decisions.get_total_cost():.4f}")
