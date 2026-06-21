"""
src/analysis/agent_deep.py -- Four targeted experiments to find positive agent signal.

Experiments:
  1. agent_chosen_horizon  -- evaluate each trade at the horizon the agent chose
  2. relevance_stratified  -- IC broken out by agent_relevance label
  3. high_conviction       -- filter to agent_size >= threshold
  4. conflict_only         -- filter to agent_signal_agreement == "conflicts"

Each function returns a DataFrame with IC, 95% bootstrap CI, p-value, and n.
run_all() runs all four and returns a dict of DataFrames.

Does NOT import from legacy/.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AGENT_DIR, PANEL_DIR, HORIZONS
from src.stats_rigor import bootstrap_ic_ci, benjamini_hochberg
from src.signals.rules import add_rule_signals

TICKERS = ["AAPL", "NVDA", "AMD", "META", "TSLA", "PEP", "GILD", "HON", "SOFI", "CELH"]
_HORIZON_MAP = {"1min": 1, "5min": 5, "15min": 15}
N_BOOT = 1000
SEED = 42


# ── helpers ────────────────────────────────────────────────────────────────────

def _load_panel(ticker: str) -> pd.DataFrame:
    path = PANEL_DIR / f"{ticker}.csv"
    df = pd.read_csv(path)
    df = add_rule_signals(df)
    return df


def _load_agent(ticker: str) -> pd.DataFrame:
    path = AGENT_DIR / f"{ticker}.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def _merge(ticker: str) -> pd.DataFrame:
    panel = _load_panel(ticker)
    agent = _load_agent(ticker)
    df = panel.merge(agent, on="event_id", how="inner")
    df["ticker"] = ticker
    return df


def _ic_row(sig: pd.Series, ret: pd.Series, label: str, group: str,
            n_boot: int = N_BOOT, seed: int = SEED) -> dict:
    sub = pd.DataFrame({"s": sig, "r": ret}).dropna()
    n = len(sub)
    row = {"label": label, "group": group, "n": n,
           "ic": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan}
    if n < 10:
        return row
    res = bootstrap_ic_ci(sub["s"], sub["r"], n_boot=n_boot, seed=seed)
    row.update({"ic": res["ic"], "ci_lo": res["ci_lo"],
                "ci_hi": res["ci_hi"], "p": res["p_boot"]})
    return row


def _apply_bh(df: pd.DataFrame, q: float = 0.05) -> pd.DataFrame:
    df = df.copy()
    valid = df["p"].notna()
    reject = pd.Series(False, index=df.index)
    if valid.any():
        rej_arr, _ = benjamini_hochberg(df.loc[valid, "p"].values, q=q)
        reject[valid] = rej_arr
    df["bh_reject"] = reject.astype(bool)
    return df


def _load_all() -> dict[str, pd.DataFrame]:
    panels = {}
    for t in TICKERS:
        try:
            panels[t] = _merge(t)
        except FileNotFoundError:
            pass
    return panels


# ── Experiment 1: agent-chosen horizon ────────────────────────────────────────

def exp1_agent_chosen_horizon(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    Evaluate each agent trade at the horizon the agent actually chose.
    Trades with direction=none (agent_signal=0) are excluded.
    """
    rows = []
    pooled_sig, pooled_ret = [], []

    for ticker, df in panels.items():
        traded = df[df["agent_signal"] != 0].copy()
        if len(traded) < 5:
            continue

        # Map agent_horizon string -> return column
        traded["eval_ret"] = traded.apply(
            lambda r: r.get(f"ret_{_HORIZON_MAP.get(r['agent_horizon'], 1)}m", np.nan),
            axis=1
        )

        row = _ic_row(traded["agent_signal"], traded["eval_ret"],
                      label="agent_chosen_h", group=ticker)
        row["trades"] = len(traded)
        rows.append(row)

        pooled_sig.append(traded["agent_signal"])
        pooled_ret.append(traded["eval_ret"])

    # Pooled
    if pooled_sig:
        ps = pd.concat(pooled_sig)
        pr = pd.concat(pooled_ret)
        row = _ic_row(ps, pr, label="agent_chosen_h", group="POOLED")
        row["trades"] = (ps != 0).sum()
        rows.append(row)

    return _apply_bh(pd.DataFrame(rows))


# ── Experiment 2: relevance-stratified IC ─────────────────────────────────────

def exp2_relevance_stratified(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    IC of agent_signal vs ret_1m, broken out by agent_relevance label.
    Pooled across all tickers for each relevance class.
    """
    relevance_classes = ["direct", "sector", "macro", "unrelated"]
    rows = []

    for rel in relevance_classes:
        sigs, rets = [], []
        per_ticker_n = []

        for ticker, df in panels.items():
            sub = df[(df["agent_relevance"] == rel) & (df["agent_signal"] != 0)]
            per_ticker_n.append(len(sub))
            if len(sub) < 3:
                continue
            for h in [1, 5, 15]:
                ret_col = f"ret_{h}m"
                row = _ic_row(sub["agent_signal"], sub[ret_col],
                              label=f"rel={rel}", group=f"{ticker}_h{h}")
                row["horizon"] = h
                row["ticker"] = ticker
                rows.append(row)

            sigs.append(sub["agent_signal"])
            rets.append(sub["ret_1m"])

        # Pooled across tickers at 1min for this relevance class
        if sigs:
            ps = pd.concat(sigs)
            pr = pd.concat(rets)
            for h in [1, 5, 15]:
                rets_h = []
                for ticker, df in panels.items():
                    sub = df[(df["agent_relevance"] == rel) & (df["agent_signal"] != 0)]
                    if len(sub) >= 3:
                        rets_h.append(sub[f"ret_{h}m"])
                pooled_r = pd.concat(rets_h) if rets_h else pd.Series(dtype=float)
                row = _ic_row(ps, pooled_r, label=f"rel={rel}", group=f"POOLED_h{h}")
                row["horizon"] = h
                row["ticker"] = "POOLED"
                rows.append(row)

    df_out = pd.DataFrame(rows)
    return _apply_bh(df_out)


# ── Experiment 3: high-conviction filter ──────────────────────────────────────

def exp3_high_conviction(panels: dict[str, pd.DataFrame],
                          thresholds: list[float] = (0.3, 0.5, 0.7)) -> pd.DataFrame:
    """
    IC of agent_signal vs ret at 1min, filtered to agent_size >= threshold.
    Compares full-traded vs high-conviction subsets across tickers.
    """
    rows = []

    for thresh in thresholds:
        pooled_sig, pooled_ret = [], []

        for ticker, df in panels.items():
            sub = df[(df["agent_signal"] != 0) & (df["agent_size"] >= thresh)]
            if len(sub) < 5:
                continue
            for h in [1, 5, 15]:
                row = _ic_row(sub["agent_signal"], sub[f"ret_{h}m"],
                              label=f"size>={thresh}", group=f"{ticker}_h{h}")
                row["threshold"] = thresh
                row["horizon"] = h
                row["ticker"] = ticker
                row["pct_of_trades"] = len(sub) / max(
                    len(df[df["agent_signal"] != 0]), 1)
                rows.append(row)

            pooled_sig.append(sub["agent_signal"])
            pooled_ret.append(sub["ret_1m"])

        if pooled_sig:
            ps = pd.concat(pooled_sig)
            for h in [1, 5, 15]:
                rets_h = []
                for ticker, df in panels.items():
                    sub = df[(df["agent_signal"] != 0) & (df["agent_size"] >= thresh)]
                    if len(sub) >= 5:
                        rets_h.append(sub[f"ret_{h}m"])
                pr = pd.concat(rets_h) if rets_h else pd.Series(dtype=float)
                row = _ic_row(ps, pr, label=f"size>={thresh}", group=f"POOLED_h{h}")
                row["threshold"] = thresh
                row["horizon"] = h
                row["ticker"] = "POOLED"
                row["pct_of_trades"] = len(ps) / max(
                    sum(len(d[d["agent_signal"] != 0]) for d in panels.values()), 1)
                rows.append(row)

    return _apply_bh(pd.DataFrame(rows))


# ── Experiment 4: conflict-only trades ────────────────────────────────────────

def exp4_conflict_only(panels: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """
    IC of agent_signal vs ret, filtered to trades where agent_signal_agreement
    == "conflicts". Per the agent prompt, conflicts + clear headline = high
    conviction. Tests whether this reasoning produces better IC.
    Also compares confirms vs conflicts side-by-side.
    """
    rows = []

    for agreement in ["conflicts", "confirms", "neutral"]:
        pooled_sig: list[pd.Series] = []
        pooled_ret: dict[int, list[pd.Series]] = {h: [] for h in [1, 5, 15]}

        for ticker, df in panels.items():
            sub = df[(df["agent_signal_agreement"] == agreement) &
                     (df["agent_signal"] != 0)]
            if len(sub) < 5:
                continue
            for h in [1, 5, 15]:
                row = _ic_row(sub["agent_signal"], sub[f"ret_{h}m"],
                              label=f"agree={agreement}", group=f"{ticker}_h{h}")
                row["agreement"] = agreement
                row["horizon"] = h
                row["ticker"] = ticker
                rows.append(row)

            pooled_sig.append(sub["agent_signal"])
            for h in [1, 5, 15]:
                pooled_ret[h].append(sub[f"ret_{h}m"])

        if pooled_sig:
            ps = pd.concat(pooled_sig)
            for h in [1, 5, 15]:
                pr = pd.concat(pooled_ret[h]) if pooled_ret[h] else pd.Series(dtype=float)
                row = _ic_row(ps, pr, label=f"agree={agreement}", group=f"POOLED_h{h}")
                row["agreement"] = agreement
                row["horizon"] = h
                row["ticker"] = "POOLED"
                rows.append(row)

    return _apply_bh(pd.DataFrame(rows))


# ── run_all ────────────────────────────────────────────────────────────────────

def run_all() -> dict[str, pd.DataFrame]:
    print("Loading panels + agent outputs...")
    panels = _load_all()
    tickers_loaded = list(panels.keys())
    total_events = sum(len(v) for v in panels.values())
    print(f"  Loaded {len(tickers_loaded)} tickers, {total_events:,} merged events")

    print("\n[1/4] Agent-chosen horizon...")
    e1 = exp1_agent_chosen_horizon(panels)

    print("[2/4] Relevance stratification...")
    e2 = exp2_relevance_stratified(panels)

    print("[3/4] High-conviction filter...")
    e3 = exp3_high_conviction(panels)

    print("[4/4] Conflict-only trades...")
    e4 = exp4_conflict_only(panels)

    print("Done.\n")
    return {"chosen_horizon": e1, "relevance": e2,
            "high_conviction": e3, "conflict": e4}
