"""
src/analysis/rigor_fixes.py — Three reviewer-mandated rigor fixes.

Fix 1: Market-excess returns
    Subtract SPY forward return (matched by date+minute) from each stock
    return before computing IC. Separates LLM directional skill from market
    beta in a +18% trending window.

Fix 2: BH across the full subgroup grid
    Pool all p-values from event-type table AND relevance table into a single
    BH correction. Reports which cells survive — likely fewer than before.

Fix 3: LM-on-traded-subset
    For every subgroup where we report agent IC on traded events, also report
    LM IC on that same traded subset. Fair head-to-head.

Usage:
    python -m src.analysis.rigor_fixes
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import AGENT_DIR, PANEL_DIR, HORIZONS
from src.stats_rigor import bootstrap_ic_ci, benjamini_hochberg
from src.signals.rules import add_rule_signals
from src.analysis.earnings_flag import add_earnings_flag, classify_headline

TICKERS = ["AAPL", "NVDA", "AMD", "META", "TSLA", "PEP", "GILD", "HON", "SOFI", "CELH"]
N_BOOT = 1000
SEED   = 42
OUT    = Path("results_final/tables")
OUT.mkdir(parents=True, exist_ok=True)


# ── Data loading ───────────────────────────────────────────────────────────────

def _load_panel(ticker: str) -> pd.DataFrame:
    df = pd.read_csv(PANEL_DIR / f"{ticker}.csv")
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = add_rule_signals(df)
    df = add_earnings_flag(df)
    return df


def _load_agent(ticker: str) -> pd.DataFrame:
    path = AGENT_DIR / f"{ticker}.jsonl"
    rows = [json.loads(l) for l in open(path) if l.strip()]
    return pd.DataFrame(rows)


def _load_spy() -> pd.DataFrame:
    spy_path = Path("data/clean10_2025/market/SPY.csv")
    spy = pd.read_csv(spy_path)
    # columns: minute (int, mins since midnight), fwd_ret_1min/5min/15min, date (YYYYMMDD int)
    spy["date_int"] = spy["date"].astype(int)
    spy["minute_int"] = spy["minute"].astype(int)
    return spy


def _merge_all() -> pd.DataFrame:
    """Load all panels merged with agent scores."""
    frames = []
    for t in TICKERS:
        try:
            panel = _load_panel(t)
            agent = _load_agent(t)
            df = panel.merge(agent, on="event_id", how="inner")
            df["ticker"] = t
            frames.append(df)
        except FileNotFoundError:
            pass
    return pd.concat(frames, ignore_index=True)


def _add_spy_returns(df: pd.DataFrame, spy: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract SPY forward return from each event's stock return.
    Produces excess_ret_1m, excess_ret_5m, excess_ret_15m columns.
    Unmatched events (no SPY bar for that minute) get NaN excess returns.
    """
    df = df.copy()
    # Build lookup key: date as YYYYMMDD int + minute of day as int
    df["_date_int"]   = pd.to_datetime(df["timestamp"]).dt.strftime("%Y%m%d").astype(int)
    df["_minute_int"] = (pd.to_datetime(df["timestamp"]).dt.hour * 60
                         + pd.to_datetime(df["timestamp"]).dt.minute)

    spy_idx = spy.set_index(["date_int", "minute_int"])

    horizon_map = {1: "fwd_ret_1min", 5: "fwd_ret_5min", 15: "fwd_ret_15min"}
    for h, spy_col in horizon_map.items():
        stock_col  = f"ret_{h}m"
        excess_col = f"excess_ret_{h}m"
        spy_ret = df.apply(
            lambda r: spy_idx.at[(r["_date_int"], r["_minute_int"]), spy_col]
            if (r["_date_int"], r["_minute_int"]) in spy_idx.index else np.nan,
            axis=1,
        )
        df[excess_col] = df[stock_col] - spy_ret

    df = df.drop(columns=["_date_int", "_minute_int"])
    return df


def _ic(sig: pd.Series, ret: pd.Series) -> dict:
    sub = pd.DataFrame({"s": sig, "r": ret}).dropna()
    n   = len(sub)
    if n < 10:
        return {"ic": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan, "n": n}
    res = bootstrap_ic_ci(sub["s"], sub["r"], n_boot=N_BOOT, seed=SEED)
    return {"ic": res["ic"], "ci_lo": res["ci_lo"], "ci_hi": res["ci_hi"],
            "p": res["p_boot"], "n": n}


def _stars(p):
    if pd.isna(p): return ""
    if p < 0.01:   return "**"
    if p < 0.05:   return "*"
    if p < 0.10:   return "."
    return ""


# ══════════════════════════════════════════════════════════════════════════════
# FIX 1: Market-excess returns
# ══════════════════════════════════════════════════════════════════════════════

def fix1_excess_returns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Recompute the primary IC table using SPY-excess returns.
    Returns a summary DataFrame for printing and paper updating.
    """
    signals  = ["ofi_z", "lm_score", "llm_score"]
    sig_lbls = {"ofi_z": "ofi", "lm_score": "lm", "llm_score": "llm"}
    horizons = [1, 5, 15]
    rows = []

    for sig, lbl in sig_lbls.items():
        for h in horizons:
            raw_col    = f"ret_{h}m"
            excess_col = f"excess_ret_{h}m"

            raw_res    = _ic(df[sig], df[raw_col])
            excess_res = _ic(df[sig], df[excess_col])

            rows.append({
                "signal":   lbl,
                "horizon":  h,
                "ic_raw":      raw_res["ic"],
                "ic_excess":   excess_res["ic"],
                "p_raw":       raw_res["p"],
                "p_excess":    excess_res["p"],
                "n_raw":       raw_res["n"],
                "n_excess":    excess_res["n"],
            })

    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════════════════
# FIX 2 + FIX 3: Event-type table with pooled BH and LM-on-traded
# ══════════════════════════════════════════════════════════════════════════════

EVENT_TYPES = ["earnings", "analyst", "ma", "legal", "other"]
# map kw_event_type values to display names
_ET_DISPLAY = {"earnings": "earnings", "analyst": "analyst",
               "ma": "M&A", "legal": "legal", "other": "other"}

RELEVANCE_CELLS = [
    # (ticker, relevance, horizon) — the key cells from the paper
    ("TSLA",   "sector", 1),
    ("TSLA",   "sector", 15),
    ("NVDA",   "direct", 1),
    ("POOLED", "sector", 1),
    ("POOLED", "direct", 1),
]


def _event_type_rows(df: pd.DataFrame) -> list[dict]:
    """
    For each event type × horizon, compute:
      - agent IC on traded events only
      - LM IC on ALL events of that type (original comparison)
      - LM IC on TRADED events only (fair head-to-head, Fix 3)
    Returns list of row dicts.
    """
    rows = []
    for et in EVENT_TYPES:
        mask_all    = df["kw_event_type"] == et
        mask_traded = mask_all & (df["agent_signal"] != 0)
        sub_all     = df[mask_all]
        sub_traded  = df[mask_traded]

        n_all    = int(mask_all.sum())
        n_traded = int(mask_traded.sum())

        for h in [1, 15]:
            ret_col = f"ret_{h}m"
            exc_col = f"excess_ret_{h}m"

            agent_res   = _ic(sub_traded["agent_signal"], sub_traded[ret_col])
            lm_all_res  = _ic(sub_all["lm_score"],        sub_all[ret_col])
            lm_trd_res  = _ic(sub_traded["lm_score"],     sub_traded[ret_col])
            agent_exc   = _ic(sub_traded["agent_signal"], sub_traded[exc_col])
            lm_exc      = _ic(sub_traded["lm_score"],     sub_traded[exc_col])

            rows.append({
                "table":        "event_type",
                "event_type":   et,
                "horizon":      h,
                "n_all":        n_all,
                "n_traded":     n_traded,
                # agent
                "agent_ic":     agent_res["ic"],
                "agent_ci_lo":  agent_res["ci_lo"],
                "agent_ci_hi":  agent_res["ci_hi"],
                "agent_p":      agent_res["p"],
                # lm on all (original)
                "lm_all_ic":    lm_all_res["ic"],
                "lm_all_p":     lm_all_res["p"],
                # lm on traded (fair fix)
                "lm_trd_ic":    lm_trd_res["ic"],
                "lm_trd_ci_lo": lm_trd_res["ci_lo"],
                "lm_trd_ci_hi": lm_trd_res["ci_hi"],
                "lm_trd_p":     lm_trd_res["p"],
                # excess returns
                "agent_exc_ic": agent_exc["ic"],
                "agent_exc_p":  agent_exc["p"],
                "lm_exc_ic":    lm_exc["ic"],
                "lm_exc_p":     lm_exc["p"],
            })
    return rows


def _relevance_rows(panels_by_ticker: dict[str, pd.DataFrame]) -> list[dict]:
    """
    Key relevance cells: TSLA sector, NVDA direct, POOLED sector.
    Also adds LM-on-traded for the same cells.
    """
    rows = []
    pooled_by_rel: dict[str, list] = {}

    for ticker, df in panels_by_ticker.items():
        for rel in ["direct", "sector", "macro"]:
            mask    = (df["agent_relevance"] == rel) & (df["agent_signal"] != 0)
            sub     = df[mask]
            if rel not in pooled_by_rel:
                pooled_by_rel[rel] = []
            if len(sub) >= 5:
                pooled_by_rel[rel].append(sub)

            for h in [1, 15]:
                if len(sub) < 10:
                    continue
                ret_col = f"ret_{h}m"
                exc_col = f"excess_ret_{h}m"
                agent_res = _ic(sub["agent_signal"], sub[ret_col])
                lm_res    = _ic(sub["lm_score"],     sub[ret_col])
                agent_exc = _ic(sub["agent_signal"], sub[exc_col])
                lm_exc    = _ic(sub["lm_score"],     sub[exc_col])
                rows.append({
                    "table":       "relevance",
                    "ticker":      ticker,
                    "relevance":   rel,
                    "horizon":     h,
                    "n":           len(sub),
                    "agent_ic":    agent_res["ic"],
                    "agent_ci_lo": agent_res["ci_lo"],
                    "agent_ci_hi": agent_res["ci_hi"],
                    "agent_p":     agent_res["p"],
                    "lm_trd_ic":   lm_res["ic"],
                    "lm_trd_p":    lm_res["p"],
                    "agent_exc_ic":agent_exc["ic"],
                    "agent_exc_p": agent_exc["p"],
                    "lm_exc_ic":   lm_exc["ic"],
                    "lm_exc_p":    lm_exc["p"],
                })

    # POOLED relevance rows
    for rel, subs in pooled_by_rel.items():
        pool = pd.concat(subs, ignore_index=True) if subs else pd.DataFrame()
        if len(pool) < 10:
            continue
        for h in [1, 15]:
            ret_col = f"ret_{h}m"
            exc_col = f"excess_ret_{h}m"
            agent_res = _ic(pool["agent_signal"], pool[ret_col])
            lm_res    = _ic(pool["lm_score"],     pool[ret_col])
            agent_exc = _ic(pool["agent_signal"], pool[exc_col])
            lm_exc    = _ic(pool["lm_score"],     pool[exc_col])
            rows.append({
                "table":       "relevance",
                "ticker":      "POOLED",
                "relevance":   rel,
                "horizon":     h,
                "n":           len(pool),
                "agent_ic":    agent_res["ic"],
                "agent_ci_lo": agent_res["ci_lo"],
                "agent_ci_hi": agent_res["ci_hi"],
                "agent_p":     agent_res["p"],
                "lm_trd_ic":   lm_res["ic"],
                "lm_trd_p":    lm_res["p"],
                "agent_exc_ic":agent_exc["ic"],
                "agent_exc_p": agent_exc["p"],
                "lm_exc_ic":   lm_exc["ic"],
                "lm_exc_p":    lm_exc["p"],
            })
    return rows


def fix2_and_fix3(df_pool: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Run event-type and relevance analyses, apply a single pooled BH correction
    across ALL cells from both tables, return (event_type_df, relevance_df, all_cells_df).
    """
    # split by ticker for relevance analysis
    panels_by_ticker = {t: df_pool[df_pool["ticker"] == t].copy()
                        for t in TICKERS if (df_pool["ticker"] == t).any()}

    et_rows  = _event_type_rows(df_pool)
    rel_rows = _relevance_rows(panels_by_ticker)

    # collect ALL p-values for pooled BH
    # we correct agent_p, lm_trd_p across both tables
    all_rows = et_rows + rel_rows
    all_df   = pd.DataFrame(all_rows)

    # gather p-values: agent_p and lm_trd_p from every row
    agent_ps  = all_df["agent_p"].values
    lm_trd_ps = all_df["lm_trd_p"].values

    # stack into one array, apply BH, split back
    combined = np.concatenate([agent_ps, lm_trd_ps])
    valid    = ~np.isnan(combined)
    reject   = np.zeros(len(combined), dtype=bool)
    if valid.any():
        rej_arr, _ = benjamini_hochberg(combined[valid], q=0.05)
        reject[valid] = rej_arr

    n = len(all_df)
    all_df["agent_bh"]  = reject[:n]
    all_df["lm_trd_bh"] = reject[n:]

    et_df  = all_df[all_df["table"] == "event_type"].copy()
    rel_df = all_df[all_df["table"] == "relevance"].copy()

    return et_df, rel_df, all_df


# ══════════════════════════════════════════════════════════════════════════════
# Print results
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(ic, p=None, bh=None):
    if pd.isna(ic):
        return "  ---  "
    s = f"{ic:+.3f}"
    if p is not None and not pd.isna(p):
        star = _stars(p)
        s += f"{'†' if bh else star}"
    return s


def print_fix1(df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("FIX 1: MARKET-EXCESS RETURNS vs RAW RETURNS")
    print("Subtract SPY forward return; separates directional skill from beta")
    print("-" * 72)
    print(f"{'Signal':<8} {'H':>4}  {'Raw IC':>8} {'p_raw':>7}  {'Exc IC':>8} {'p_exc':>7}  {'Δ IC':>7}")
    print("-" * 72)
    for _, r in df.iterrows():
        delta = (r["ic_excess"] - r["ic_raw"]) if not pd.isna(r["ic_excess"]) else np.nan
        print(f"{r['signal']:<8} {r['horizon']:>3}m  "
              f"{r['ic_raw']:+.4f} {r['p_raw']:>7.3f}  "
              f"{r['ic_excess']:+.4f} {r['p_excess']:>7.3f}  "
              f"{delta:+.4f}")
    n_spy = int(df["n_excess"].iloc[0]) if not df.empty else 0
    pct_matched = n_spy / max(int(df["n_raw"].iloc[0]), 1) * 100 if not df.empty else 0
    print(f"\n  SPY-matched events: {n_spy} ({pct_matched:.1f}% of panel)")


def print_fix2_fix3(et_df: pd.DataFrame, rel_df: pd.DataFrame):
    print("\n" + "=" * 72)
    print("FIX 2: POOLED BH ACROSS ALL SUBGROUP TESTS")
    print("FIX 3: LM ON TRADED SUBSET (fair head-to-head)")
    print("† = survives pooled BH (q=0.05 across ALL event-type + relevance cells)")
    print("-" * 72)

    print("\n── EVENT-TYPE TABLE ──")
    print(f"{'Type':<10} {'H':>3} {'n_all':>6} {'n_trd':>6} "
          f"  {'Agent IC':>10} {'BH':>3} "
          f"  {'LM(all) IC':>10} "
          f"  {'LM(traded) IC':>13} {'BH':>3}  Winner")
    print("-" * 72)
    for _, r in et_df.sort_values(["event_type","horizon"]).iterrows():
        agent_s   = _fmt(r["agent_ic"],  r["agent_p"],   r["agent_bh"])
        lm_all_s  = _fmt(r["lm_all_ic"], r["lm_all_p"])
        lm_trd_s  = _fmt(r["lm_trd_ic"], r["lm_trd_p"],  r["lm_trd_bh"])
        # winner based on agent_ic vs lm_trd_ic (fair comparison)
        if pd.isna(r["agent_ic"]) or pd.isna(r["lm_trd_ic"]):
            winner = "?"
        else:
            winner = "Agent" if r["agent_ic"] > r["lm_trd_ic"] else "LM"
        print(f"{r['event_type']:<10} {r['horizon']:>3}m {int(r['n_all']):>6} {int(r['n_traded']):>6} "
              f"  {agent_s:>10} {'†' if r['agent_bh'] else ' ':>3} "
              f"  {lm_all_s:>10} "
              f"  {lm_trd_s:>13} {'†' if r['lm_trd_bh'] else ' ':>3}  {winner}")

    print("\n── RELEVANCE TABLE (key cells) ──")
    print(f"{'Ticker':<8} {'Rel':<8} {'H':>3} {'n':>5} "
          f"  {'Agent IC':>10} {'BH':>3} "
          f"  {'LM(trd) IC':>10} {'BH':>3}")
    print("-" * 72)
    show_tickers = ["TSLA", "NVDA", "POOLED"]
    show_rels    = ["sector", "direct"]
    sub = rel_df[rel_df["ticker"].isin(show_tickers) & rel_df["relevance"].isin(show_rels)]
    for _, r in sub.sort_values(["relevance","ticker","horizon"]).iterrows():
        agent_s  = _fmt(r["agent_ic"],   r["agent_p"],   r["agent_bh"])
        lm_trd_s = _fmt(r["lm_trd_ic"],  r["lm_trd_p"],  r["lm_trd_bh"])
        print(f"{r['ticker']:<8} {r['relevance']:<8} {r['horizon']:>3}m {int(r['n']):>5} "
              f"  {agent_s:>10} {'†' if r['agent_bh'] else ' ':>3} "
              f"  {lm_trd_s:>10} {'†' if r['lm_trd_bh'] else ' ':>3}")

    print("\n── EXCESS RETURNS (agent + LM on traded, SPY-adjusted) ──")
    print(f"{'Type':<10} {'H':>3} {'AgExc IC':>10} {'LMExc IC':>10}")
    print("-" * 50)
    for _, r in et_df.sort_values(["event_type","horizon"]).iterrows():
        print(f"{r['event_type']:<10} {r['horizon']:>3}m "
              f"{_fmt(r['agent_exc_ic'], r['agent_exc_p']):>10} "
              f"{_fmt(r['lm_exc_ic'],    r['lm_exc_p']):>10}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def run():
    print("Loading panels + agent scores...")
    df_pool = _merge_all()
    print(f"  {len(df_pool):,} total events across {df_pool['ticker'].nunique()} tickers")

    print("\nLoading SPY and computing excess returns...")
    spy = _load_spy()
    df_pool = _add_spy_returns(df_pool, spy)
    n_spy = df_pool["excess_ret_1m"].notna().sum()
    print(f"  SPY matched: {n_spy:,} / {len(df_pool):,} events ({n_spy/len(df_pool)*100:.1f}%)")

    print("\n[Fix 1] Computing raw vs excess IC...")
    fix1_df = fix1_excess_returns(df_pool)
    print_fix1(fix1_df)
    fix1_df.to_csv(OUT / "fix1_excess_returns.csv", index=False)

    print("\n[Fix 2+3] Event-type + relevance with pooled BH and LM-on-traded...")
    et_df, rel_df, all_df = fix2_and_fix3(df_pool)
    print_fix2_fix3(et_df, rel_df)
    et_df.to_csv(OUT / "fix2_eventtype_bh.csv",  index=False)
    rel_df.to_csv(OUT / "fix2_relevance_bh.csv", index=False)
    all_df.to_csv(OUT / "fix2_all_cells.csv",    index=False)

    print(f"\nAll results saved to {OUT}/")
    return fix1_df, et_df, rel_df


if __name__ == "__main__":
    run()
