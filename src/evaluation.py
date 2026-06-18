"""
evaluation.py — IC evaluation, stratified analysis, and reporting for LNA-Agent.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

RESULTS       = Path("results")
AGENT_RESULTS = RESULTS / "agent"
HORIZONS      = [1, 5, 15, 30, 60]


# ── IC helper ─────────────────────────────────────────────────────────────────

def _ic(signal: pd.Series, returns: pd.Series) -> tuple[float, float]:
    """Spearman IC and p-value; NaN if insufficient data."""
    mask = signal.notna() & returns.notna()
    if mask.sum() < 10:
        return np.nan, np.nan
    ic, pval = spearmanr(signal[mask], returns[mask])
    return float(ic), float(pval)


# ── Main IC table ─────────────────────────────────────────────────────────────

def compute_comparison(ticked_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build IC comparison table for one ticker.

    Evaluates agent_signal and constituent signals (LM, LLM, OFI) against
    all available return horizons (1/5/15/30/60 min if present).

    Returns DataFrame with columns: strategy, horizon, ic, pval, n
    """
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

    # Chosen-horizon metric: evaluate each event against the agent's chosen window
    chosen_returns = []
    chosen_signal  = ticked_df["agent_signal"]
    horizon_to_return = {
        "1min":  "ret_1m",
        "5min":  "ret_5m",
        "15min": "ret_15m",
    }
    for _, row in ticked_df.iterrows():
        h       = str(row.get("agent_horizon", "1min"))
        ret_col = horizon_to_return.get(h, "ret_1m")
        chosen_returns.append(row.get(ret_col, np.nan))
    chosen_returns = pd.Series(chosen_returns, index=ticked_df.index)
    mask   = chosen_signal.notna() & chosen_returns.notna()
    ic, pval = _ic(chosen_signal, chosen_returns)
    rows.append({
        "strategy": "Agent chosen horizon",
        "horizon":  "chosen",
        "ic":       ic,
        "pval":     pval,
        "n":        int(mask.sum()),
    })

    # Bootstrap baseline
    rng      = np.random.default_rng(42)
    boot_ics = []
    ret_cols = [f"ret_{h}m" for h in HORIZONS if f"ret_{h}m" in ticked_df.columns]
    for _ in range(1000):
        random_col = ret_cols[rng.integers(len(ret_cols))]
        shuffled   = ticked_df[random_col].values.copy()
        rng.shuffle(shuffled)
        shuffled = pd.Series(shuffled, index=ticked_df.index)
        b_mask   = chosen_signal.notna() & shuffled.notna()
        if b_mask.sum() >= 10:
            b_ic, _ = spearmanr(chosen_signal[b_mask], shuffled[b_mask])
            boot_ics.append(float(b_ic))
    rows.append({
        "strategy": "Agent chosen horizon (random baseline)",
        "horizon":  "chosen_random",
        "ic":       float(np.mean(boot_ics)) if boot_ics else np.nan,
        "pval":     np.nan,
        "n":        int(mask.sum()),
    })

    # Horizon-group evaluation
    for h_choice in ["1min", "5min", "15min"]:
        subset = ticked_df[ticked_df["agent_horizon"] == h_choice]
        if len(subset) < 10:
            continue
        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in subset.columns:
                continue
            sig_sub  = subset["agent_signal"]
            ret_sub  = subset[ret_col]
            h_ic, h_pval = _ic(sig_sub, ret_sub)
            h_mask   = sig_sub.notna() & ret_sub.notna()
            rows.append({
                "strategy": f"Agent (chose {h_choice})",
                "horizon":  h,
                "ic":       h_ic,
                "pval":     h_pval,
                "n":        int(h_mask.sum()),
            })

    return pd.DataFrame(rows)


# ── Stratified IC ─────────────────────────────────────────────────────────────

def size_stratified_ic(
    ticked_df: pd.DataFrame,
    ticker: str,
    suffix: str,
    mode_tag: str,
) -> pd.DataFrame:
    """IC by agent_size quartile × return horizon."""
    df = ticked_df[ticked_df["agent_direction"] != "none"].copy()
    if len(df) < 20:
        return pd.DataFrame(columns=["quartile", "horizon", "ic", "pval", "n", "mean_size"])

    codes = pd.qcut(df["agent_size"], q=4, labels=False, duplicates="drop")
    df["size_quartile"] = codes.map(lambda c: f"Q{c + 1}")

    rows = []
    for quartile in sorted(df["size_quartile"].unique()):
        sub = df[df["size_quartile"] == quartile]
        for h in HORIZONS:
            ret_col = f"ret_{h}m"
            if ret_col not in sub.columns:
                continue
            sig    = sub["agent_signal"]
            ret    = sub[ret_col]
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


def relevance_stratified_ic(
    ticked_df: pd.DataFrame,
    ticker: str,
    suffix: str,
    mode_tag: str,
) -> pd.DataFrame:
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


def agreement_stratified_ic(
    ticked_df: pd.DataFrame,
    ticker: str,
    suffix: str,
    mode_tag: str,
) -> pd.DataFrame:
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


# ── Cross-asset spillover ─────────────────────────────────────────────────────

def cross_asset_spillover(suffix: str) -> pd.DataFrame:
    """
    Load agent panels for all available tickers and compute cross-ticker IC.

    For each ordered pair (source, target), computes Spearman IC of
    source agent_signal vs target ret_{h}m, using only rows where the
    source ticker had an active trade.  Panels are aligned by their
    DatetimeIndex intersection.

    Saves to results/spillover_{suffix}_llm.csv.
    """
    panels: dict[str, pd.DataFrame] = {}
    for path in sorted(AGENT_RESULTS.glob(f"agent_panel_*_{suffix}_llm.parquet")):
        # agent_panel_{TICKER}_{suffix}_llm.parquet
        stem   = path.stem  # e.g. agent_panel_AMD_2016-01-01_2020-12-31_llm
        prefix = f"agent_panel_"
        tail   = f"_{suffix}_llm"
        ticker = stem[len(prefix):-len(tail)] if stem.startswith(prefix) else None
        if not ticker:
            continue
        try:
            panels[ticker] = pd.read_parquet(path)
        except Exception as e:
            print(f"  [spillover] Could not load {path.name}: {e}")

    if len(panels) < 2:
        print("  [spillover] Need ≥2 ticker panels. Run agent.py first.")
        return pd.DataFrame()

    rows    = []
    tickers = list(panels.keys())
    for src in tickers:
        src_df     = panels[src]
        src_traded = src_df[src_df["agent_direction"] != "none"]
        if len(src_traded) < 10:
            continue
        for tgt in tickers:
            if tgt == src:
                continue
            tgt_df  = panels[tgt]
            common  = src_traded.index.intersection(tgt_df.index)
            if len(common) < 10:
                continue
            src_sig = src_traded.loc[common, "agent_signal"]
            for h in [1, 5, 15]:
                ret_col = f"ret_{h}m"
                if ret_col not in tgt_df.columns:
                    continue
                tgt_ret  = tgt_df.loc[common, ret_col]
                ic, pval = _ic(src_sig, tgt_ret)
                mask     = src_sig.notna() & tgt_ret.notna()
                rows.append({
                    "source_ticker": src,
                    "target_ticker": tgt,
                    "horizon":       h,
                    "ic":            ic,
                    "pval":          pval,
                    "n":             int(mask.sum()),
                })

    df = pd.DataFrame(rows)
    if df.empty:
        print("  [spillover] No aligned cross-ticker rows found.")
        return df

    print(f"\n{'═' * 70}")
    print(f"  CROSS-ASSET SPILLOVER  (suffix={suffix})")
    print(f"{'═' * 70}")
    print(df.to_string(index=False))

    out_path = RESULTS / f"spillover_{suffix}_llm.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved → {out_path}")

    return df


# ── Pretty-print ──────────────────────────────────────────────────────────────

def _print_comparison(ticker: str, comp: pd.DataFrame) -> None:
    strategies = ["Agent", "LM", "LLM", "OFI"]

    # Derive which numeric horizons are actually present in comp
    avail_h = sorted([
        h for h in comp["horizon"].unique()
        if isinstance(h, (int, float, np.integer)) and not pd.isna(h)
    ])

    col_w    = 9
    h_labels = [f"{int(h)}-min" for h in avail_h]

    print(f"\n{'═' * 70}")
    print(f"  IC COMPARISON — {ticker}")
    print(f"{'═' * 70}")
    header = (f"  {'Strategy':<22}  "
              + "  ".join(f"{lbl:>{col_w}}" for lbl in h_labels)
              + f"  {'N':>5}")
    sep    = (f"  {'-'*22}  "
              + "  ".join(f"{'-'*col_w}" for _ in avail_h)
              + f"  {'-'*5}")
    print(header)
    print(sep)

    def _fmt(strat: str, h) -> str:
        sub = comp[(comp["strategy"] == strat) & (comp["horizon"] == h)]
        if sub.empty or pd.isna(sub.iloc[0]["ic"]):
            return f"{'NaN':>{col_w}}"
        v   = sub.iloc[0]["ic"]
        pv  = sub.iloc[0]["pval"]
        sig = ("***" if pv < 0.01 else "**" if pv < 0.05
               else "*" if pv < 0.10 else " ")
        return f"{v:>+{col_w-1}.4f}{sig}"

    for strat in strategies:
        sub = comp[comp["strategy"] == strat]
        if sub.empty:
            continue
        n_  = int(sub.iloc[0]["n"])
        row = f"  {strat:<22}  " + "  ".join(_fmt(strat, h) for h in avail_h)
        print(f"{row}  {n_:>5}")

    chosen = comp[comp["strategy"] == "Agent chosen horizon"]
    if not chosen.empty:
        v  = chosen.iloc[0]["ic"]
        pv = chosen.iloc[0]["pval"]
        n  = int(chosen.iloc[0]["n"])
        if pd.isna(v):
            shown = "NaN"
        else:
            sig   = ("***" if pv < 0.01 else "**" if pv < 0.05
                     else "*" if pv < 0.10 else "")
            shown = f"{v:+.4f}{sig}"
        print(f"  {'Agent chosen horizon':<22}  {shown:>9}  n={n}")

    rand_row = comp[comp["strategy"] == "Agent chosen horizon (random baseline)"]
    if not rand_row.empty:
        r_ic = rand_row.iloc[0]["ic"]
        r_n  = int(rand_row.iloc[0]["n"])
        if not pd.isna(r_ic):
            print(f"  {'Agent chosen (random)':22}  {r_ic:>9.4f}  n={r_n}")
    print()
