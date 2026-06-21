"""
src/analysis/compare.py — Headline comparison table: agent_signal vs all fixed rules.

Produces a side-by-side comparison per (signal, ticker, horizon, return_type) cell:
  IC + CI, PnL stats (total, mean/trade, hit rate), BH-corrected rejection flag.

BH correction is applied ONCE across the entire grid (all signals × tickers × horizons
× return types) per call to build_comparison_table().

Does NOT import from legacy/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.stats_rigor import bootstrap_ic_ci, benjamini_hochberg

from src.analysis.pnl import pnl
from src.signals.rules import RULE_SIGNALS


_ALL_SIGNALS = ["agent_signal"] + RULE_SIGNALS


def _ic_pnl_row(
    sig_series: pd.Series,
    ret_series: pd.Series,
    signal: str,
    ticker: str,
    horizon: int,
    ret_type: str,
    n_boot: int = 1000,
    random_state: int = 42,
) -> dict:
    """Compute one (signal, ticker, horizon, ret_type) row."""
    sub = pd.DataFrame({"sig": sig_series, "ret": ret_series}).dropna()
    n   = len(sub)

    row: dict = {
        "signal":   signal,
        "ticker":   ticker,
        "horizon":  horizon,
        "ret_type": ret_type,
        "n":        n,
    }

    if n < 10:
        row.update({
            "ic": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "p": np.nan,
            "total_pnl": np.nan, "mean_per_trade": np.nan,
            "hit_rate": np.nan, "std_pnl": np.nan,
        })
        return row

    ic_result = bootstrap_ic_ci(
        sub["sig"], sub["ret"], n_boot=n_boot, seed=random_state
    )
    pnl_stats = pnl(sub["sig"], sub["ret"])
    row.update({
        "ic":             ic_result["ic"],
        "ci_lo":          ic_result["ci_lo"],
        "ci_hi":          ic_result["ci_hi"],
        "p":              ic_result["p_boot"],
        "total_pnl":      pnl_stats["total_pnl"],
        "mean_per_trade": pnl_stats["mean_per_trade"],
        "hit_rate":       pnl_stats["hit_rate"],
        "std_pnl":        pnl_stats["std_pnl"],
    })
    return row


def build_comparison_table(
    panels: dict[str, pd.DataFrame],
    horizons: list[int],
    return_types: list[str] = ("raw",),
    signals: list[str] | None = None,
    n_boot: int = 1000,
    q: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Build the headline comparison table across all signals, tickers, horizons.

    Parameters
    ----------
    panels       : Dict mapping ticker → per-event panel DataFrame.
    horizons     : Forward-return horizons in minutes.
    return_types : "raw" uses ret_{h}m; "excess" uses ret_{h}m_excess.
                   Pass ("raw",) or ("excess",) or ("raw", "excess") for both.
    signals      : Signals to evaluate. Defaults to agent_signal + all rule signals.
    n_boot       : Bootstrap replications for IC CIs.
    q            : BH FDR threshold applied across the full grid.
    random_state : Random seed.

    Returns
    -------
    DataFrame with columns:
      signal, ticker, horizon, ret_type, n,
      ic, ci_lo, ci_hi, p, bh_reject,
      total_pnl, mean_per_trade, hit_rate, std_pnl
    """
    if signals is None:
        signals = _ALL_SIGNALS

    rows = []
    for ticker, panel in panels.items():
        for sig in signals:
            if sig not in panel.columns:
                continue
            for h in horizons:
                for rt in return_types:
                    ret_col = f"ret_{h}m" if rt == "raw" else f"ret_{h}m_excess"
                    if ret_col not in panel.columns:
                        continue
                    row = _ic_pnl_row(
                        panel[sig], panel[ret_col],
                        signal=sig, ticker=ticker,
                        horizon=h, ret_type=rt,
                        n_boot=n_boot, random_state=random_state,
                    )
                    rows.append(row)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        return df

    # BH correction once across the full grid
    valid_p  = df["p"].notna()
    reject   = pd.Series(False, index=df.index)
    if valid_p.any():
        reject_arr, _ = benjamini_hochberg(df.loc[valid_p, "p"].values, q=q)
        reject[valid_p] = reject_arr
    df["bh_reject"] = reject.astype(bool)

    col_order = [
        "signal", "ticker", "horizon", "ret_type", "n",
        "ic", "ci_lo", "ci_hi", "p", "bh_reject",
        "total_pnl", "mean_per_trade", "hit_rate", "std_pnl",
    ]
    return df[[c for c in col_order if c in df.columns]].sort_values(
        ["signal", "ticker", "horizon", "ret_type"]
    ).reset_index(drop=True)
