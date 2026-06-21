"""
src/analysis/pnl.py — Simple paper-trading PnL from signal and return.

PnL per event = sign(signal) * target_return  (1-share, long/short, no scaling)

Does NOT import from legacy/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def pnl(
    signal: pd.Series,
    target: pd.Series,
) -> dict:
    """
    Compute paper-trading PnL statistics.

    Parameters
    ----------
    signal : Signal series (can be raw or direction-only; sign is taken).
    target : Forward return series (raw or excess).

    Returns
    -------
    dict with keys:
      total_pnl     : sum of sign(signal) * target across events
      mean_per_trade: mean PnL per event
      hit_rate      : fraction of events where sign(signal) == sign(target), both non-zero
      std_pnl       : std of per-event PnL
      n             : number of events used
    """
    df = pd.DataFrame({"sig": signal, "tgt": target}).dropna()
    if len(df) == 0:
        return {
            "total_pnl": np.nan, "mean_per_trade": np.nan,
            "hit_rate": np.nan, "std_pnl": np.nan, "n": 0,
        }

    direction   = np.sign(df["sig"])
    per_trade   = direction * df["tgt"]

    hits_mask   = (direction != 0) & (np.sign(df["tgt"]) != 0)
    hit_trades  = per_trade[hits_mask]

    return {
        "total_pnl":      float(per_trade.sum()),
        "mean_per_trade": float(per_trade.mean()),
        "hit_rate":       float((hit_trades > 0).sum() / max(len(hit_trades), 1)),
        "std_pnl":        float(per_trade.std()),
        "n":              int(len(df)),
    }


def pnl_table(
    panel: pd.DataFrame,
    signals: list[str],
    targets: list[str],
) -> pd.DataFrame:
    """
    Compute PnL stats for every (signal, target) pair.

    Returns DataFrame with columns:
      signal, target, total_pnl, mean_per_trade, hit_rate, std_pnl, n
    """
    rows = []
    for sig in signals:
        for tgt in targets:
            if sig not in panel.columns or tgt not in panel.columns:
                continue
            stats = pnl(panel[sig], panel[tgt])
            rows.append({"signal": sig, "target": tgt, **stats})
    return pd.DataFrame(rows).sort_values(["signal", "target"]).reset_index(drop=True)
