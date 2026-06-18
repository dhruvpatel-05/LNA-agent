"""
costs.py — Net-of-cost IC and PnL for the confirmed OFI×LLM strategy.
LNA-Agent: Lobster News Alpha

Transaction cost model:
  half_spread = (ask - bid) / 2   at the decision bar
  depth_slippage = f(order_size, ask_size/bid_size at L1)
  total_cost_per_trade = half_spread + depth_slippage

Reuses load_lob() from legacy/src/ofi.py for raw book access.
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# Pull the existing LOBSTER loader without reimplementing it
_LEGACY = Path(__file__).parent.parent / "legacy" / "src"
if str(_LEGACY) not in sys.path:
    sys.path.insert(0, str(_LEGACY))

from ofi import load_lob  # type: ignore  # legacy loader


# ── Confirmed strategy: OFI × LLM ─────────────────────────────────────────────
# TODO(schema): verify signal column name once all notebooks are wired
_SIGNAL_COL  = "ofi_x_llm"   # ofi_z * llm_score
_TS_COL      = "timestamp"
_BAR_COL     = "bar_time"

# TODO(schema): LOBSTER L2 depth — current load_lob() returns only L1
#   (ask_size, bid_size).  For depth-aware slippage beyond L1, load the full
#   10-level book and sum cumulative size up to the order size.


def _half_spread_bps(ask: float, bid: float) -> float:
    """Quoted half-spread in basis points."""
    mid = (ask + bid) / 2
    return (ask - bid) / 2 / mid * 10_000 if mid > 0 else 0.0


def _depth_slippage_bps(
    ask: float,
    ask_size: float,
    bid: float,
    bid_size: float,
    direction: float,
    trade_size_shares: float = 100.0,
) -> float:
    """
    Linear price-impact slippage when order exceeds top-of-book depth.

    For a buy (direction > 0): if trade_size > ask_size, assumes price
    moves proportionally. Uses L1 only; L2 depth is TODO(schema).
    """
    mid = (ask + bid) / 2
    if mid <= 0:
        return 0.0

    if direction > 0:
        depth = ask_size if ask_size > 0 else 1.0
        excess_fraction = max(0.0, (trade_size_shares - depth) / depth)
        price_impact = excess_fraction * (ask - mid) / mid * 10_000
    else:
        depth = bid_size if bid_size > 0 else 1.0
        excess_fraction = max(0.0, (trade_size_shares - depth) / depth)
        price_impact = excess_fraction * (mid - bid) / mid * 10_000

    return float(price_impact)


def cost_at_event(row: pd.Series, trade_size_shares: float = 100.0) -> float:
    """
    Total round-trip cost in bps for one event row.

    Requires columns: ask, bid, ask_size, bid_size from the LOB snapshot.
    TODO(schema): these columns come from joining the raw LOB at bar_time;
    not present in the aligned panel directly.
    """
    ask      = float(row.get("ask",      np.nan))
    bid      = float(row.get("bid",      np.nan))
    ask_size = float(row.get("ask_size", np.nan))
    bid_size = float(row.get("bid_size", np.nan))
    signal   = float(row.get(_SIGNAL_COL, 0.0) or 0.0)

    if np.isnan(ask) or np.isnan(bid):
        return np.nan

    direction  = np.sign(signal)
    half_sp    = _half_spread_bps(ask, bid)
    depth_slip = _depth_slippage_bps(ask, ask_size, bid, bid_size,
                                     direction, trade_size_shares)
    return 2.0 * (half_sp + depth_slip)  # round-trip


def net_ic(panel: pd.DataFrame, horizon: int) -> dict:
    """
    Spearman IC of OFI×LLM after subtracting round-trip cost from returns.

    panel must have columns: ofi_z, llm_score, ret_{horizon}m.
    If ask/bid columns are absent, cost is estimated from a
    TODO(schema) fallback using the spread stored in the OFI aggregate CSV.

    Returns dict: gross_ic, net_ic, mean_cost_bps, n.
    """
    ret_col = f"ret_{horizon}m"
    req     = ["ofi_z", "llm_score", ret_col]
    df      = panel.dropna(subset=req).copy()

    if _SIGNAL_COL not in df.columns:
        df[_SIGNAL_COL] = df["ofi_z"] * df["llm_score"]

    gross_ret = df[ret_col].copy()

    # Subtract cost if book data is attached; else TODO(schema) fallback
    if {"ask", "bid"}.issubset(df.columns):
        costs_bps = df.apply(cost_at_event, axis=1)
    else:
        # TODO(schema): join raw LOB ask/bid at bar_time to get actual spread.
        # Temporary fallback: use a fixed 5 bps round-trip estimate.
        costs_bps = pd.Series(5.0, index=df.index)

    # Convert bps to return units (1 bps = 0.0001)
    net_ret   = gross_ret - costs_bps * 1e-4 * np.sign(df[_SIGNAL_COL])

    n         = len(df)
    signal    = df[_SIGNAL_COL]
    mask      = signal.notna() & gross_ret.notna()
    gross_ic  = float(spearmanr(signal[mask], gross_ret[mask])[0]) if mask.sum() >= 10 else np.nan
    net_mask  = signal.notna() & net_ret.notna()
    net_ic_v  = float(spearmanr(signal[net_mask], net_ret[net_mask])[0]) if net_mask.sum() >= 10 else np.nan

    return {
        "horizon":       horizon,
        "gross_ic":      gross_ic,
        "net_ic":        net_ic_v,
        "mean_cost_bps": float(costs_bps.mean()),
        "n":             n,
    }


def net_pnl(panel: pd.DataFrame, trade_size_shares: float = 100.0) -> pd.DataFrame:
    """
    Event-level gross and net PnL (bps) for the confirmed strategy.

    Uses agent_horizon when present; falls back to ret_1m.
    Returns DataFrame with columns: timestamp, signal, gross_pnl_bps,
    cost_bps, net_pnl_bps.
    """
    df = panel.copy()
    if _SIGNAL_COL not in df.columns:
        df[_SIGNAL_COL] = df.get("ofi_z", 0.0) * df.get("llm_score", 0.0)

    horizon_map = {"1min": "ret_1m", "5min": "ret_5m", "15min": "ret_15m"}
    if "agent_horizon" in df.columns:
        ret_col = df["agent_horizon"].map(horizon_map).fillna("ret_1m")
    else:
        ret_col = pd.Series("ret_1m", index=df.index)

    gross_pnl = pd.Series(index=df.index, dtype=float)
    for col_name in ["ret_1m", "ret_5m", "ret_15m"]:
        mask = (ret_col == col_name) & (col_name in df.columns)
        if mask.any():
            gross_pnl.loc[mask] = df.loc[mask, col_name] * df.loc[mask, _SIGNAL_COL] * 10_000

    if {"ask", "bid"}.issubset(df.columns):
        costs_bps = df.apply(lambda r: cost_at_event(r, trade_size_shares), axis=1)
    else:
        # TODO(schema): join LOB book columns before calling net_pnl.
        costs_bps = pd.Series(5.0, index=df.index)

    ts_col = _TS_COL if _TS_COL in df.columns else df.index

    return pd.DataFrame({
        "timestamp":    df[_TS_COL] if _TS_COL in df.columns else df.index,
        "signal":       df[_SIGNAL_COL],
        "gross_pnl_bps": gross_pnl,
        "cost_bps":     costs_bps,
        "net_pnl_bps":  gross_pnl - costs_bps * np.sign(df[_SIGNAL_COL]),
    })
