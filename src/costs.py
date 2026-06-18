"""
costs.py — Net-of-cost IC and PnL for the confirmed OFI×LLM strategy.
LNA-Agent: Lobster News Alpha

Transaction cost model:
  half_spread_bps = (ask - bid) / 2 / mid * 1e4   from L1 LOB at bar_time
  depth_slippage  = excess shares beyond L1 depth (L1 only; L2 TODO(schema))
  round_trip_bps  = 2 * (half_spread_bps + depth_slippage_bps)

Quotes are attached via join_lob_quotes.join_l1_quotes().
Per-row fallback (5 bps round-trip) applies only when lob_matched is False.

Resolved column mapping (from inspection):
  Signal column    : "ofi_x_llm" = ofi_z * llm_score  (computed on demand)
  Ticker column    : "stock"  (not "ticker")
  Bar timestamp    : "bar_time"
  Return columns   : "ret_1m", "ret_5m", "ret_15m"
  Quote columns    : "ask", "bid", "ask_size", "bid_size", "spread_bps"
                     (added by join_l1_quotes; absent until that step is run)
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# Confirmed signal column: computed as ofi_z * llm_score
_SIGNAL_COL   = "ofi_x_llm"
_TS_COL       = "timestamp"
_BAR_COL      = "bar_time"
_FALLBACK_BPS = 5.0   # round-trip fallback when LOB quote is unavailable

# TODO(schema): LOBSTER L2 depth — load_lob() returns only L1 (ask_size,
#   bid_size).  Multi-level slippage requires the full 10-level book.


def _ensure_signal(df: pd.DataFrame) -> pd.DataFrame:
    """Add ofi_x_llm column if not already present."""
    if _SIGNAL_COL not in df.columns:
        df = df.copy()
        df[_SIGNAL_COL] = df["ofi_z"] * df["llm_score"]
    return df


def _depth_slippage_bps(
    ask: float,
    ask_size: float,
    bid: float,
    bid_size: float,
    direction: float,
    trade_size_shares: float,
) -> float:
    """Linear L1 impact when order size exceeds top-of-book depth (in bps)."""
    mid = (ask + bid) / 2
    if mid <= 0:
        return 0.0
    if direction > 0:
        depth  = max(ask_size, 1.0) if not np.isnan(ask_size) else 1.0
        excess = max(0.0, (trade_size_shares - depth) / depth)
        return excess * (ask - mid) / mid * 1e4
    else:
        depth  = max(bid_size, 1.0) if not np.isnan(bid_size) else 1.0
        excess = max(0.0, (trade_size_shares - depth) / depth)
        return excess * (mid - bid) / mid * 1e4


def cost_at_event(row: pd.Series, trade_size_shares: float = 100.0) -> float:
    """
    Round-trip cost in bps for one event row.

    Uses spread_bps from join_l1_quotes output when lob_matched is True.
    Falls back to _FALLBACK_BPS when the row has no real quote (lob_matched
    is False or missing, or ask/bid are NaN).

    Depth slippage is added on top of the half-spread when lob_matched=True.
    """
    lob_matched = bool(row.get("lob_matched", False))
    ask         = float(row.get("ask",        np.nan))
    bid         = float(row.get("bid",        np.nan))
    ask_size    = float(row.get("ask_size",   np.nan))
    bid_size    = float(row.get("bid_size",   np.nan))
    spread_bps  = float(row.get("spread_bps", np.nan))  # half-spread bps from join

    if not lob_matched or np.isnan(ask) or np.isnan(bid):
        return _FALLBACK_BPS  # round-trip fallback

    signal    = float(row.get(_SIGNAL_COL, 0.0) or 0.0)
    direction = np.sign(signal)

    # spread_bps is the HALF-spread; round-trip = 2 * half + 2 * depth impact
    half_sp_bps = spread_bps if not np.isnan(spread_bps) else (ask - bid) / 2 / ((ask + bid) / 2) * 1e4
    depth_bps   = _depth_slippage_bps(ask, ask_size, bid, bid_size, direction, trade_size_shares)

    return 2.0 * (half_sp_bps + depth_bps)


def _coverage_fraction(df: pd.DataFrame) -> float:
    """Fraction of rows with a real LOB quote (lob_matched=True or ask is non-NaN)."""
    if "lob_matched" in df.columns:
        return float(df["lob_matched"].mean())
    if "ask" in df.columns:
        return float(df["ask"].notna().mean())
    return 0.0


def net_ic(panel: pd.DataFrame, horizon: int) -> dict:
    """
    Spearman IC of OFI×LLM after subtracting round-trip costs from returns.

    panel should have been processed by join_lob_quotes.join_l1_quotes()
    to attach ask/bid/spread_bps.  Rows without a real quote (lob_matched=False)
    incur the _FALLBACK_BPS cost.

    Returns dict: horizon, gross_ic, net_ic, mean_cost_bps, coverage_frac, n.
    """
    ret_col = f"ret_{horizon}m"
    req     = ["ofi_z", "llm_score", ret_col]
    df      = panel.dropna(subset=req).copy()
    df      = _ensure_signal(df)

    gross_ret    = df[ret_col].copy()
    costs_bps    = df.apply(cost_at_event, axis=1)
    coverage     = _coverage_fraction(df)

    # Net return: subtract one-way cost (fraction of round-trip) per event
    net_ret  = gross_ret - costs_bps * 1e-4 * np.sign(df[_SIGNAL_COL])

    signal   = df[_SIGNAL_COL]
    mask_g   = signal.notna() & gross_ret.notna()
    mask_n   = signal.notna() & net_ret.notna()
    gross_ic = float(spearmanr(signal[mask_g], gross_ret[mask_g])[0]) if mask_g.sum() >= 10 else np.nan
    net_ic_v = float(spearmanr(signal[mask_n], net_ret[mask_n])[0])   if mask_n.sum() >= 10 else np.nan

    return {
        "horizon":       horizon,
        "gross_ic":      gross_ic,
        "net_ic":        net_ic_v,
        "mean_cost_bps": float(costs_bps.mean()),
        "coverage_frac": coverage,
        "n":             len(df),
    }


def net_pnl(panel: pd.DataFrame, trade_size_shares: float = 100.0) -> pd.DataFrame:
    """
    Event-level gross and net PnL (bps) for the confirmed OFI×LLM strategy.

    Uses agent_horizon when present; falls back to ret_1m.

    Returns DataFrame with columns:
      timestamp, signal, gross_pnl_bps, cost_bps, net_pnl_bps,
      lob_matched, coverage_frac (scalar repeated for easy aggregation)
    """
    df = panel.copy()
    df = _ensure_signal(df)

    horizon_map = {"1min": "ret_1m", "5min": "ret_5m", "15min": "ret_15m"}
    if "agent_horizon" in df.columns:
        ret_cols = df["agent_horizon"].map(horizon_map).fillna("ret_1m")
    else:
        ret_cols = pd.Series("ret_1m", index=df.index)

    gross_pnl = pd.Series(np.nan, index=df.index)
    for col_name in ["ret_1m", "ret_5m", "ret_15m"]:
        mask = (ret_cols == col_name)
        if mask.any() and col_name in df.columns:
            gross_pnl.loc[mask] = (
                df.loc[mask, col_name] * df.loc[mask, _SIGNAL_COL] * 1e4
            )

    costs_bps    = df.apply(lambda r: cost_at_event(r, trade_size_shares), axis=1)
    coverage     = _coverage_fraction(df)

    return pd.DataFrame({
        "timestamp":     df[_TS_COL] if _TS_COL in df.columns else df.index,
        "signal":        df[_SIGNAL_COL],
        "gross_pnl_bps": gross_pnl,
        "cost_bps":      costs_bps,
        "net_pnl_bps":   gross_pnl - costs_bps * np.sign(df[_SIGNAL_COL]),
        "lob_matched":   df.get("lob_matched", pd.Series(False, index=df.index)),
        "coverage_frac": coverage,
    })


def run_net_of_cost(
    panel: pd.DataFrame,
    horizons: list[int] = [1, 5, 15],
) -> pd.DataFrame:
    """
    Run net_ic for all horizons and print a coverage-annotated summary.

    Caller should pass a panel already processed by join_l1_quotes().
    """
    rows = []
    for h in horizons:
        r = net_ic(panel, h)
        rows.append(r)

    df = pd.DataFrame(rows)
    cov = df["coverage_frac"].iloc[0] if not df.empty else 0.0
    print(f"\n  Net-of-cost summary  (LOB quote coverage: {cov:.1%})")
    print(df[["horizon", "n", "gross_ic", "net_ic",
              "mean_cost_bps", "coverage_frac"]].to_string(index=False))
    return df
