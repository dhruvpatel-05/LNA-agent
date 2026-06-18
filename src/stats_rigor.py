"""
stats_rigor.py — Bootstrap IC inference, FDR correction, IC decay curve.
LNA-Agent: Lobster News Alpha
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def spearman_ic(signal: pd.Series, ret: pd.Series) -> float:
    """Spearman IC between signal and returns, dropping NaNs. Returns NaN if n < 10."""
    mask = signal.notna() & ret.notna()
    if mask.sum() < 10:
        return float("nan")
    ic, _ = spearmanr(signal[mask], ret[mask])
    return float(ic)


def bootstrap_ic_ci(
    signal: pd.Series,
    ret: pd.Series,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Percentile bootstrap confidence interval for Spearman IC over events.

    Returns dict: ic, ci_lo, ci_hi, p_boot (fraction of boots ≤ 0), n.
    """
    mask = signal.notna() & ret.notna()
    x = signal[mask].values
    y = ret[mask].values
    n = len(x)

    ic_obs = float(spearmanr(x, y)[0]) if n >= 10 else float("nan")

    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i], _ = spearmanr(x[idx], y[idx])

    ci_lo = float(np.percentile(boots, 100 * alpha / 2))
    ci_hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    p_boot = float(np.mean(boots <= 0))

    return {"ic": ic_obs, "ci_lo": ci_lo, "ci_hi": ci_hi, "p_boot": p_boot, "n": n}


def paired_ic_diff(
    signal_a: pd.Series,
    signal_b: pd.Series,
    ret: pd.Series,
    n_boot: int = 5000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict:
    """
    Bootstrap the IC drop between two signals on the same events.

    Both signals must be available for the same rows (inner join on index).
    Returns dict: ic_a, ic_b, diff (ic_a - ic_b), ci_lo, ci_hi, p_boot, n.
    p_boot = fraction of bootstrap replicates where diff ≤ 0.
    """
    df = pd.DataFrame({"a": signal_a, "b": signal_b, "r": ret}).dropna()
    n  = len(df)

    ic_a = float(spearmanr(df["a"], df["r"])[0]) if n >= 10 else float("nan")
    ic_b = float(spearmanr(df["b"], df["r"])[0]) if n >= 10 else float("nan")
    diff_obs = ic_a - ic_b

    x_a = df["a"].values
    x_b = df["b"].values
    y   = df["r"].values

    rng = np.random.default_rng(seed)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        a_i, _ = spearmanr(x_a[idx], y[idx])
        b_i, _ = spearmanr(x_b[idx], y[idx])
        diffs[i] = a_i - b_i

    ci_lo  = float(np.percentile(diffs, 100 * alpha / 2))
    ci_hi  = float(np.percentile(diffs, 100 * (1 - alpha / 2)))
    p_boot = float(np.mean(diffs <= 0))

    return {
        "ic_a":   ic_a,
        "ic_b":   ic_b,
        "diff":   diff_obs,
        "ci_lo":  ci_lo,
        "ci_hi":  ci_hi,
        "p_boot": p_boot,
        "n":      n,
    }


def benjamini_hochberg(pvals: np.ndarray, q: float = 0.05):
    """
    Benjamini-Hochberg FDR correction.

    Returns (reject: bool array, adjusted_pvals: float array).
    """
    pvals = np.asarray(pvals, dtype=float)
    m     = len(pvals)
    order = np.argsort(pvals)
    ranks = np.empty(m, dtype=int)
    ranks[order] = np.arange(1, m + 1)

    adj    = np.minimum(1.0, pvals * m / ranks)
    # Enforce monotonicity: adj[i] = min(adj[i:])
    adj    = np.minimum.accumulate(adj[order][::-1])[::-1]
    adj_sorted = adj.copy()
    adj[order] = adj_sorted  # map back to original order

    reject = adj <= q
    return reject, adj


def fdr_table(df: pd.DataFrame, p_col: str = "p", q: float = 0.05) -> pd.DataFrame:
    """
    Append BH-adjusted p-values and reject_fdr column, sorted ascending by p_adj.
    """
    df = df.copy()
    reject, adj = benjamini_hochberg(df[p_col].values, q=q)
    df["p_adj"]      = adj
    df["reject_fdr"] = reject
    return df.sort_values("p_adj").reset_index(drop=True)


def ic_decay(
    signal: pd.Series,
    returns_by_minute: pd.DataFrame,
    horizons=range(1, 31),
) -> pd.DataFrame:
    """
    IC decay curve: Spearman IC of signal vs each 1-min return horizon.

    returns_by_minute must have columns ret_{h}m for each h in horizons
    (or ret_1m, ret_2m, … ret_30m built externally from the LOB midprice).
    Returns DataFrame with columns: horizon, ic, n.
    """
    rows = []
    for h in horizons:
        col = f"ret_{h}m"
        if col not in returns_by_minute.columns:
            rows.append({"horizon": h, "ic": float("nan"), "n": 0})
            continue
        ret  = returns_by_minute[col]
        mask = signal.notna() & ret.notna()
        n    = int(mask.sum())
        ic   = float(spearmanr(signal[mask], ret[mask])[0]) if n >= 10 else float("nan")
        rows.append({"horizon": h, "ic": ic, "n": n})
    return pd.DataFrame(rows)
