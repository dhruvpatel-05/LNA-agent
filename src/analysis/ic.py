"""
src/analysis/ic.py — Information Coefficient (IC) evaluation.

Thin wrapper over stats_rigor.bootstrap_ic_ci and BH correction.

Does NOT import from legacy/.
"""

from __future__ import annotations

import pandas as pd

from src.stats_rigor import bootstrap_ic_ci, benjamini_hochberg


def ic_table(
    panel: pd.DataFrame,
    signals: list[str],
    targets: list[str],
    n_boot: int = 1000,
    q: float = 0.05,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Compute Spearman IC + bootstrap CI for every (signal, target) pair.

    Parameters
    ----------
    panel    : Per-event DataFrame with signal and target columns.
    signals  : List of signal column names.
    targets  : List of return target column names.
    n_boot   : Bootstrap replications.
    q        : BH FDR threshold.

    Returns
    -------
    DataFrame with columns:
      signal, target, ic, ci_lo, ci_hi, p, n, bh_reject
    Rows are sorted by signal, then target.
    """
    rows = []
    for sig in signals:
        for tgt in targets:
            sub = panel[[sig, tgt]].dropna()
            n   = len(sub)
            if n < 10:
                rows.append({
                    "signal": sig, "target": tgt,
                    "ic": float("nan"), "ci_lo": float("nan"),
                    "ci_hi": float("nan"), "p": float("nan"), "n": n,
                })
                continue
            result = bootstrap_ic_ci(
                sub[sig], sub[tgt],
                n_boot=n_boot, seed=random_state,
            )
            rows.append({
                "signal": sig,
                "target": tgt,
                "ic":     result["ic"],
                "ci_lo":  result["ci_lo"],
                "ci_hi":  result["ci_hi"],
                "p":      result["p_boot"],
                "n":      n,
            })

    df = pd.DataFrame(rows)
    if len(df) > 0 and "p" in df.columns:
        valid_p = df["p"].notna()
        reject  = pd.Series(False, index=df.index)
        if valid_p.any():
            reject_arr, _ = benjamini_hochberg(df.loc[valid_p, "p"].values, q=q)
            reject[valid_p] = reject_arr
        df["bh_reject"] = reject.astype(bool)

    return df.sort_values(["signal", "target"]).reset_index(drop=True)


def confirmed_mask(panel: pd.DataFrame) -> pd.Series:
    """
    Boolean mask for 'confirmed events': sign(llm_score) == sign(ofi_z), both non-zero.

    This is the primary hypothesis subset.
    """
    llm = panel["llm_score"].fillna(0)
    ofi = panel["ofi_z"].fillna(0)
    return (llm != 0) & (ofi != 0) & (llm.apply(lambda x: 1 if x > 0 else -1)
                                        == ofi.apply(lambda x: 1 if x > 0 else -1))
