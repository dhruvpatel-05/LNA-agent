"""
src/analysis/regression.py — Conditional-OFI regression wrapper.

Delegates to src.regression.fit_conditional_ofi (already-clean module).
Provides a convenience function that runs all horizons and formats results.

Does NOT import from legacy/.
"""

from __future__ import annotations

import pandas as pd

from src.config import HORIZONS
from src.regression import fit_conditional_ofi


_SCALAR_KEYS = [
    "horizon", "delta", "scorer", "n",
    "alpha", "gamma", "delta1", "delta2",
    "b_pos", "b_neg",
    "se_b_pos", "se_b_neg",
    "wald_t", "wald_p",
    "r2", "aic",
]


def run_conditional_regressions(
    panel: pd.DataFrame,
    horizons: list[int] = HORIZONS,
    delta: int = 0,
    scorer: str = "llm_score",
) -> pd.DataFrame:
    """
    Fit the conditional-OFI regression at each horizon.

    Model: r_{t+h} = α + γ·OFI + δ₁s⁺ + δ₂s⁻ + β₊·OFI·s⁺ + β₋·OFI·s⁻
    HAC SEs (Newey-West). Wald H₀: β₊ = β₋.

    Parameters
    ----------
    panel    : Per-event panel with ofi_z, {scorer}, ret_{h}m columns.
    horizons : Forward-return horizons in minutes.
    delta    : Separation window (0 = no shift; delta>0 not yet wired).
    scorer   : Sentiment column for the hinge split.

    Returns
    -------
    DataFrame with one row per horizon, columns from _SCALAR_KEYS.
    """
    rows = []
    for h in horizons:
        result = fit_conditional_ofi(panel, horizon=h, delta=delta, scorer=scorer)
        rows.append({k: result.get(k) for k in _SCALAR_KEYS})
    return pd.DataFrame(rows)
