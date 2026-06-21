"""
src/signals/interactions.py — Interaction signals between OFI and sentiment scores.

ofi_x_llm = ofi_z * llm_score  (primary interaction signal)
ofi_x_lm  = ofi_z * lm_score

All functions accept a panel DataFrame and return a copy with the new column(s) added.
Does NOT import from legacy/.
"""

from __future__ import annotations

import pandas as pd


def add_ofi_x_llm(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ofi_x_llm = ofi_z * llm_score."""
    out = panel.copy()
    out["ofi_x_llm"] = out["ofi_z"] * out["llm_score"]
    return out


def add_ofi_x_lm(panel: pd.DataFrame) -> pd.DataFrame:
    """Add ofi_x_lm = ofi_z * lm_score."""
    out = panel.copy()
    out["ofi_x_lm"] = out["ofi_z"] * out["lm_score"]
    return out


def add_all_interactions(panel: pd.DataFrame) -> pd.DataFrame:
    """Add all interaction signals in one pass."""
    out = panel.copy()
    out["ofi_x_llm"] = out["ofi_z"] * out["llm_score"]
    out["ofi_x_lm"]  = out["ofi_z"] * out["lm_score"]
    return out
