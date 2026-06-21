"""
src/signals/rules.py — Fixed-rule (non-LLM-agent) trading signals.

Signal definitions:
  ofi       : ofi_z_w                    — OFI z-score winsorized at ±OFI_WINSOR
  ofi_x_llm : ofi_z_w * llm_score       — OFI × LLM interaction (winsorized OFI)
  ofi_x_lm  : ofi_z_w * lm_score        — OFI × LM interaction (winsorized OFI)
  llm        : llm_score                 — LLM sentiment only
  lm         : lm_score                  — Loughran-McDonald only

Winsorization: raw ofi_z values can reach ±15 on low-variance days (single outlier
bar / near-zero day std). OFI_WINSOR = 5.0 clips these artifacts while preserving
genuine ≥5σ moves (which are extremely rare under a well-behaved distribution).
Raw ofi_z is left untouched in the panel; ofi_z_w is added as a derived column.

`add_rule_signals(panel)` adds all columns in one pass.
`RULE_SIGNALS` lists the signal names for downstream analysis.
Does NOT import from legacy/.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.signals.interactions import add_all_interactions

OFI_WINSOR: float = 5.0   # clip ofi_z at ±5σ before constructing signals

RULE_SIGNALS: list[str] = ["ofi", "ofi_x_llm", "ofi_x_lm", "llm", "lm"]


def add_rule_signals(panel: pd.DataFrame) -> pd.DataFrame:
    """
    Add all fixed-rule signal columns to the panel.

    Requires: ofi_z, llm_score, lm_score columns in panel.
    Returns a copy with additional columns:
      ofi_z_w  — winsorized OFI z-score (intermediate; kept for transparency)
      ofi, ofi_x_llm, ofi_x_lm, llm, lm — tradeable signals
    """
    out = add_all_interactions(panel)
    out["ofi_z_w"]   = out["ofi_z"].clip(-OFI_WINSOR, OFI_WINSOR)
    out["ofi"]       = out["ofi_z_w"]
    out["ofi_x_llm"] = out["ofi_z_w"] * out["llm_score"]
    out["ofi_x_lm"]  = out["ofi_z_w"] * out["lm_score"]
    out["llm"]       = out["llm_score"]
    out["lm"]        = out["lm_score"]
    return out
