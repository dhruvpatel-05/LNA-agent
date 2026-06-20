"""
config.py — Machine-local paths for LOBSTER data and aligned panels.

Update LOB_DIRS entries if your LOBSTER downloads live elsewhere.
These paths are intentionally not committed as data — the dict keys are
the standard tickers; values are the local flat directories of CSV files.
"""
from pathlib import Path

_DL = Path("/Users/dhruvpatel/Downloads")

# One directory per ticker containing {TICKER}_{YYYY-MM-DD}_*.csv files.
# 2025-window tickers (TSLA, META, NVDA) have no LOB downloads → empty string.
LOB_DIRS: dict[str, str] = {
    "AAPL": str(_DL / "_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60"),
    "AMD":  str(_DL / "_data_dwn_32_302__AMD_2007-06-27_2021-07-01_10_60"),
    "JPM":  str(_DL / "_data_dwn_32_302__JPM_2007-06-27_2021-07-01_10_60"),
    "NFLX": str(_DL / "_data_dwn_32_302__NFLX_2007-06-27_2021-07-01_10_60"),
    "QCOM": str(_DL / "_data_dwn_32_302__QCOM_2007-06-27_2021-07-01_10_60"),
    "MU":   str(_DL / "_data_dwn_32_302__MU_2007-06-27_2021-07-01_10_60"),
    "TSLA": "",   # no LOB download available
    "META": "",   # no LOB download available
    "NVDA": "",   # no LOB download available
}

_RESULTS = Path("results")

# Aligned panel paths — both date windows
PANELS_2016: dict[str, Path] = {
    t: _RESULTS / "align" / f"aligned_{t}_2016-01-01_2020-12-31.csv"
    for t in ["AAPL", "AMD", "JPM", "MU", "NFLX", "QCOM"]
}

PANELS_2025: dict[str, Path] = {
    t: _RESULTS / "align" / f"aligned_{t}_2025-08-01_2025-12-01.csv"
    for t in ["AAPL", "AMD", "META", "NVDA", "TSLA"]
}

# Agent panel paths (LLM mode only; exclude AAPL 2025 n=10)
AGENT_PANELS: dict[str, Path] = {
    "AMD_2016":  _RESULTS / "agent" / "agent_panel_AMD_2016-01-01_2020-12-31_llm.parquet",
    "NFLX_2016": _RESULTS / "agent" / "agent_panel_NFLX_2016-01-01_2020-12-31_llm.parquet",
    "QCOM_2016": _RESULTS / "agent" / "agent_panel_QCOM_2016-01-01_2020-12-31_llm.parquet",
    "AMD_2025":  _RESULTS / "agent" / "agent_panel_AMD_2025-08-01_2025-12-01_llm.parquet",
    "TSLA_2025": _RESULTS / "agent" / "agent_panel_TSLA_2025-08-01_2025-12-01_llm.parquet",
}

AGENT_RULE_PANELS: dict[str, Path] = {
    "AAPL_2025": _RESULTS / "agent" / "agent_panel_AAPL_2025-08-01_2025-12-01_rule.parquet",
    "AMD_2025":  _RESULTS / "agent" / "agent_panel_AMD_2025-08-01_2025-12-01_rule.parquet",
    "META_2025": _RESULTS / "agent" / "agent_panel_META_2025-08-01_2025-12-01_rule.parquet",
    "NVDA_2025": _RESULTS / "agent" / "agent_panel_NVDA_2025-08-01_2025-12-01_rule.parquet",
    "TSLA_2025": _RESULTS / "agent" / "agent_panel_TSLA_2025-08-01_2025-12-01_rule.parquet",
}

COLORS = {
    "blue":   "#2E86AB",
    "purple": "#A23B72",
    "orange": "#F18F01",
    "red":    "#C73E1D",
    "dark":   "#3B1F2B",
    "steel":  "#5E8FAA",
    "navy":   "#1A3A52",
}
