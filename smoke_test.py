#!/usr/bin/env python3
"""
smoke_test.py  — Phase 0 gate check before filling notebooks.

Section 1: LOB quote coverage (join_l1_quotes on AAPL 2016-2020 sample)
Section 2: Conditional-OFI regression (fit_conditional_ofi on AAPL 2016-2020)
Section 4: Net-of-cost (net_ic on joined panel)

Exits 0 if all gates pass, 1 otherwise.
Sets an exit code so CI can pick it up; also prints a GATE line at the end.
"""
import sys, traceback, os
sys.path.insert(0, ".")
sys.path.insert(0, "legacy/src")

import pandas as pd
import numpy as np

RESULTS: dict = {}

AAPL_PANEL  = "results/align/aligned_AAPL_2016-01-01_2020-12-31.csv"
AAPL_LOB    = "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60"

# ── Section 1: LOB quote coverage ─────────────────────────────────────────────
print("\n=== Section 1: LOB quote coverage ===")
try:
    from src.join_lob_quotes import join_l1_quotes
    panel  = pd.read_csv(AAPL_PANEL, parse_dates=["bar_time", "timestamp"])
    sample = panel.head(200).copy()
    joined = join_l1_quotes(sample, lob_dir=AAPL_LOB, ticker_col="stock")
    cov    = float(joined["lob_matched"].mean())
    RESULTS["s1_coverage"] = cov
    RESULTS["s1_pass"]     = cov >= 0.50
    print(f"  Coverage : {cov:.1%}  [{'PASS' if cov >= 0.50 else 'FAIL'}]")
    if cov >= 0.50:
        print(f"  mid range: {joined['mid'].dropna().describe()[['min','50%','max']].to_dict()}")
except Exception:
    print("  FAIL — traceback follows:")
    traceback.print_exc()
    RESULTS["s1_pass"]     = False
    RESULTS["s1_coverage"] = 0.0

# ── Section 2: Conditional-OFI regression ─────────────────────────────────────
print("\n=== Section 2: Conditional-OFI regression ===")
try:
    from src.regression import fit_conditional_ofi
    panel = pd.read_csv(AAPL_PANEL)
    r     = fit_conditional_ofi(panel, horizon=1, delta=0, scorer="llm_score")
    if "error" in r:
        print(f"  FAIL — {r['error']}")
        RESULTS["s2_pass"] = False
    else:
        print(f"  n={r['n']}  b_pos={r['b_pos']:.5f}  b_neg={r['b_neg']:.5f}"
              f"  wald_p={r['wald_p']:.4f}  r2={r['r2']:.5f}  [PASS]")
        RESULTS["s2_pass"] = True
except Exception:
    print("  FAIL — traceback follows:")
    traceback.print_exc()
    RESULTS["s2_pass"] = False

# ── Section 4: Net-of-cost ────────────────────────────────────────────────────
print("\n=== Section 4: Net-of-cost ===")
try:
    from src.costs import net_ic
    panel  = pd.read_csv(AAPL_PANEL)
    r_raw  = net_ic(panel, horizon=1)
    print(f"  (fallback 5bps) gross={r_raw['gross_ic']:.4f}  net={r_raw['net_ic']:.4f}"
          f"  cov={r_raw['coverage_frac']:.1%}")

    if RESULTS.get("s1_pass"):
        from src.join_lob_quotes import join_l1_quotes
        joined = join_l1_quotes(
            pd.read_csv(AAPL_PANEL, parse_dates=["bar_time", "timestamp"]).head(200),
            lob_dir=AAPL_LOB, ticker_col="stock",
        )
        r_lob = net_ic(joined, horizon=1)
        print(f"  (real quotes)  gross={r_lob['gross_ic']:.4f}  net={r_lob['net_ic']:.4f}"
              f"  cov={r_lob['coverage_frac']:.1%}  [PASS]")
    else:
        print("  Real-quote path skipped (Section 1 failed)")
    RESULTS["s4_pass"] = True
except Exception:
    print("  FAIL — traceback follows:")
    traceback.print_exc()
    RESULTS["s4_pass"] = False

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("SMOKE TEST SUMMARY")
print("=" * 50)
s1  = RESULTS.get("s1_pass", False)
s2  = RESULTS.get("s2_pass", False)
s4  = RESULTS.get("s4_pass", False)
cov = RESULTS.get("s1_coverage", 0.0)
gate = s1 and s2 and s4

print(f"  Section 1 LOB coverage : {cov:.1%}   {'PASS' if s1 else 'FAIL'}")
print(f"  Section 2 regression   :            {'PASS' if s2 else 'FAIL'}")
print(f"  Section 4 net-of-cost  :            {'PASS' if s4 else 'FAIL'}")
print()
if gate:
    print("GATE: PASS — proceed with all notebooks")
else:
    print("GATE: FAIL — skip net-of-cost notebook and LLM-spend cells")
    if not s1:
        print("  Reason: LOB coverage < 50% — no real quote data for cost estimation")
    if not s2:
        print("  Reason: Regression failed — check statsmodels install")
    if not s4:
        print("  Reason: net_ic() raised an exception")

sys.exit(0 if gate else 1)
