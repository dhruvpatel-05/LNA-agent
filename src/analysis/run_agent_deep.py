"""
src/analysis/run_agent_deep.py -- Run all four agent deep-dive experiments and print results.

Usage:
    python -m src.analysis.run_agent_deep
"""

from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.analysis.agent_deep import run_all


def _stars(p: float) -> str:
    if pd.isna(p):
        return "   "
    if p < 0.01:
        return "** "
    if p < 0.05:
        return "*  "
    if p < 0.10:
        return ".  "
    return "   "


def _fmt_ic(ic, ci_lo, ci_hi):
    if pd.isna(ic):
        return "  n/a  "
    return f"{ic:+.3f} [{ci_lo:+.3f}, {ci_hi:+.3f}]"


def print_exp1(df: pd.DataFrame):
    print("=" * 70)
    print("EXP 1: AGENT-CHOSEN HORIZON")
    print("Evaluating each trade at the horizon the agent selected.")
    print("-" * 70)
    print(f"{'Ticker':<10} {'Trades':>6} {'IC (95% CI)':>30} {'p':>7}  sig")
    print("-" * 70)
    for _, r in df.iterrows():
        if r["n"] < 5:
            continue
        print(f"{r['group']:<10} {int(r['n']):>6}  "
              f"{_fmt_ic(r['ic'], r['ci_lo'], r['ci_hi']):<30} "
              f"{r['p']:>7.3f}  {_stars(r['p'])}")
    print()


def print_exp2(df: pd.DataFrame):
    print("=" * 70)
    print("EXP 2: RELEVANCE-STRATIFIED IC (pooled across tickers)")
    print("Agent classifies each headline: direct / sector / macro / unrelated")
    print("-" * 70)
    pooled = df[df["ticker"] == "POOLED"].copy()
    for rel in ["direct", "sector", "macro", "unrelated"]:
        sub = pooled[pooled["label"] == f"rel={rel}"].sort_values("horizon")
        if sub.empty:
            continue
        print(f"\n  Relevance = {rel.upper()}")
        print(f"  {'Horizon':>8} {'n':>6} {'IC (95% CI)':>30} {'p':>7}  sig  BH")
        for _, r in sub.iterrows():
            bh = "REJ" if r.get("bh_reject") else "   "
            print(f"  {r['horizon']:>7}m {int(r['n']):>6}  "
                  f"{_fmt_ic(r['ic'], r['ci_lo'], r['ci_hi']):<30} "
                  f"{r['p']:>7.3f}  {_stars(r['p'])} {bh}")
    print()


def print_exp3(df: pd.DataFrame):
    print("=" * 70)
    print("EXP 3: HIGH-CONVICTION FILTER (pooled across tickers)")
    print("Filtering to trades where agent_size >= threshold")
    print("-" * 70)
    pooled = df[df["ticker"] == "POOLED"].copy()
    for thresh in sorted(pooled["threshold"].unique()):
        sub = pooled[pooled["threshold"] == thresh].sort_values("horizon")
        pct = sub["pct_of_trades"].iloc[0] * 100 if not sub.empty else 0
        print(f"\n  size >= {thresh:.1f}  ({pct:.0f}% of all trades retained)")
        print(f"  {'Horizon':>8} {'n':>6} {'IC (95% CI)':>30} {'p':>7}  sig")
        for _, r in sub.iterrows():
            print(f"  {r['horizon']:>7}m {int(r['n']):>6}  "
                  f"{_fmt_ic(r['ic'], r['ci_lo'], r['ci_hi']):<30} "
                  f"{r['p']:>7.3f}  {_stars(r['p'])}")
    print()


def print_exp4(df: pd.DataFrame):
    print("=" * 70)
    print("EXP 4: SIGNAL AGREEMENT STRATIFICATION (pooled across tickers)")
    print("'conflicts' = OFI/LM/LLM disagree; per prompt design this is")
    print("where the agent should resolve genuine ambiguity -> higher IC")
    print("-" * 70)
    pooled = df[df["ticker"] == "POOLED"].copy()
    for agreement in ["conflicts", "neutral", "confirms"]:
        sub = pooled[pooled["agreement"] == agreement].sort_values("horizon")
        if sub.empty:
            continue
        print(f"\n  agreement = {agreement.upper()}")
        print(f"  {'Horizon':>8} {'n':>6} {'IC (95% CI)':>30} {'p':>7}  sig")
        for _, r in sub.iterrows():
            print(f"  {r['horizon']:>7}m {int(r['n']):>6}  "
                  f"{_fmt_ic(r['ic'], r['ci_lo'], r['ci_hi']):<30} "
                  f"{r['p']:>7.3f}  {_stars(r['p'])}")
    print()


def print_summary(results: dict):
    print("=" * 70)
    print("SUMMARY: BEST POSITIVE CELLS (IC > 0, p < 0.10)")
    print("-" * 70)
    hits = []
    for exp_name, df in results.items():
        for _, r in df.iterrows():
            if r["n"] >= 20 and r["ic"] > 0 and r["p"] < 0.10:
                desc = f"[{exp_name}] group={r['group']} label={r['label']}"
                hits.append((r["ic"], r["p"], r["n"], desc,
                             _fmt_ic(r["ic"], r["ci_lo"], r["ci_hi"])))
    hits.sort(reverse=True)
    if not hits:
        print("  No cells with IC > 0 and p < 0.10 and n >= 20.")
    for ic, p, n, desc, ic_str in hits[:15]:
        print(f"  IC={ic_str}  p={p:.3f}  n={int(n):>5}  {desc}")
    print()


if __name__ == "__main__":
    results = run_all()

    print_exp1(results["chosen_horizon"])
    print_exp2(results["relevance"])
    print_exp3(results["high_conviction"])
    print_exp4(results["conflict"])
    print_summary(results)

    # Save tables
    out_dir = Path("results_final/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in results.items():
        df.to_csv(out_dir / f"agent_deep_{name}.csv", index=False)
    print(f"Tables saved to {out_dir}/")
