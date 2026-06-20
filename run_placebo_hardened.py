"""
Hardened anonymization placebo for LNA-Agent.
Skips shuffled condition this run (anonymization is the key test).

Named scores: re-use existing panel llm_score (pre-computed with
  claude-haiku-4-5-20251001; avoids double-billing for the named condition).
Anonymized scores: fresh Haiku calls via src.placebo.anonymize + score_llm.

Results saved incrementally after each window so a kill still leaves data.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "legacy/src")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import PANELS_2025, PANELS_2016
from src.placebo import anonymize
from src.stats_rigor import paired_ic_diff
from src.report_io import save_table

# Pinned model (must match legacy/src/sentiment.py LLM_MODEL)
_HAIKU_MODEL = "claude-haiku-4-5-20251001"
print(f"Model pinned: {_HAIKU_MODEL}")

# Import scorer from legacy — confirms the same model string is in use
from sentiment import score_llm, LLM_MODEL as _LEGACY_MODEL  # type: ignore

assert _HAIKU_MODEL == _LEGACY_MODEL, (
    f"Model mismatch: expected {_HAIKU_MODEL}, got {_LEGACY_MODEL}"
)

N_EVENTS = 200
N_BOOT   = 1000
SEED     = 42
OUT_CSV  = Path("results/tables/placebo_clean_window.csv")
SNAP_DIR = Path("results_runs/2026-06-19")
SNAP_DIR.mkdir(parents=True, exist_ok=True)
(SNAP_DIR / "tables").mkdir(exist_ok=True)

HORIZONS = [1, 5, 15]
all_results = []


def score_anon_batch(rows_df: pd.DataFrame, ticker_col: str = "stock") -> list[float]:
    """
    Score anonymized versions of headlines. Returns a list of floats.
    Prints progress every 50 calls.
    """
    scores = []
    total = len(rows_df)
    for i, (_, row) in enumerate(rows_df.iterrows()):
        ticker = str(row.get(ticker_col, ""))
        hl     = str(row.get("headline", ""))
        anon   = anonymize(hl, ticker)
        scores.append(score_llm(anon))
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"    [{i+1}/{total}] anon scored", flush=True)
    return scores


def run_window(window_label: str, panel: pd.DataFrame) -> list[dict]:
    """
    Score anonymized headlines, compute paired IC drop per horizon.
    Returns list of result dicts.
    """
    df = panel.dropna(subset=["headline", "llm_score"]).copy()

    # Stratified sample: ~equal events per ticker if pool
    if len(df) > N_EVENTS:
        per_ticker = max(1, N_EVENTS // df["stock"].nunique())
        sampled = (
            df.groupby("stock", group_keys=False)
              .apply(lambda g: g.sample(min(len(g), per_ticker), random_state=SEED))
        )
        if len(sampled) < N_EVENTS:
            extra = df.drop(sampled.index).sample(
                min(N_EVENTS - len(sampled), len(df) - len(sampled)),
                random_state=SEED + 1
            )
            sampled = pd.concat([sampled, extra])
        df = sampled.head(N_EVENTS).reset_index(drop=True)

    n_calls = len(df)
    print(f"\n  Window: {window_label}")
    print(f"  Call estimate: {n_calls} anon calls "
          f"(named scores reused from panel llm_score; model={_HAIKU_MODEL})")
    print(f"  Tickers: {df['stock'].value_counts().to_dict()}")

    t0 = time.time()
    df["anon_score"] = score_anon_batch(df)
    elapsed = time.time() - t0
    print(f"  Scored {n_calls} headlines in {elapsed:.1f}s "
          f"({elapsed/n_calls:.2f}s/call)", flush=True)

    # Compute paired IC drop per horizon
    rows = []
    for h in HORIZONS:
        ret_col = f"ret_{h}m"
        if ret_col not in df.columns:
            continue
        valid = df.dropna(subset=["llm_score", "anon_score", ret_col])
        if len(valid) < 10:
            continue
        res = paired_ic_diff(
            valid["llm_score"], valid["anon_score"], valid[ret_col],
            n_boot=N_BOOT, seed=SEED,
        )
        rows.append({
            "window":   window_label,
            "horizon":  h,
            "ic_named": res["ic_a"],
            "ic_anon":  res["ic_b"],
            "diff":     res["diff"],
            "ci_lo":    res["ci_lo"],
            "ci_hi":    res["ci_hi"],
            "p":        res["p_boot"],
            "n":        res["n"],
            "model":    _HAIKU_MODEL,
        })
    return rows


# ── WINDOW 1: Clean window (Aug-Dec 2025) — run first ────────────────────────

print("\n" + "="*60)
print("WINDOW 1: 2025 clean window (Aug-Dec 2025)")
print("="*60)

frames_2025 = []
for ticker, path in PANELS_2025.items():
    p = Path(path)
    if not p.exists():
        print(f"  skip {ticker}")
        continue
    df = pd.read_csv(p)
    df["stock"] = ticker
    frames_2025.append(df)

if frames_2025:
    pool_2025 = pd.concat(frames_2025, ignore_index=True)
    results_2025 = run_window("2025", pool_2025)
    all_results.extend(results_2025)

    # Save immediately after clean window
    tbl_partial = pd.DataFrame(all_results)
    tbl_partial.to_csv(OUT_CSV, index=False)
    import shutil
    shutil.copy(OUT_CSV, SNAP_DIR / "tables" / OUT_CSV.name)
    print(f"\n  Saved clean-window results → {OUT_CSV}")

    # Print clean-window result immediately
    print("\n" + "="*60)
    print("CLEAN WINDOW (2025) RESULTS")
    print("="*60)
    for r in results_2025:
        sig = "CI excludes 0 (significant)" if (r["ci_lo"] > 0 or r["ci_hi"] < 0) else "CI crosses 0 (not significant)"
        print(f"  {r['horizon']}-min: ic_named={r['ic_named']:+.4f}, "
              f"ic_anon={r['ic_anon']:+.4f}, "
              f"diff={r['diff']:+.4f}, "
              f"95%CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}], "
              f"p={r['p']:.3f}, n={r['n']}")
        h1 = r["horizon"] == 1
        if h1:
            print(f"  → {'NO significant IC collapse at 1-min' if r['p'] > 0.05 else 'SIGNIFICANT IC collapse at 1-min'} "
                  f"({'consistent with leakage-free window' if r['p'] > 0.05 else 'contamination suspected'})")
    print("="*60)
else:
    print("  No 2025 panels found — skipping window 1.")

# ── WINDOW 2: 2016-2020 (if time budget allows) ───────────────────────────────

print("\n" + "="*60)
print("WINDOW 2: 2016-2020")
print("="*60)

frames_2016 = []
for ticker, path in PANELS_2016.items():
    p = Path(path)
    if not p.exists():
        print(f"  skip {ticker}")
        continue
    df = pd.read_csv(p)
    df["stock"] = ticker
    frames_2016.append(df)

if frames_2016:
    pool_2016 = pd.concat(frames_2016, ignore_index=True)
    results_2016 = run_window("2016-2020", pool_2016)
    all_results.extend(results_2016)

    # Save combined results
    tbl_final = pd.DataFrame(all_results)
    tbl_final.to_csv(OUT_CSV, index=False)
    import shutil
    shutil.copy(OUT_CSV, SNAP_DIR / "tables" / OUT_CSV.name)
    print(f"\n  Saved combined results → {OUT_CSV}")

    print("\n2016-2020 placebo IC drops:")
    for r in results_2016:
        print(f"  {r['horizon']}-min: ic_named={r['ic_named']:+.4f}, "
              f"ic_anon={r['ic_anon']:+.4f}, "
              f"diff={r['diff']:+.4f}, "
              f"95%CI=[{r['ci_lo']:+.4f},{r['ci_hi']:+.4f}], "
              f"p={r['p']:.3f}, n={r['n']}")

# ── Final table + LaTeX ───────────────────────────────────────────────────────

if all_results:
    tbl = pd.DataFrame(all_results)
    save_table(
        tbl,
        "placebo_clean_window",
        caption=(
            f"Anonymization placebo: LLM sentiment IC before and after entity "
            f"anonymization. Named scores use existing panel llm\\_score; "
            f"anonymized scores freshly computed via claude-haiku (model={_HAIKU_MODEL}). "
            f"diff = ic\\_named $-$ ic\\_anon; 95\\% bootstrap CI; "
            f"p = fraction of bootstrap draws with diff $\\leq 0$."
        ),
        label="tab:placebo",
    )
    shutil.copy(
        Path("results/tables/placebo_clean_window.csv"),
        SNAP_DIR / "tables" / "placebo_clean_window.csv",
    )

    # Side-by-side comparison
    print("\n--- SIDE-BY-SIDE ---")
    print(tbl[["window","horizon","ic_named","ic_anon","diff","ci_lo","ci_hi","p","n"]].to_string(
        index=False, float_format="{:.4f}".format))

print("\nDone.")
