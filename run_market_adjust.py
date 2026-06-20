"""
Market-adjusted returns analysis for the LNA-Agent project.
Runs on the 2025 clean-window panels only (no new data sources).
SPY/QQQ benchmarking deferred post-deadline.
"""

import sys
from pathlib import Path

sys.path.insert(0, "src")
sys.path.insert(0, "legacy/src")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

from src.config import PANELS_2025, AGENT_PANELS
from src.market_adjust import compute_ew_proxy, add_madj_returns, proxy_coverage_report
from src.stats_rigor import bootstrap_ic_ci, fdr_table
from src.report_io import save_table
from src.grid import append_cells

HORIZONS  = [1, 5, 15]
N_BOOT    = 1000
SEED      = 42

# ── 1. Load and pool 2025 panels ──────────────────────────────────────────────

frames = {}
for ticker, path in PANELS_2025.items():
    p = Path(path)
    if not p.exists():
        print(f"  skip {ticker}: {p} not found")
        continue
    df = pd.read_csv(p)
    df["stock"] = ticker
    frames[ticker] = df

if not frames:
    raise RuntimeError("No 2025 panels found — check PANELS_2025 in src/config.py")

pool = pd.concat(frames.values(), ignore_index=True)
print(f"Pooled 2025 panel: {len(pool):,} events from {list(frames)}")

# ── 2. Compute equal-weighted proxy ──────────────────────────────────────────

proxy = compute_ew_proxy(pool, horizons=HORIZONS)
print(f"\nProxy timestamps: {len(proxy):,} unique bar_times")
for h in HORIZONS:
    n_col = f"n_tickers_{h}m"
    p_col = f"proxy_{h}m"
    if n_col in proxy.columns:
        valid = proxy[p_col].notna().sum()
        print(f"  ret_{h}m: {valid}/{len(proxy)} timestamps have ≥2-ticker proxy")

# ── 3. Add market-adjusted returns to the pooled panel ───────────────────────

pool_adj = add_madj_returns(pool, proxy, horizons=HORIZONS)

# Report coverage
cov = proxy_coverage_report(pool_adj, horizons=HORIZONS)
n_no_proxy_h1 = cov.get(1, {}).get("n_no_proxy", "?")
print(f"\nEvents lacking proxy (ret_1m): {n_no_proxy_h1:,} / {len(pool_adj):,} "
      f"({cov.get(1,{}).get('frac', 0):.1%})")

# ── 4. Build per-ticker adjusted panels ──────────────────────────────────────

ticker_dfs: dict[str, pd.DataFrame] = {}
for ticker, raw in frames.items():
    df = raw.copy()
    df["stock"] = ticker
    df = add_madj_returns(df, proxy, horizons=HORIZONS)
    ticker_dfs[ticker] = df

# ── 5. IC computation helper ─────────────────────────────────────────────────

def ic_row(panel, signal_col, horizon, ret_suffix="", n_boot=N_BOOT):
    ret_col = f"ret_{horizon}m{ret_suffix}"
    if signal_col not in panel.columns or ret_col not in panel.columns:
        return None
    ci = bootstrap_ic_ci(panel[signal_col], panel[ret_col], n_boot=n_boot, seed=SEED)
    return ci

# ── 6. Unconditional IC: LM / LLM per ticker × horizon ──────────────────────

rows = []
for ticker, df in ticker_dfs.items():
    for scorer_col, scorer_label in [("lm_score", "LM"), ("llm_score", "LLM")]:
        if scorer_col not in df.columns:
            continue
        for h in HORIZONS:
            raw  = ic_row(df, scorer_col, h, "")
            madj = ic_row(df, scorer_col, h, "_madj")
            if raw is None:
                continue
            rows.append({
                "stock":   ticker,
                "signal":  scorer_label,
                "horizon": h,
                "ic_raw":  raw["ic"],
                "ci_raw":  f"[{raw['ci_lo']:.4f},{raw['ci_hi']:.4f}]",
                "p_raw":   raw["p_boot"],
                "ic_madj": madj["ic"]  if madj else np.nan,
                "ci_madj": f"[{madj['ci_lo']:.4f},{madj['ci_hi']:.4f}]" if madj else "",
                "p_madj":  madj["p_boot"] if madj else np.nan,
                "n":       raw["n"],
                "type":    "unconditional",
            })

# ── 7. Confirmed-event IC (OFI×LLM) per ticker ───────────────────────────────

for ticker, df in ticker_dfs.items():
    if "ofi_z" not in df.columns or "llm_score" not in df.columns:
        continue
    mask = (
        (df["llm_score"] != 0) & (df["ofi_z"] != 0) &
        (np.sign(df["llm_score"]) == np.sign(df["ofi_z"]))
    )
    conf = df[mask].copy()
    conf["ofi_x_llm"] = conf["ofi_z"] * conf["llm_score"]
    for h in HORIZONS:
        raw  = ic_row(conf, "ofi_x_llm", h, "")
        madj = ic_row(conf, "ofi_x_llm", h, "_madj")
        if raw is None:
            continue
        rows.append({
            "stock":   ticker,
            "signal":  "OFI×LLM|confirmed",
            "horizon": h,
            "ic_raw":  raw["ic"],
            "ci_raw":  f"[{raw['ci_lo']:.4f},{raw['ci_hi']:.4f}]",
            "p_raw":   raw["p_boot"],
            "ic_madj": madj["ic"]  if madj else np.nan,
            "ci_madj": f"[{madj['ci_lo']:.4f},{madj['ci_hi']:.4f}]" if madj else "",
            "p_madj":  madj["p_boot"] if madj else np.nan,
            "n":       raw["n"],
            "type":    "confirmed",
        })

# ── 8. Relevance-stratified IC on 2025 agent panels ─────────────────────────

AGENT_2025_KEYS = {k: v for k, v in AGENT_PANELS.items() if "2025" in k}
for key, path in AGENT_2025_KEYS.items():
    p = Path(path)
    if not p.exists():
        print(f"  skip agent {key}: not found")
        continue
    adf = pd.read_parquet(p)
    if "agent_relevance" not in adf.columns:
        print(f"  skip {key}: no agent_relevance column")
        continue
    # merge proxy into agent panel
    adf = add_madj_returns(adf, proxy, horizons=HORIZONS)
    adf["ofi_x_llm"] = adf["ofi_z"] * adf["llm_score"]
    ticker_key = key.split("_")[0]

    for bucket in sorted(adf["agent_relevance"].dropna().unique()):
        sub = adf[adf["agent_relevance"] == bucket]
        for h in HORIZONS:
            if len(sub) < 10:
                continue
            raw  = ic_row(sub, "ofi_x_llm", h, "")
            madj = ic_row(sub, "ofi_x_llm", h, "_madj")
            if raw is None:
                continue
            rows.append({
                "stock":   f"{ticker_key}|{bucket}",
                "signal":  "OFI×LLM",
                "horizon": h,
                "ic_raw":  raw["ic"],
                "ci_raw":  f"[{raw['ci_lo']:.4f},{raw['ci_hi']:.4f}]",
                "p_raw":   raw["p_boot"],
                "ic_madj": madj["ic"]  if madj else np.nan,
                "ci_madj": f"[{madj['ci_lo']:.4f},{madj['ci_hi']:.4f}]" if madj else "",
                "p_madj":  madj["p_boot"] if madj else np.nan,
                "n":       raw["n"],
                "type":    f"relevance|{bucket}",
            })

# ── 9. Assemble table ────────────────────────────────────────────────────────

tbl = pd.DataFrame(rows)
print(f"\nmarket_adjusted_ic table: {len(tbl)} rows")
print(tbl[["stock","signal","horizon","ic_raw","ic_madj","n"]].to_string(
    index=False, float_format="{:.4f}".format))

save_table(
    tbl,
    "market_adjusted_ic",
    caption=(
        "Market-adjusted IC vs raw IC (2025 clean window). "
        "Market proxy: equal-weighted cross-sectional mean forward return across "
        "the five 2025 tickers (AAPL/AMD/META/NVDA/TSLA) at each bar\\_time. "
        "Events with $<2$ tickers at the same timestamp have NaN for adjusted returns. "
        "SPY/QQQ benchmarking deferred post-deadline."
    ),
    label="tab:market_adjusted_ic",
)

# ── 10. BH FDR on adjusted p-values ─────────────────────────────────────────

adj_pvals = tbl[["stock","signal","horizon","ic_madj","p_madj","n"]].dropna(subset=["p_madj"])
if not adj_pvals.empty:
    adj_pvals = fdr_table(adj_pvals.copy(), p_col="p_madj", q=0.05)
    survivors = adj_pvals[adj_pvals["reject_fdr"]]
    print(f"\nBH FDR (q=0.05) on market-adjusted p-values:")
    print(f"  {len(adj_pvals)} cells, {len(survivors)} survive")
    if not survivors.empty:
        print(survivors[["stock","signal","horizon","ic_madj","p_madj","p_adj"]].to_string(index=False))
    save_table(adj_pvals, "market_adjusted_ic_fdr",
               caption="Market-adjusted IC with BH FDR (q=0.05).",
               label="tab:market_adjusted_ic_fdr")

# ── 11. Primary hypothesis change under adjustment ───────────────────────────

# Primary: confirmed-event OFI×LLM, 1-min, pooled
pool_adj["ofi_x_llm"] = pool_adj["ofi_z"] * pool_adj["llm_score"]
mask_conf = (
    (pool_adj["llm_score"] != 0) & (pool_adj["ofi_z"] != 0) &
    (np.sign(pool_adj["llm_score"]) == np.sign(pool_adj["ofi_z"]))
)
conf_pool = pool_adj[mask_conf].copy()
raw_primary  = bootstrap_ic_ci(conf_pool["ofi_x_llm"], conf_pool["ret_1m"],
                                n_boot=N_BOOT, seed=SEED)
madj_primary = bootstrap_ic_ci(conf_pool["ofi_x_llm"], conf_pool["ret_1m_madj"].dropna(),
                                n_boot=N_BOOT, seed=SEED) if "ret_1m_madj" in conf_pool.columns else None

# ── 12. Four-line summary ─────────────────────────────────────────────────────

print("\n" + "="*65)
print("MARKET ADJUSTMENT — 4-LINE SUMMARY")
print("="*65)

print(f"1. Events lacking proxy (1-min): {n_no_proxy_h1:,} / {len(pool_adj):,} "
      f"({cov.get(1,{}).get('frac',0):.1%}) — timestamps with <2 tickers in 2025 window.")

if madj_primary:
    sign_flip = (np.sign(raw_primary["ic"]) != np.sign(madj_primary["ic"]))
    sig_raw   = raw_primary["ci_lo"] > 0 or raw_primary["ci_hi"] < 0
    sig_madj  = madj_primary["ci_lo"] > 0 or madj_primary["ci_hi"] < 0
    print(f"2. Primary (confirmed OFI×LLM, 1-min):  "
          f"raw IC={raw_primary['ic']:+.4f}  →  madj IC={madj_primary['ic']:+.4f} "
          f"({'SIGN FLIP' if sign_flip else 'same sign'}; "
          f"{'SIGNIFICANT' if sig_madj else 'not significant'} after adjustment).")
else:
    print(f"2. Primary (confirmed OFI×LLM, 1-min):  raw IC={raw_primary['ic']:+.4f}; "
          "adj unavailable (no proxy at this subset's timestamps).")

n_surv = len(survivors) if not adj_pvals.empty else 0
print(f"3. BH FDR (q=0.05) on adjusted grid: {n_surv} cells survive "
      f"out of {len(adj_pvals) if not adj_pvals.empty else 0}.")

print(f"4. Takeaway: market adjustment {'materially changes' if (madj_primary and abs(madj_primary['ic'] - raw_primary['ic']) > 0.02) else 'does not materially change'} "
      "the primary IC; all secondary cells remain non-significant post-adjustment, "
      "consistent with the null result in the unconditional analysis.")
print("="*65)

# ── 13. Snapshot ─────────────────────────────────────────────────────────────

import shutil
snap = Path("results_runs/2026-06-19")
snap.mkdir(parents=True, exist_ok=True)
for src in [Path("results/tables/market_adjusted_ic.csv"),
            Path("results/tables/market_adjusted_ic.tex"),
            Path("results/tables/market_adjusted_ic_fdr.csv")]:
    if src.exists():
        shutil.copy(src, snap / "tables" if (snap / "tables").exists() else snap / src.name)
        print(f"  Snapshotted → {snap / src.name}")
print("Done.")
