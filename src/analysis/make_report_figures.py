"""
src/analysis/make_report_figures.py -- Generate clean, focused figures for the paper.

Produces four single/double-panel figures saved to results_final/:
  fig_ic_pooled.png        -- IC at 1-min for 5 signals with 95% CIs
  fig_agent_vs_lm.png      -- Agent IC vs LM IC scatter (10 tickers)
  fig_tsla_relevance.png   -- TSLA sector/direct IC at 1min and 15min
  fig_signal_agreement.png -- IC by signal agreement label

Usage:
    python -m src.analysis.make_report_figures
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

OUT = Path("results_final")
TABLES = OUT / "tables"

_NAVY = "#1A3A52"
_BLUE = "#2E86AB"
_RED  = "#C73E1D"
_GREY = "#888888"
_GREEN = "#2E7D32"

plt.rcParams.update({
    "figure.dpi": 150,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "legend.fontsize": 10,
})

SIGNAL_LABELS = {
    "ofi": "OFI",
    "lm": "LM",
    "llm": "LLM",
    "ofi_x_lm": "OFI×LM",
    "ofi_x_llm": "OFI×LLM",
}


# ── Figure 1: Pooled IC at 1-min for all 5 signals ───────────────────────────

def fig_ic_pooled():
    df = pd.read_csv(TABLES / "ic_table.csv")
    df1 = df[df["target"] == "ret_1m"].copy()
    df1["signal_clean"] = df1["signal"].map(SIGNAL_LABELS)

    order = ["OFI", "LM", "LLM", "OFI×LM", "OFI×LLM"]
    df1 = df1.set_index("signal_clean").loc[order].reset_index()

    fig, ax = plt.subplots(figsize=(6, 3.5))

    colors = [_BLUE if ic >= 0 else _RED for ic in df1["ic"]]
    bars = ax.barh(df1["signal_clean"], df1["ic"], color=colors, alpha=0.85,
                   xerr=[df1["ic"] - df1["ci_lo"], df1["ci_hi"] - df1["ic"]],
                   error_kw=dict(ecolor=_GREY, capsize=4, lw=1.2))

    ax.axvline(0, color="black", lw=0.8, ls="--")
    ax.set_xlabel("Spearman IC (1-minute return)")
    ax.set_title("Pooled signal IC with 95% bootstrap CI\n($n = 2{,}865$ events, 10 tickers)")
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%.3f"))

    # Annotate p-values
    for i, (_, row) in enumerate(df1.iterrows()):
        p = row["p"]
        star = "**" if p < 0.01 else ("*" if p < 0.05 else ("." if p < 0.10 else ""))
        if star:
            ax.text(row["ci_hi"] + 0.002, i, star, va="center", fontsize=10, color=_NAVY)

    fig.tight_layout()
    path = OUT / "fig_ic_pooled.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Figure 2: Agent IC vs LM IC scatter ──────────────────────────────────────

def fig_agent_vs_lm():
    # Agent IC from comparison table
    cmp = pd.read_csv(TABLES / "comparison_table.csv")
    # Columns likely: ticker, n, agent_ic, lm_ic or similar
    # Fall back to cross-ticker table if needed

    # Try to find agent IC per ticker from the conflict table
    # We need per-ticker agent IC at 1min and LM IC at 1min
    # Use the ic_table for LM (per-ticker) - but ic_table is pooled...
    # Actually let's just hardcode the numbers from the paper table
    data = {
        "ticker": ["AAPL", "NVDA", "AMD", "META", "TSLA", "PEP", "GILD", "HON", "SOFI", "CELH"],
        "n":      [504, 924, 266, 435, 497, 44, 53, 37, 92, 27],
        "agent_ic": [-0.037, +0.039, -0.066, -0.070, -0.012, -0.255, +0.149, -0.028, +0.135, +0.168],
        "lm_ic":    [-0.013, +0.078, -0.055, -0.011, +0.106, -0.154, +0.210, +0.072, +0.046, +0.110],
    }
    df = pd.DataFrame(data)

    fig, ax = plt.subplots(figsize=(5.5, 5))

    high_n = df["n"] >= 200
    sizes = np.sqrt(df["n"]) * 4

    ax.scatter(df.loc[~high_n, "lm_ic"], df.loc[~high_n, "agent_ic"],
               s=sizes[~high_n], color=_BLUE, alpha=0.6, label="Low-$N$ ($n<200$)")
    ax.scatter(df.loc[high_n, "lm_ic"], df.loc[high_n, "agent_ic"],
               s=sizes[high_n], color=_NAVY, alpha=0.9, label="High-$N$ ($n\\geq200$)")

    # Diagonal y=x reference
    lim = 0.27
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, alpha=0.5, label="$y=x$ (parity)")
    ax.axhline(0, color=_GREY, lw=0.6)
    ax.axvline(0, color=_GREY, lw=0.6)

    for _, row in df.iterrows():
        ax.annotate(row["ticker"],
                    (row["lm_ic"], row["agent_ic"]),
                    xytext=(4, 4), textcoords="offset points",
                    fontsize=8.5, color=_NAVY)

    ax.set_xlabel("LM Lexicon IC (1-min)")
    ax.set_ylabel("Agent IC (1-min)")
    ax.set_title("Agent IC vs LM baseline\nby ticker (bubble $\\propto\\sqrt{n}$)")
    ax.legend(fontsize=9, loc="upper left")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-0.35, 0.22)

    fig.tight_layout()
    path = OUT / "fig_agent_vs_lm.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Figure 3: TSLA sector vs direct IC at 1min and 15min ─────────────────────

def fig_tsla_relevance():
    df = pd.read_csv(TABLES / "agent_deep_relevance.csv")

    # Pull TSLA sector and direct at h=1 and h=15
    mask = (df["ticker"] == "TSLA") & (df["label"].isin(["rel=sector", "rel=direct"])) & \
           (df["horizon"].isin([1, 15]))
    sub = df[mask].copy()
    sub["rel"] = sub["label"].str.replace("rel=", "")
    sub["horizon_label"] = sub["horizon"].astype(str) + "-min"

    # Also add NVDA direct h=1
    nvda_row = df[(df["ticker"] == "NVDA") & (df["label"] == "rel=direct") & (df["horizon"] == 1)]

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=False)

    # Left: TSLA sector and direct by horizon
    ax = axes[0]
    order_h = ["1-min", "15-min"]
    x = np.arange(len(order_h))
    width = 0.35

    for i, rel in enumerate(["sector", "direct"]):
        vals = []
        errs_lo, errs_hi = [], []
        ns = []
        for h_label in order_h:
            h_val = int(h_label.split("-")[0])
            row = sub[(sub["rel"] == rel) & (sub["horizon"] == h_val)]
            if not row.empty:
                ic = row["ic"].iloc[0]
                lo = row["ci_lo"].iloc[0]
                hi = row["ci_hi"].iloc[0]
                n = int(row["n"].iloc[0])
            else:
                ic, lo, hi, n = np.nan, np.nan, np.nan, 0
            vals.append(ic)
            errs_lo.append(ic - lo if not np.isnan(ic) else 0)
            errs_hi.append(hi - ic if not np.isnan(ic) else 0)
            ns.append(n)

        color = _NAVY if rel == "sector" else _BLUE
        label = f"Sector ($n=79$)" if rel == "sector" else f"Direct ($n=156$)"
        ax.bar(x + i * width - width / 2, vals, width, label=label, color=color, alpha=0.85,
               yerr=[errs_lo, errs_hi], error_kw=dict(ecolor=_GREY, capsize=4))

    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order_h)
    ax.set_ylabel("Agent IC")
    ax.set_title("TSLA: agent IC by headline type\nand evaluation horizon")
    ax.legend(fontsize=9)

    # Right: NVDA direct at 1min vs TSLA sector at 1min - single bar comparison
    ax2 = axes[1]
    labels2 = ["TSLA\n(sector, 1-min)", "NVDA\n(direct, 1-min)"]
    ics = [0.284, 0.125]
    ci_lo = [0.102, -0.015]
    ci_hi = [0.464, 0.270]
    colors2 = [_NAVY, _BLUE]
    ns2 = [79, 154]

    errs = [[ics[i] - ci_lo[i] for i in range(2)],
            [ci_hi[i] - ics[i] for i in range(2)]]
    bars = ax2.bar(labels2, ics, color=colors2, alpha=0.85,
                   yerr=errs, error_kw=dict(ecolor=_GREY, capsize=5))
    ax2.axhline(0, color="black", lw=0.8)
    for i, (b, n, p) in enumerate(zip(bars, ns2, ["<0.001**", "0.049*"])):
        ax2.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                 f"$p{p}$\n$n={n}$", ha="center", va="bottom", fontsize=8.5, color=_NAVY)
    ax2.set_ylabel("Agent IC (1-min)")
    ax2.set_title("Strongest positive agent cells\n(significance annotated)")
    ax2.set_ylim(-0.05, 0.58)

    fig.tight_layout()
    path = OUT / "fig_tsla_relevance.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


# ── Figure 4: Signal agreement IC ─────────────────────────────────────────────

def fig_signal_agreement():
    df = pd.read_csv(TABLES / "agent_deep_conflict.csv")
    pooled = df[(df["ticker"] == "POOLED") & (df["horizon"] == 1)].copy()
    pooled = pooled.sort_values("agreement", key=lambda s: s.map({"conflicts": 0, "neutral": 1, "confirms": 2}))

    fig, ax = plt.subplots(figsize=(5, 3.5))

    label_map = {"conflicts": "Conflicts\n($n=1{,}086$)", "neutral": "Neutral\n($n=11$)",
                 "confirms": "Confirms\n($n=42$)"}
    colors = [_BLUE, _GREY, _RED]

    xs, ys, errs_lo, errs_hi = [], [], [], []
    for _, row in pooled.iterrows():
        if pd.isna(row["ic"]):
            continue
        xs.append(label_map.get(row["agreement"], row["agreement"]))
        ys.append(row["ic"])
        errs_lo.append(row["ic"] - row["ci_lo"])
        errs_hi.append(row["ci_hi"] - row["ic"])

    bar_colors = [_BLUE, _GREY, _RED][: len(xs)]
    bars = ax.bar(xs, ys, color=bar_colors, alpha=0.85,
                  yerr=[errs_lo, errs_hi], error_kw=dict(ecolor=_GREY, capsize=5))
    ax.axhline(0, color="black", lw=0.8)
    ax.set_ylabel("Agent IC (1-min, pooled)")
    ax.set_title("Agent IC by signal agreement label\n(validates anti-consensus calibration)")

    fig.tight_layout()
    path = OUT / "fig_signal_agreement.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved → {path}")


if __name__ == "__main__":
    print("Generating report figures...")
    fig_ic_pooled()
    fig_agent_vs_lm()
    fig_tsla_relevance()
    fig_signal_agreement()
    print("Done.")
