#!/usr/bin/env bash
# run_all.sh — Ordered pipeline skeleton for LNA-Agent.
# Edit TICKER and SUFFIX before running.
# Do NOT run this script directly without reviewing each step.

set -euo pipefail

TICKER="${TICKER:-AAPL}"
DATE_FROM="${DATE_FROM:-2019-01-01}"
DATE_TO="${DATE_TO:-2020-12-31}"
SUFFIX="${DATE_FROM}_${DATE_TO}"
LOB_DIR="data/lobster/raw/${TICKER}"

echo "=== LNA-Agent pipeline: ${TICKER} [${SUFFIX}] ==="

# ── Step 1: OFI baseline ────────────────────────────────────────────────────
# Computes ofi_z, mid, ret_1m/5m/15m from 1-min LOBSTER snapshots.
# Output: results/ofi/ofi_data_${TICKER}_${SUFFIX}.csv
# python legacy/src/ofi.py --batch "${TICKER}" \
#     --data-dir "${LOB_DIR}" \
#     --date-from "${DATE_FROM}" \
#     --date-to   "${DATE_TO}"

# ── Step 2: Sentiment scoring ────────────────────────────────────────────────
# Scores headlines with LM and LLM (claude-haiku). Requires ANTHROPIC_API_KEY.
# Output: results/sentiment/sentiment_${TICKER}_${SUFFIX}.csv
# python legacy/src/sentiment.py \
#     --ticker "${TICKER}" \
#     --suffix "${SUFFIX}"

# ── Step 3: FinBERT scoring (optional) ──────────────────────────────────────
# Adds finbert_score to the scored headlines. Requires GPU or slow CPU run.
# (No standalone CLI; import sentiment_finbert.score_headlines in a notebook.)

# ── Step 4: Align headlines to LOB bars ─────────────────────────────────────
# Snaps headlines to preceding 1-min bar (2-min tolerance).
# Output: results/align/aligned_${TICKER}_${SUFFIX}.csv
# python legacy/src/align.py \
#     --ticker "${TICKER}" \
#     --suffix "${SUFFIX}"

# ── Step 5: Unconditional IC analysis (Notebook 01) ─────────────────────────
# jupyter nbconvert --to notebook --execute \
#     notebooks/01_unconditional.ipynb \
#     --output notebooks/01_unconditional_executed.ipynb

# ── Step 6: Conditional OFI regression (Notebook 02) ────────────────────────
# jupyter nbconvert --to notebook --execute \
#     notebooks/02_conditional_ofi.ipynb \
#     --output notebooks/02_conditional_ofi_executed.ipynb

# ── Step 7: Contamination placebos (Notebook 03) ────────────────────────────
# (Makes ~900 LLM API calls for ~300 events × 3 conditions.)
# jupyter nbconvert --to notebook --execute \
#     notebooks/03_contamination_placebo.ipynb \
#     --output notebooks/03_contamination_placebo_executed.ipynb

# ── Step 8: Clean window checks (Notebook 04) ───────────────────────────────
# jupyter nbconvert --to notebook --execute \
#     notebooks/04_clean_window.ipynb \
#     --output notebooks/04_clean_window_executed.ipynb

# ── Step 9: Net-of-cost (Notebook 05) ───────────────────────────────────────
# Requires LOB ask/bid columns joined to the panel (see costs.py TODO(schema)).
# jupyter nbconvert --to notebook --execute \
#     notebooks/05_net_of_cost.ipynb \
#     --output notebooks/05_net_of_cost_executed.ipynb

# ── Step 10: Baselines (Notebook 06) ────────────────────────────────────────
# jupyter nbconvert --to notebook --execute \
#     notebooks/06_baselines.ipynb \
#     --output notebooks/06_baselines_executed.ipynb

# ── Step 11: Agent run (optional) ───────────────────────────────────────────
# Runs the full LLM agentic decision layer. Expensive in API calls.
# python legacy/src/agent.py \
#     --ticker "${TICKER}" \
#     --suffix "${SUFFIX}"

# ── Step 12: Compile report ──────────────────────────────────────────────────
# pdflatex -output-directory report report/report.tex
# pdflatex -output-directory report report/report.tex   # second pass for TOC

echo "=== Pipeline skeleton complete. Uncomment steps to run. ==="
