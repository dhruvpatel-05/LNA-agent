# Preregistration — LNA-Agent Math 285J

**Timestamp:** 2026-06-19  
**Author:** Dhruv Patel  
**Project:** News-Conditional Intraday Alpha (Math 285J, UCLA)

---

## Primary Hypothesis

> **The OFI×LLM interaction signal on "confirmed" events has a positive
> 1-minute Spearman IC.**

Formally: among events where `sign(llm_score) == sign(ofi_z)` and both are
nonzero ("confirmed" events), `Spearman(ofi_z * llm_score, ret_1m) > 0`.

**Operationalisation**  
- Signal: `ofi_x_llm = ofi_z * llm_score`  
- Confirmed filter: `sign(llm_score) != 0` AND `sign(ofi_z) != 0` AND
  `sign(llm_score) == sign(ofi_z)`  
- Horizon: 1 minute  
- Estimator: Spearman rank correlation (distribution-free)  
- Confidence interval: 5 000-replicate percentile bootstrap  
- Reported: **uncorrected**, one-sided (IC > 0); this is the primary claim

This test is run **once** in notebook 02, section "Primary test", on the pooled
2016–2020 panel. The result is not chosen post-hoc.

---

## Secondary Analyses (all subject to BH FDR at q = 0.05)

The following cells are exploratory. All p-values are reported in a pooled
Benjamini–Hochberg FDR table (notebook 06). A cell "survives correction" only
if `p_adj ≤ 0.05`.

| Cell group | Variables crossed |
|---|---|
| Unconditional IC | scorer (LM, LLM, OFI) × horizon (1, 5, 15 min) × window (2016-20, 2025) |
| Conditional OFI regression | scorer × horizon × coefficient (b_pos, b_neg, Wald) |
| Relevance-stratified IC | bucket (direct/sector/macro/unrelated) × horizon |
| Per-ticker baseline | ticker × strategy (agent, rule, logistic, GBM, trivial) |
| Agent chosen-horizon IC | ticker × agent mode (LLM, rule) |

All secondary findings are flagged as such in the notebooks and must survive BH
correction before being described as statistically reliable.

---

## What is NOT pre-specified

- Placebo IC drops (anonymized/shuffled headlines): descriptive, no p-value cutoff
- Net-of-cost PnL: economic, not statistical inference
- IC decay curve shape beyond 1/5/15 min
- Cross-ticker spillover patterns

---

## Data exclusions pre-committed

- Events outside market hours (09:30–16:00) already excluded at alignment step
- Rows with missing `llm_score`, `ofi_z`, or `ret_{h}m` dropped per analysis
- AAPL 2025 LLM agent panel (n=10) excluded from agent comparisons (too small)

---

## Model version

`claude-haiku-4-5-20251001` (Anthropic). Version is fixed; re-scoring with a
newer model version is a different experiment.
