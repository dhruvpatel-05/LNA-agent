# LNA-Agent Baseline Results

**Generated:** 2026-06-20  
**Data window:** 2025-08-01 – 2025-12-31  
**Tickers:** AAPL, NVDA, AMD, META, TSLA, PEP, GILD, HON, SOFI, CELH  
**Panel:** 2,879 total events (news × OFI matches); 979 confirmed events (34%)  
**Returns:** Raw 1min / 5min / 15min forward returns (no excess/SPY adjustment yet)  
**OFI:** Pre-computed on LOBSTER server, winsorized at ±5σ before signal construction  
**Inference:** Spearman IC, 1,000-bootstrap CI (seed=42), p = fraction of bootstraps ≤ 0, BH FDR q=0.05 pooled across all 15 tests in each table

> **Low-N tickers:** PEP (44), GILD (53), HON (37), SOFI (92), CELH (27) are underpowered. CIs are wide and individual results unreliable; rely on pooled analysis only.

---

## 1. Pooled IC Table — All Events (n = 2,743–2,865)

Signals: `ofi` = ofi_z winsorized ±5σ; `llm`/`lm` = raw scores; `ofi_x_llm` / `ofi_x_lm` = interaction.

| signal    | target  |    IC  | CI 95%              |    p  | BH reject |
|-----------|---------|--------|---------------------|-------|-----------|
| lm        | ret_1m  | +0.044 | [+0.008, +0.082]    | 0.012 | False     |
| lm        | ret_15m | +0.014 | [−0.023, +0.052]    | 0.229 | False     |
| lm        | ret_5m  | −0.020 | [−0.057, +0.016]    | 0.862 | False     |
| ofi       | ret_5m  | +0.023 | [−0.014, +0.060]    | 0.110 | False     |
| ofi       | ret_15m | +0.014 | [−0.027, +0.054]    | 0.253 | False     |
| ofi       | ret_1m  | +0.015 | [−0.026, +0.054]    | 0.234 | False     |
| ofi_x_llm | ret_5m  | +0.019 | [−0.020, +0.059]    | 0.172 | False     |
| ofi_x_llm | ret_1m  | +0.003 | [−0.036, +0.044]    | 0.453 | False     |
| ofi_x_llm | ret_15m | −0.000 | [−0.041, +0.036]    | 0.517 | False     |
| ofi_x_lm  | ret_1m  | −0.032 | [−0.065, +0.003]    | 0.960 | False     |
| ofi_x_lm  | ret_15m | −0.022 | [−0.060, +0.013]    | 0.880 | False     |
| ofi_x_lm  | ret_5m  | −0.011 | [−0.048, +0.028]    | 0.710 | False     |
| llm       | ret_15m | −0.040 | [−0.077, +0.000]    | 0.975 | False     |
| llm       | ret_5m  | −0.037 | [−0.075, +0.000]    | 0.974 | False     |
| llm       | ret_1m  | −0.014 | [−0.051, +0.021]    | 0.783 | False     |

**Key finding:** No signal survives BH correction at q=0.05. `lm` at 1min is nominally significant (p=0.012) but does not clear the BH threshold given 15 simultaneous tests. LLM sentiment is directionally anti-predictive (negative IC at all horizons), suggesting it reflects information already reflected in price.

---

## 2. Pooled IC Table — Confirmed Events Only (n = 913–971)

Confirmed = sign(llm_score) == sign(ofi_z), both non-zero (~34% of panel). This is the primary pre-registered hypothesis subset.

| signal    | target  |    IC  | CI 95%              |    p  | BH reject |
|-----------|---------|--------|---------------------|-------|-----------|
| ofi_x_llm | ret_5m  | +0.040 | [−0.024, +0.108]    | 0.115 | False     |
| ofi_x_llm | ret_15m | +0.023 | [−0.045, +0.083]    | 0.248 | False     |
| lm        | ret_1m  | +0.046 | [−0.012, +0.104]    | 0.061 | False     |
| lm        | ret_15m | +0.020 | [−0.044, +0.077]    | 0.286 | False     |
| ofi_x_llm | ret_1m  | +0.007 | [−0.055, +0.070]    | 0.430 | False     |
| ofi       | ret_15m | −0.020 | [−0.085, +0.048]    | 0.725 | False     |
| ofi       | ret_5m  | −0.019 | [−0.087, +0.050]    | 0.720 | False     |
| ofi       | ret_1m  | −0.037 | [−0.106, +0.033]    | 0.868 | False     |
| llm       | ret_15m | −0.058 | [−0.123, +0.003]    | 0.968 | False     |
| llm       | ret_1m  | −0.023 | [−0.084, +0.046]    | 0.749 | False     |
| llm       | ret_5m  | −0.031 | [−0.097, +0.031]    | 0.842 | False     |
| ofi_x_lm  | ret_15m | −0.052 | [−0.117, +0.010]    | 0.948 | False     |
| ofi_x_lm  | ret_1m  | −0.022 | [−0.085, +0.040]    | 0.755 | False     |
| ofi_x_lm  | ret_5m  | −0.026 | [−0.093, +0.034]    | 0.784 | False     |
| lm        | ret_5m  | −0.018 | [−0.081, +0.044]    | 0.728 | False     |

**Key finding:** Confirmed events show *weaker* predictability than the full panel for OFI-based signals. The confirmation condition (OFI and LLM agreeing) does not amplify the IC — consistent with the calibration note in the agent prompt that agreement implies information is already priced.

---

## 3. Conditional OFI Regression

OFI regressed on returns split by `sign(llm_score)` (positive vs negative sentiment). HAC standard errors. Wald test: H₀ = β_pos == β_neg.

| horizon | n    | β_pos      | β_neg      | Wald p  |
|---------|------|------------|------------|---------|
| 1 min   | 2865 | −0.000112  | +0.000201  | 0.116   |
| 5 min   | 2836 | +0.000007  | +0.000130  | 0.769   |
| 15 min  | 2743 | +0.000146  | +0.000299  | 0.865   |

**Key finding:** No significant asymmetry — OFI's predictive slope does not differ by sentiment direction at any horizon. The hypothesis that LLM sentiment moderates OFI informativeness is not supported in the baseline.

---

## 4. Paper PnL Table — Pooled (no costs, no slippage)

Sign-of-signal strategy: long if signal > 0, short if signal < 0, flat if zero.

| signal    | target  | total_pnl | mean/trade | hit_rate |
|-----------|---------|-----------|------------|----------|
| ofi       | ret_5m  |  +0.257   | +0.000090  | 50.4%    |
| ofi       | ret_15m |  +0.141   | +0.000052  | 50.6%    |
| ofi       | ret_1m  |  +0.029   | +0.000010  | 50.1%    |
| ofi_x_llm | ret_5m  |  +0.105   | +0.000037  | 49.9%    |
| lm        | ret_1m  |  +0.050   | +0.000017  | 54.6%    |
| lm        | ret_15m |  +0.044   | +0.000016  | 51.6%    |
| ofi_x_llm | ret_15m |  −0.009   | −0.000003  | 50.1%    |
| ofi_x_llm | ret_1m  |  −0.021   | −0.000007  | 49.7%    |
| lm        | ret_5m  |  −0.066   | −0.000023  | 47.1%    |
| ofi_x_lm  | ret_5m  |  −0.085   | −0.000030  | 49.9%    |
| ofi_x_lm  | ret_15m |  −0.139   | −0.000051  | 49.9%    |
| ofi_x_lm  | ret_1m  |  −0.080   | −0.000028  | 46.7%    |
| llm       | ret_1m  |  −0.004   | −0.000001  | 50.0%    |
| llm       | ret_5m  |  −0.248   | −0.000087  | 47.9%    |
| llm       | ret_15m |  −0.385   | −0.000140  | 48.5%    |

**Key finding:** OFI alone is the best-performing baseline on paper PnL (not IC-significant, but consistently directionally positive). LLM sentiment loses money at 5min and 15min — suggesting it is a lagging/consensus signal at those horizons. Mean per-trade figures are very small (< 1bp), highlighting the need for tight execution.

---

## 5. Per-Ticker IC Highlights (Comparison Table)

Strongest individual signals from the comparison table (before BH correction within-ticker):

| signal    | ticker | horizon | IC     | CI                  | p     |
|-----------|--------|---------|--------|---------------------|-------|
| lm        | NVDA   | 1 min   | +0.078 | [+0.014, +0.139]    | 0.008 |
| lm        | TSLA   | 1 min   | +0.106 | [+0.015, +0.189]    | 0.010 |
| ofi       | NVDA   | 1 min   | +0.058 | [−0.011, +0.136]    | 0.049 |
| ofi_x_llm | HON   | 15 min  | +0.289 | [−0.042, +0.585]    | 0.045 |
| ofi_x_lm  | META  | 5 min   | +0.089 | [−0.010, +0.181]    | 0.042 |
| ofi_x_lm  | TSLA  | 1 min   | −0.122 | [−0.210, −0.034]    | 0.998 |
| ofi_x_llm | AAPL  | 5 min   | −0.105 | [−0.193, −0.017]    | 0.987 |

Note: HON 15min ofi_x_llm (IC=+0.289, p=0.045) and CELH 15min ofi_x_lm (IC=+0.475, n=26) are artifacts of very low N. Wide CIs confirm they are not informative.

NVDA and TSLA show the most consistent LM signal at 1min. Both are high-N tickers (918 and 494 events). These are the tickers to watch in the agent comparison.

---

## 6. Summary

| Question | Answer |
|----------|--------|
| Does any signal survive BH-FDR at q=0.05? | **No.** All 15 pooled tests fail to reject. |
| Does OFI predict returns? | **Weakly positive but not significant.** IC ≈ +0.015–+0.023, wide CIs span zero. |
| Does LLM sentiment add to OFI? | **No.** ofi_x_llm underperforms ofi alone on both IC and PnL. |
| Does LM lexicon add to OFI? | **Weakly yes at 1min** (lm IC=+0.044, p=0.012 before BH), but doesn't survive correction. |
| Does the confirmation condition help? | **No.** Confirmed events show weaker IC than the full panel. |
| Is OFI slope asymmetric by sentiment? | **No.** Wald p > 0.10 at all horizons. |
| Best paper PnL signal? | **ofi at 5min** (+0.257 total, 50.4% hit rate). LLM loses at 5min/15min. |

---

## Next Steps

- [ ] Run agent scoring: `python src/run.py agent`
- [ ] Run analysis with agent: `python src/run.py analyze --with-agent --returns raw`
- [ ] Add excess returns once SPY data available: `--returns both`
- [ ] Per-ticker agent vs baseline comparison will populate after agent run
