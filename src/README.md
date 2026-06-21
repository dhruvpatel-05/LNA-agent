# src/ — LNA Pipeline Modules

## Module map

| Module | Purpose |
|---|---|
| `config.py` | All constants: model string, horizons, paths, sector profiles, agent prompt. |
| `run.py` | CLI entry point: `build \| agent \| analyze \| all`. |
| `data/ofi.py` | Load and normalise pre-computed OFI files uploaded from the LOBSTER server. |
| `data/market.py` | Market-excess returns — SPY primary, EW cross-section fallback. |
| `data/sentiment.py` | Score headlines: LM lexicon (offline) + LLM (claude-haiku). Resumable. |
| `data/panel.py` | Join OFI bars with scored headlines via asof merge → per-event panel. |
| `signals/interactions.py` | `ofi_x_llm = ofi_z × llm_score`, `ofi_x_lm = ofi_z × lm_score`. |
| `signals/rules.py` | All fixed-rule signals in one pass: `ofi`, `ofi_x_llm`, `ofi_x_lm`, `llm`, `lm`. |
| `signals/agent.py` | LLM trading agent — one call per event, resumable `.jsonl` cache. |
| `analysis/ic.py` | Spearman IC + bootstrap CI + BH FDR correction. |
| `analysis/regression.py` | Conditional OFI regression across all horizons. |
| `analysis/pnl.py` | Paper-trading PnL: total, mean/trade, hit rate, std. |
| `analysis/compare.py` | Agent vs all fixed rules — IC + PnL side-by-side, BH once across full grid. |
| `stats_rigor.py` | IC bootstrap, BH correction (reused, not modified). |
| `regression.py` | `fit_conditional_ofi`, HAC SEs, Wald test (reused, not modified). |
| `placebo.py` | Anonymization placebo (reused). |
| `baselines.py` | Logistic / GBM baselines (reused). |
| `costs.py` | Round-trip cost model (reused). |
| `join_lob_quotes.py` | LOB–quote asof merge utility (reused). |

## Run order

```bash
# 1. Upload pre-computed OFI files to data/clean10_2025/ofi/{ticker}.csv

# 2. Fetch headlines (scripts/fetch_alpaca_news.py → data/clean10_2025/news/)
python scripts/fetch_alpaca_news.py

# 3. Score headlines + assemble per-event panels
python src/run.py build

# 4. LLM trading agent (resumable)
python src/run.py agent --limit 50   # smoke test
python src/run.py agent               # full run

# 5. IC, regression, PnL, comparison tables
python src/run.py analyze --with-agent --returns both

# Or all at once
python src/run.py all --with-agent --returns both
```

## CLI flags

| Flag | Stage | Effect |
|---|---|---|
| `--tickers AAPL NVDA` | all | Override tickers.txt |
| `--no-llm` | build, all | LM lexicon only; skip LLM headline scoring |
| `--limit N` | agent, all | Score at most N new events per ticker |
| `--dry-run` | agent, all | No API calls; shows what would be scored |
| `--returns raw\|excess\|both` | analyze, all | Return type for IC/PnL |
| `--with-agent` | analyze, all | Include agent_signal in comparison table |
| `--horizons 1 5 15` | analyze, all | Override horizon list |
| `--out PATH` | analyze, all | Output directory for CSV tables |

## Outstanding TODO

- `src/regression.py` — `delta>0` return-window shift not yet wired (raw LOB midprice required)
