# LNA-Agent: Lobster News Alpha

Intraday alpha research using an LLM agent that combines news headlines, Loughran-McDonald sentiment, and LOBSTER order-flow imbalance to make structured trade decisions.

## Research Question

Does an LLM agent that reasons jointly about headline content, order-flow state, and sector context produce higher Spearman IC than mechanical signal combinations (LM lexicon, LLM sentiment, OFI alone)?

The agent is evaluated on its *chosen* horizon (the return window it selected per event), not a fixed window, making the IC comparison fairer than the constituent-signal baselines.

---

## Data Sources

### News — SentARL
- **Dataset**: `financial-news-multisource/sentarl_combined` (HuggingFace)
- **Local cache**: `data/news/sentarl_full.parquet` (318k rows, 1997–2021)
- **Schema**: `date` (UTC ISO-8601), `text` (headline + body), `extra_fields.stocks` (ticker tags)
- Headlines extracted as the first line of `text`
- Coverage (2016–2020, market hours): AAPL ~14k, AMZN ~13k, MSFT ~6k, JPM ~5k, NFLX ~5k

### LOB — LOBSTER Level-2 snapshots
- 1-minute snapshots, 10 levels, NASDAQ via LOBSTER
- Folder: `/Users/dhruvpatel/Downloads/_data_dwn_32_302__{TICKER}_...`
- OFI computed in `src/ofi.py`, saved to `results/ofi/ofi_data_{ticker}_{suffix}.csv`

### Sentiment scoring
- **LM**: Loughran-McDonald lexicon via `pysentiment2`
- **LLM**: `claude-haiku-4-5-20251001` (Anthropic API, prompt caching on system prompt per ticker)

---

## Pipeline

```
news.py ──► sentiment.py ──► align.py ──► ic_signal.py   (signal baselines)
                                 │
                                 └──► agent.py            (agentic layer)
                                          │
                                          └──► summarize_results.py
```

### Step-by-step (single ticker)
```bash
python src/ofi.py       --ticker AAPL --suffix 2016-01-01_2020-12-31
python src/news.py      --ticker AAPL --date-from 2016-01-01 --date-to 2020-12-31
python src/sentiment.py --ticker AAPL --suffix 2016-01-01_2020-12-31
python src/align.py     --ticker AAPL --suffix 2016-01-01_2020-12-31
python src/ic_signal.py --ticker AAPL --suffix 2016-01-01_2020-12-31

# Run agent (LLM mode)
python src/agent.py --ticker AAPL --suffix 2016-01-01_2020-12-31

# Run agent (rule-based ablation, no API calls)
python src/agent.py --ticker AAPL --suffix 2016-01-01_2020-12-31 --no-llm

# All tickers at once
python src/run_all.py
# or
python src/agent.py --suffix 2016-01-01_2020-12-31
```

### Utility flags
```bash
--max-events N     Cap events per ticker (for fast testing)
--no-llm           Use deterministic LM+OFI rule; no API calls
--anonymize        Replace entity names in headlines with TICKER before agent call
--spillover        After per-ticker runs, compute cross-asset IC spillover matrix
```

### Summarize saved results (no API, no data loading)
```bash
python src/summarize_results.py --suffix 2016-01-01_2020-12-31
```

---

## Agent Design

Each news event is processed by a structured LLM agent (`claude-haiku`).

**Inputs per event**
| Signal | Description |
|---|---|
| `headline` | Raw (or anonymized) news headline |
| `lm_score` | Loughran-McDonald polarity \[-1, 1\] |
| `llm_score` | Claude sentiment score \[-1, 1\] |
| `ofi_z` | Normalized order-flow imbalance |

**Agent output (structured JSON)**
| Field | Values |
|---|---|
| `relevance` | `direct` / `sector` / `macro` / `unrelated` |
| `signal_agreement` | `confirms` / `conflicts` / `neutral` |
| `conviction_reasoning` | One sentence on size driver |
| `direction` | `long` / `short` / `none` |
| `size` | \[0, 1\] — probability-like confidence |
| `horizon` | `1min` / `5min` / `15min` |
| `reasoning` | One sentence on the trade decision |

**Agent signal**: `agent_signal = sign(direction) × size`

**Prompt caching**: the sector profile is sent as a `cache_control: ephemeral` system prompt block, eliminating ~95% of input token cost for all but the first event per ticker per 5-minute window.

**Calibration rule in prompt**: when signals confirm each other (OFI, LM, LLM all agree), the agent is instructed to *reduce* size — the information is likely already priced. When signals conflict but the headline is clearly material and directly relevant, the agent is instructed to *raise* size.

---

## Evaluation

**IC metrics** (Spearman rank correlation of `agent_signal` vs realized returns):

| Metric | Description |
|---|---|
| Agent 1m/5m/15m | IC at each fixed horizon |
| Agent chosen horizon | IC where each event is evaluated at the horizon the agent selected |
| Agent chosen (random baseline) | Mean IC from 1000 shuffled-return bootstrap draws |
| LM / LLM / OFI | Constituent signal baselines at each horizon |

**Stratified analyses** (saved as CSVs):
- `size_ic_{ticker}_{suffix}_{mode}.csv` — IC by agent size quartile
- `relevance_ic_{ticker}_{suffix}_{mode}.csv` — IC by headline relevance category
- `agreement_ic_{ticker}_{suffix}_{mode}.csv` — IC by signal agreement category
- `spillover_{suffix}_llm.csv` — cross-ticker IC spillover matrix

**Decay curve**: `compute_comparison` checks for `ret_1m`, `ret_5m`, `ret_15m`, `ret_30m`, `ret_60m` and includes whichever are present, so the IC table automatically extends if longer-horizon returns are added to the aligned panel.

---

## Module Structure

```
src/
  profiles.py          Sector profiles, _AGENT_TASK prompt, _build_system_prompt
  decisions.py         decide_llm, decide_rule, _parse_decision, cost tracker
  evaluation.py        _ic, compute_comparison, stratified IC, _print_comparison,
                       cross_asset_spillover; HORIZONS = [1, 5, 15, 30, 60]
  agent.py             run_ticker, anonymize_headlines, argparse, __main__
  summarize_results.py Standalone results summary (no API, no data loading)
  align.py             News ↔ LOB alignment, forward return computation
  sentiment.py         LM + LLM scoring pipeline
  ofi.py               Order-flow imbalance from LOBSTER snapshots
  news.py              SentARL news loading and filtering
  ic_signal.py         Baseline IC comparison (pre-agent)
  run_all.py           Run full pipeline for all tickers
  run_pipeline.py      Run full pipeline for a single ticker

results/
  agent/
    agent_ic_{ticker}_{suffix}_{mode}.csv      IC comparison table
    agent_panel_{ticker}_{suffix}_{mode}.parquet   Full event panel with agent cols
    size_ic_{ticker}_{suffix}_{mode}.csv
    relevance_ic_{ticker}_{suffix}_{mode}.csv
    agreement_ic_{ticker}_{suffix}_{mode}.csv
  spillover_{suffix}_llm.csv
  summary_{suffix}.csv                          Cross-ticker summary
  ofi/
  sentiment/
  align/
  ic_signal/

data/news/
  sentarl_full.parquet
  processed/           Per-ticker filtered headline CSVs
```

---

## Tickers

| Ticker | Exchange | Notes |
|---|---|---|
| AAPL | NASDAQ | Mega-cap; news prices in within seconds |
| JPM | NYSE | Macro-sensitive; rate news leads company news |
| AMD | NASDAQ | Mid-cap; minutes to price in complex headlines |
| QCOM | NASDAQ | Event-driven; binary patent/licensing events |
| NFLX | NASDAQ | Earnings-driven; sustained 15-min moves |

---

## Cost

API cost is tracked per-run and printed after each ticker and at the end of the run. Prompt caching (system prompt is ~800 tokens per ticker) reduces typical cost to ~$0.01–0.03 per 100 events in LLM mode. `--no-llm` mode has zero API cost.
