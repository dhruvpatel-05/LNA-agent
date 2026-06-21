# LNA-Agent — News-Conditional Intraday Alpha

LLM sentiment (claude-haiku) + LOBSTER limit-order-book OFI confirmation.
Math 285J final project, UCLA, June 2026.

## Repo layout

```
src/
  config.py           Tickers, paths, model string, sector profiles, agent prompt
  stats_rigor.py      Bootstrap IC, BH FDR correction, paired IC diff
  run.py              CLI entry point (sentiment / agent / analyze subcommands)
  data/               Panel assembly, OFI loading, sentiment loading, SPY merge
  signals/            OFI×sentiment interaction signals, agent signal, rule signals
  analysis/           IC tables, regression, PnL, agent deep-dives, rigor_fixes
  baselines.py        LM / LLM scorer wrappers
report/
  paper.tex           Final paper (Math 285J)
  refs.bib            Bibliography
results_final/
  fig_*.png           Paper figures
  tables/             All output CSVs (IC, PnL, regression, event-type BH, excess IC)
scripts/
  fetch_alpaca_news.py  News headline fetcher
legacy/             Archived original pipeline (reference only)
data/               LOBSTER + news panels (licensed, not committed — see .gitignore)
```

## Data access (LOBSTER — licensed, not committed)

1. Request access at <https://lobsterdata.com>.
2. Download level-2 1-minute snapshot files for your tickers.
3. Place under `data/lobster/raw/{TICKER}/`.
4. The loader `legacy/src/ofi.py:load_lob()` expects columns
   `ask`, `bid`, `ask_size`, `bid_size` (or their LOBSTER equivalents).

News headlines: `stock_news_2024_2025.csv` is included for reference;
replace with a licensed news feed for publication-quality results.

## Environment setup

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env          # add ANTHROPIC_API_KEY
```

Requires Python ≥ 3.11.

## Run order

```bash
# 1. Score sentiment for all tickers
python -m src.run sentiment

# 2. Run agent on all tickers
python -m src.run agent

# 3. Analyze: IC tables, BH correction, rigor fixes
python -m src.run analyze
```

## Environment variables

Set in `.env` (never commit):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Citation

Model: `claude-haiku-4-5-20251001` (Anthropic, 2025).
LOB data: LOBSTER (<https://lobsterdata.com>).
