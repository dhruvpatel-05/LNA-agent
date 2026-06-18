# LNA-Agent — News-Conditional Intraday Alpha

LLM sentiment (claude-haiku) + LOBSTER limit-order-book OFI confirmation.
Math 285J final project, UCLA, June 2026.

## Hypothesis

`OFI_t × sent_t` predicts 1/5/15-minute forward returns better than either
signal alone. Tested via Spearman IC with bootstrap CIs and BH FDR correction.

## Repo layout

```
src/              New research modules (stats_rigor, regression, costs, placebo, baselines)
notebooks/        One notebook per finding (01–06)
report/           report.tex (article) and slides.tex (beamer)
results/
  figures/        Saved plots
  tables/         Saved CSV tables
legacy/           Archived original pipeline code (reference only)
  src/            Original ofi.py, align.py, sentiment.py, agent.py, …
  tests/
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

See `run_all.sh` for the full ordered pipeline.  Quick start:

```bash
# 1. Compute OFI for one ticker
python legacy/src/ofi.py --batch AAPL \
    --data-dir data/lobster/raw/AAPL \
    --date-from 2019-01-01 --date-to 2020-12-31

# 2. Score headlines
python legacy/src/sentiment.py --ticker AAPL \
    --suffix 2019-01-01_2020-12-31

# 3. Align events to LOB bars
python legacy/src/align.py --ticker AAPL \
    --suffix 2019-01-01_2020-12-31

# 4. Run notebooks in order
jupyter notebook notebooks/
```

## Notebooks

| # | Notebook | Finding |
|---|----------|---------|
| 01 | `01_unconditional` | IC table, bootstrap CIs, FDR correction, IC decay |
| 02 | `02_conditional_ofi` | Conditional OFI regression, Wald test |
| 03 | `03_contamination_placebo` | Anonymization / shuffled-headline placebos |
| 04 | `04_clean_window` | Snap-lag robustness, open/close filter |
| 05 | `05_net_of_cost` | Gross vs net IC, event-level PnL |
| 06 | `06_baselines` | Logistic, GBM, trivial anchors |

## Environment variables

Set in `.env` (never commit):

```
ANTHROPIC_API_KEY=sk-ant-...
```

## Citation

Model: `claude-haiku-4-5-20251001` (Anthropic, 2025).
LOB data: LOBSTER (<https://lobsterdata.com>).
