# LNA-Agent: Lobster News Alpha
News-conditional intraday alpha via LLM sentiment and LOB microstructure.

## Research Question
Does combining LLM sentiment with LOBSTER order flow imbalance produce higher out-of-sample IC than Loughran-McDonald lexicon alone?

## Data Sources

### News — SentARL (financial-news-multisource)
- **Dataset**: `financial-news-multisource/sentarl_combined` (HuggingFace)
- **Local cache**: `data/news/sentarl_full.parquet` (318k rows, 1997–2021)
- **Schema**: `date` (UTC ISO-8601), `text` (headline + article body), `extra_fields` (JSON)
- `extra_fields.stocks` contains the list of tickers tagged to each article
- Headlines extracted as first line of `text` (before first `\n`)
- Coverage (2016–2020, market hours): AAPL ~14k, AMZN ~13k, MSFT ~6k, JPM ~5k, NFLX ~5k

### LOB — LOBSTER Level-2 limit order book snapshots
- 1-minute snapshots, 10 levels, sourced from NASDAQ via LOBSTER
- Folder convention: `/Users/dhruvpatel/Downloads/_data_dwn_32_302__{TICKER}_2007-06-27_2021-07-01_10_60`
- OFI computed in `src/ofi.py`, saved to `results/ofi_data_{ticker}_{suffix}.csv`

### Sentiment scoring
- **LM**: Loughran-McDonald lexicon via `pysentiment2`
- **LLM**: `claude-haiku-4-5-20251001` (Anthropic API, prompt caching enabled)

## Pipeline
```
news.py → sentiment.py → align.py → ic_signal.py
                ↘ agent.py (agentic trade decision layer)
```
Run all tickers: `python src/run_all.py`
Single ticker: `python src/run_pipeline.py --ticker AAPL --date-from 2016-01-01 --date-to 2020-12-31 --llm`
Scan coverage: `python src/news.py --scan --ticker AAPL AMZN MSFT JPM NFLX`

## Structure
- `src/` — data pipeline, OFI computation, sentiment scoring, IC comparison, agent
- `notebooks/` — exploratory analysis and results
- `results/` — IC tables, aligned panels (CSV), event study plots
- `data/news/sentarl_full.parquet` — SentARL news cache
- `data/news/processed/` — per-ticker filtered headline CSVs


python src/ofi.py --batch AAPL --date-from 2016-01-01 --date-to 2020-12-31 --data-dir "/Users/dhruvpatel/Downloads/_data_dwn_32_302__AAPL_2007-06-27_2021-07-01_10_60"
python src/news.py --ticker AAPL --date-from 2016-01-01 --date-to 2020-12-31
python src/sentiment.py --ticker AAPL --suffix 2016-01-01_2020-12-31
python src/align.py --ticker AAPL --suffix 2016-01-01_2020-12-31
python src/ic_signal.py --ticker AAPL --suffix 2016-01-01_2020-12-31

python src/agent.py --ticker QCOM --suffix 2016-01-01_2020-12-31 --max-events 20

python src/agent.py --suffix 2016-01-01_2020-12-31