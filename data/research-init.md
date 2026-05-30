# Research Initialization: News-Conditional Intraday Alpha
**Hypothesis:** LLM sentiment scores on news headlines combined with LOBSTER order flow imbalance (OFI) and depth thinning produces higher out-of-sample information coefficient (IC) than Loughran-McDonald (LM) lexicon sentiment alone, across 10 large-cap S&P 500 names at 1/5/15 minute horizons.

**Date initialized:** 2026-05-29  
**Status:** Proposed — under investigation

---

## 1. Conceptual Background

### 1.1 The Intraday Alpha Problem

Intraday return predictability at short horizons (1–15 min) is driven by two orthogonal information channels:

1. **Soft information** — news, earnings calls, analyst commentary. Directional but noisy; signal decay is fast (often sub-5-minute for liquid names).
2. **Hard microstructure information** — order flow, queue depth, trade imbalance. High frequency, low latency, but blind to *why* pressure exists.

The research hypothesis proposes that conditioning soft-information signals on hard microstructure state recovers alpha that is washed out when either channel is used alone. The key intuition: a bullish LLM sentiment score on a headline has very different implications depending on whether the LOB is already thinning on the ask side (corroborating) vs. deepening (contradicting).

---

## 2. Component Knowledge

### 2.1 Loughran-McDonald (LM) Lexicon

**Source:** Loughran & McDonald (2011), *J. Finance* — "When Is a Liability Not a Liability? Textual Analysis, Dictionaries, and 10-Ks."

**What it is:**  
A domain-specific word list for financial text, comprising ~2,700 negative and ~354 positive words calibrated against 10-K filings. Replaces general-purpose lexicons (e.g., Harvard GI) that misclassify finance-specific words ("liability", "risk", "tax" are negative in finance but neutral in general use).

**Strengths:**
- Deterministic, interpretable, zero inference cost.
- Well-calibrated for SEC filings and earnings text.
- No API dependency, fully reproducible.

**Weaknesses for intraday news:**
- Calibrated on *long-form* regulatory text (10-Ks), not short-form news headlines.
- Cannot capture negation scope ("not better than expected" → counts positives from "better", misses negation).
- Cannot process context, sarcasm, or implicit sentiment ("beats by a penny" is positive but contains no positive words).
- Dictionary is static; new financial terminology post-2011 (e.g., "quantitative tightening", "SPAC", "deglobalization") is unscored.
- Headline-length text has very few words: often 0 sentiment words hit, giving a zero signal.

**IC baseline expectation:** Literature suggests rank IC of LM sentiment on next-5-minute mid-quote returns in the range of **0.01–0.03** for liquid large-cap names, with high zero-inflation from vocabulary miss-rate.

---

### 2.2 LLM Sentiment Scoring

**Paradigm:** Use a large language model (e.g., GPT-4o, Llama-3, FinBERT, or purpose-tuned models) to score a news headline or short article on a continuous or categorical sentiment scale.

**Approaches (ordered by cost/quality tradeoff):**
| Approach | Latency | Cost | In-context reasoning |
|---|---|---|---|
| FinBERT (fine-tuned BERT) | <10ms | near-zero | None |
| GPT-3.5 zero-shot | ~200ms | low | Moderate |
| GPT-4o zero-shot | ~800ms | moderate | Strong |
| GPT-4o + CoT | ~2s | high | Strong + explicit |
| Ensemble of above | high | high | Best |

**Why LLMs should outperform LM lexicon on headlines:**
- **Negation handling:** "Misses estimates by wide margin" → strong negative, no lexicon words needed.
- **Implicit sentiment:** "Earnings in line" vs. "Earnings in line but guidance cut" → LLM differentiates.
- **Entity disambiguation:** Same headline can be positive for one firm and negative for its competitor.
- **Headline-specific calibration:** Short-form text is the natural training regime for transformer-based models.

**Key confound:** LLMs are slow relative to the 1-minute horizon. Practical constraint: the sentiment score must be available before the horizon start. For 1-minute IC testing, the timestamp of headline release relative to bar open is the critical data alignment issue.

**Relevant prior results (training knowledge):**
- Lopez-Lira & Tang (2023, *Journal of Finance*): ChatGPT sentiment on tweets predicts next-day abnormal returns; outperforms Vader and LM.
- Koa et al. (2024): GPT-4 zero-shot sentiment on financial news outperforms FinBERT and LM on daily return prediction.
- At *intraday* horizons: evidence is thinner. The shorter the horizon, the more microstructure noise dominates, but also the shorter the window before signal decay.

---

### 2.3 Order Flow Imbalance (OFI)

**Source:** Cont, Kukanov & Stoikov (2014), *Management Science* — "The Price Impact of Order Book Events."

**Definition:**  
OFI at level $\ell$ over interval $[t_{k-1}, t_k]$:

$$\text{OFI}^{(\ell)}_k = \Delta q^{bid,\ell}_k \cdot \mathbf{1}[\Delta P^{bid,\ell}_k \geq 0] - \Delta q^{ask,\ell}_k \cdot \mathbf{1}[\Delta P^{ask,\ell}_k \leq 0]$$

where $q^{bid,\ell}$ and $q^{ask,\ell}$ are the quantities at the $\ell$-th price level on each side.

**Multi-level OFI (Xu et al. 2023):** Aggregate across $L$ levels with weights $w_\ell \propto e^{-\lambda \ell}$, capturing queue pressure beyond best bid/ask.

**Why OFI predicts returns:**
- A positive OFI (buy-side pressure: bid queue growing or ask queue depleting) mechanically precedes upward price movement as market orders arrive to clear the imbalance.
- At 1-minute horizons, OFI rank IC against forward mid-quote returns typically runs **0.05–0.12** for liquid large-caps — an order of magnitude above LM sentiment alone.
- OFI is essentially a *revealed preference* signal: traders who know something are putting orders in the book.

**Integration with news:** The critical question is whether LLM sentiment and OFI are *conditionally* informative. Hypothesized mechanism:
- A positive LLM sentiment score on a news headline, *if* followed by rising OFI in the next 30 seconds, signals that informed traders are acting on the news.
- A positive LLM sentiment score *without* OFI confirmation may be noise (already priced, or the market disagrees).
- This interaction is not capturable by either signal alone.

---

### 2.4 Depth Thinning

**Definition:** Depth thinning is the reduction in resting limit order volume at the best bid or ask (or across the top $N$ levels), typically in the seconds/minutes preceding a large directional move.

**Mechanism:** Liquidity providers (market makers) withdraw quotes when they detect adverse selection risk — i.e., when they believe an informed trade is imminent. Thinning is therefore a *leading indicator* of informed order flow, preceding OFI by seconds.

**Measurement:** 
$$\text{DThin}^{ask}_{t} = \frac{q^{ask}_{t_0} - q^{ask}_{t}}{q^{ask}_{t_0}}$$
where $t_0$ is a rolling baseline (e.g., 5-minute VWAP of depth). Asymmetric thinning (ask side thins more than bid) is bullish; symmetric thinning indicates a liquidity crisis not a directional signal.

**News linkage:** When a headline arrives, market makers face an immediate adverse selection problem: they do not know if the news is significant to this stock. The model prediction:
- Sophisticated market makers will thin depth within 10–30 seconds of a material headline.
- This creates a "depth thinning follows news" pattern that can be detected in LOBSTER data.
- Combining LLM sentiment direction with depth thinning asymmetry should sharpen the IC.

---

### 2.5 LOBSTER Data

**Source:** LOBSTER (Limit Order Book System — The Efficient Reconstructor), maintained by Humboldt University Berlin.

**What it provides:**
- Full limit order book reconstruction for NASDAQ-listed stocks.
- Message file: timestamp (nanosecond), event type (1=submission, 2=cancellation, 3=deletion, 4=execution visible, 5=execution hidden, 7=trading halt), direction, size, price.
- Orderbook file: snapshot of top-$N$ bid/ask price-quantity pairs at every event.
- Available for most NASDAQ-listed large-caps; coverage approximately 2007–present.

**For this study:**
- Need to select 10 large-cap S&P 500 names that are NASDAQ-listed (e.g., AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, INTC, CSCO, CMCSA).
- Request top-10 levels (sufficient for multi-level OFI and depth thinning computation).
- Align with news timestamps — requires a news data source (e.g., Refinitiv/LSEG, Bloomberg, Dow Jones Newswires) that provides millisecond-precision timestamps.

**Computational considerations:**
- A single trading day for one stock at 10 levels generates ~200–500MB of data.
- Reconstructing OFI at 1-minute bars from event-level LOBSTER requires careful bar alignment (use last event before bar close for snapshot; sum events within bar for flow).
- Timezone alignment: LOBSTER timestamps are in Eastern Time (US/Eastern); news feeds vary.

---

### 2.6 Information Coefficient (IC)

**Definition:** Spearman rank correlation between the signal (factor score) and forward return, computed cross-sectionally across assets at each time step $t$, then averaged:

$$\text{IC}_t = \text{SpearmanCorr}\left(\text{signal}_{i,t},\ r_{i,t \to t+h}\right)$$
$$\overline{\text{IC}} = \frac{1}{T} \sum_t \text{IC}_t$$

**For this study, IC is computed:**
- **Cross-sectionally** across 10 names at each news event time.
- **Or time-serially** per stock across all news events, then averaged across stocks — likely the more statistically powerful approach given only 10 names.
- **Out-of-sample:** Signals are constructed using only information available at or before $t$; returns are measured over $[t, t+h]$ for $h \in \{1, 5, 15\}$ minutes.

**IC benchmarks for intraday equity factors (large-cap US):**
| Signal type | Typical IC range (1-min) | Typical IC range (5-min) | Typical IC range (15-min) |
|---|---|---|---|
| LM lexicon sentiment | 0.01–0.03 | 0.01–0.02 | 0.005–0.015 |
| LLM sentiment (GPT-class) | 0.02–0.05 (estimated) | 0.02–0.04 | 0.01–0.025 |
| OFI (multi-level) | 0.05–0.12 | 0.04–0.09 | 0.02–0.06 |
| OFI + depth thinning | 0.06–0.14 | 0.05–0.10 | 0.03–0.07 |
| LLM + OFI + depth thinning | **hypothesis: 0.08–0.18** | **hypothesis: 0.06–0.12** | **hypothesis: 0.03–0.08** |

**Statistical testing of IC difference:**
- Primary test: Diebold-Mariano test on time series of $\text{IC}_t$ for composite vs. LM-only signals.
- Secondary test: Newey-West HAC t-test on $\overline{\text{IC}}$ difference, accounting for autocorrelation in IC series.
- Minimum sample for reliable IC estimation: ~500 news events per stock; at ~3–5 market-moving events/day per large-cap, this implies ~100–170 trading days minimum.

---

## 3. Claim Graph (Initial)

### Observations (Layer 0)
- **O1:** LM sentiment on headlines has high vocabulary miss-rate for short-form text (~40–60% of headlines have zero LM words).
- **O2:** LOBSTER OFI at 1-minute bars is positively autocorrelated and mean-reverting at 15-minute horizons.
- **O3:** Depth thinning events precede large directional moves by 10–60 seconds in existing microstructure literature.
- **O4:** LLM-class models demonstrate superior classification accuracy on financial news sentiment benchmarks (FPB, FiQA SA).

### Explanations (Layer 1)
- **E1:** LM vocabulary miss-rate → downward bias in measured IC (zero signal pollutes the rank correlation).
- **E2:** LLMs capture implicit and negated sentiment that LM cannot → higher signal-to-noise.
- **E3:** OFI is a revealed-preference signal from informed traders → mechanically predictive but blind to *why*.
- **E4:** Depth thinning is a leading indicator of adverse selection → measures *when* informed flow is imminent.

### Exploitation (Layer 2 — the hypothesis)
- **X1:** LLM sentiment + OFI + depth thinning as a composite signal outperforms LM-only.
- **X2:** The interaction term (LLM sentiment × OFI sign agreement) carries incremental IC beyond additive combination.

### Justifications (Layer 3)
- **J1:** IC framework is the standard factor evaluation tool (Grinold & Kahn, *Active Portfolio Management*).
- **J2:** Spearman IC is robust to outliers in return distributions, appropriate for intraday returns.
- **J3 (UNVERIFIED):** Multi-level OFI dominates single-level OFI — needs experimental confirmation.

---

## 4. Key Risks and Bridge Gaps

| Gap | Risk | Severity | Mitigation |
|---|---|---|---|
| News-to-LOB timestamp alignment | News headline timestamp may lag actual market awareness by seconds | **High** | Use LOBSTER message timestamps; test 0/5/10/30s lags |
| LLM inference latency vs. 1-min horizon | GPT-4o ~800ms latency — may not be actionable at 1-min | **Medium** | Use FinBERT or distilled model for production; GPT-4o for offline IC test |
| Sample selection (10 names) | Only NASDAQ large-caps available on LOBSTER; excludes NYSE names | **Medium** | Clearly scope to NASDAQ universe; note generalizability limit |
| News event sparsity | Market-moving headlines are rare; many news items move price 0 | **Medium** | Filter to events where $|\Delta mid| > 0.05\%$ in 5-min post-release |
| Multiple testing | 3 horizons × multiple signal variants → inflation | **Medium** | Bonferroni correction or pre-register primary hypothesis (5-min IC) |
| LM vs. LLM confound | LLMs also trained on financial text — may implicitly know post-news returns | **Low** | Use zero-shot scoring only; no fine-tuning on return outcomes |

---

## 5. Proposed Experimental Design

### Phase 1: Data Construction (Tier 1)
1. **Stock universe:** AAPL, MSFT, AMZN, GOOGL, META, NVDA, TSLA, CSCO, INTC, CMCSA (10 NASDAQ large-caps).
2. **LOBSTER data:** Request 2022-01-01 to 2023-12-31 (2 years), top-10 levels, all event types.
3. **News data:** Align with a news provider giving millisecond timestamps (Refinitiv, Bloomberg, or Benzinga API).
4. **LOB features per bar:**
   - Multi-level OFI (levels 1–5, exponential decay weights)
   - Depth thinning ratio: ask-side and bid-side, 30s rolling baseline
   - Mid-quote return: $r_{t \to t+h}$ for $h \in \{1, 5, 15\}$ min

### Phase 2: Sentiment Scoring (Tier 1)
1. **LM baseline:** Score each headline using LM word lists → net sentiment = (pos\_words - neg\_words) / total\_words; zero-fill when no words match.
2. **LLM scoring:** Score each headline using GPT-4o (or FinBERT for speed comparison) → continuous score in $[-1, 1]$.
3. **Composite signal:** $S_{composite} = \alpha \cdot S_{LLM} + \beta \cdot \text{OFI} + \gamma \cdot \text{DThin}_{asymmetry}$; fit $\alpha, \beta, \gamma$ via expanding-window ridge regression (walk-forward, no lookahead).

### Phase 3: IC Evaluation (Tier 2)
1. **Time-series IC per stock:** $\text{IC}_{i,h} = \text{SpearmanCorr}\left(S_{i,t},\ r_{i,t \to t+h}\right)$ over all news events for stock $i$.
2. **Cross-stock average:** $\overline{\text{IC}}_h = \frac{1}{10} \sum_i \text{IC}_{i,h}$
3. **DM test:** Test $H_0$: $\overline{\text{IC}}^{composite}_h = \overline{\text{IC}}^{LM}_h$ vs. $H_1$: composite $>$ LM, for each $h$.
4. **Interaction test:** Does $S_{LLM} \times \text{sign}(\text{OFI})$ add IC beyond additive model? (incremental IC regression).
5. **Decay analysis:** Plot IC as function of bar lag post-headline (0, 1, 2, ..., 15 min) — measures signal half-life.

### Phase 4: Robustness (Tier 2)
- Sub-sample by market regime (high VIX vs. low VIX days).
- Sub-sample by news category (earnings, macro, analyst, M&A).
- Sub-sample by time of day (open 9:30–10:30, midday, close 15:00–16:00).
- Comparison: FinBERT vs. GPT-4o sentiment IC.

---

## 6. Admission Gate Status

| Claim | Layer | Status | Blocker |
|---|---|---|---|
| LM sentiment has lower IC than LLM on intraday news | Exploitation | `proposed` | Needs experimental IC measurement |
| OFI is positively correlated with 1-min forward returns | Observation | `admitted` (literature-grounded) | — |
| Depth thinning precedes directional moves | Observation | `admitted` (literature-grounded) | — |
| Composite IC > LM-only IC (primary hypothesis) | Exploitation | `under_investigation` | Phase 3 IC evaluation |
| LLM × OFI interaction adds incremental IC | Exploitation | `proposed` | Needs Phase 3 interaction test |
| Signal is actionable within 1-min latency constraint | Justification | `proposed` | Needs FinBERT latency benchmark |

---

## 7. Key References (Training Knowledge — Unverified Citations)

- Loughran, T. & McDonald, B. (2011). "When Is a Liability Not a Liability?" *Journal of Finance*, 66(1), 35–65.
- Cont, R., Kukanov, A. & Stoikov, S. (2014). "The Price Impact of Order Book Events." *Management Science*, 60(5), 1356–1372.
- Grinold, R. & Kahn, R. (2000). *Active Portfolio Management*. McGraw-Hill. (IC framework)
- Lopez-Lira, A. & Tang, Y. (2023). "Can ChatGPT Forecast Stock Price Movements?" *Journal of Finance* (forthcoming/SSRN).
- Xu, K. et al. (2023). "Multi-level Order Flow Imbalance." *(Working paper — unverified exact citation)*
- Lehalle, C-A. & Laruelle, S. (2013). *Market Microstructure in Practice*. World Scientific. (depth thinning background)
- Diebold, F.X. & Mariano, R.S. (1995). "Comparing Predictive Accuracy." *Journal of Business & Economic Statistics*, 13(3), 253–263. (DM test)

> **Note:** All citations marked "unverified" should be confirmed via literature search before paper submission. The LOBSTER-specific citation is: Huang, R. & Polak, T. (2011). "LOBSTER: Limit Order Book Reconstruction System." Humboldt University Working Paper.

---

## 8. Next Actions (Ordered by Priority)

1. **[IMMEDIATE]** Verify the Lopez-Lira & Tang (2023) and Xu et al. multi-level OFI papers via arXiv/S2 search — these are the closest prior work and must be cited accurately.
2. **[IMMEDIATE]** Confirm LOBSTER data availability for the 10 target names and the news data source with millisecond timestamps.
3. **[SHORT-TERM]** Run Phase 1 data construction: compute OFI and depth thinning from LOBSTER at 1/5/15 min bars.
4. **[SHORT-TERM]** Score the news sample with both LM and FinBERT (fast proxy for LLM) to get preliminary IC estimates.
5. **[MEDIUM-TERM]** Run Phase 3 IC evaluation with DM test as primary statistical test.
6. **[MEDIUM-TERM]** If composite IC > LM IC with p < 0.05 on DM test → admit `X1` claim, proceed to write-up.
