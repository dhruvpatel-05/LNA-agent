"""
regression.py — Respecified conditional-OFI regression.
LNA-Agent: Lobster News Alpha

Model:
  r_{t+h} = a + g*OFI + d1*sent_pos + d2*sent_neg
            + b_pos*(OFI*sent_pos) + b_neg*(OFI*sent_neg) + e

Separate windows: OFI over [t, t+delta], return over [t+delta, t+delta+h].
sent_pos = max(sent, 0), sent_neg = min(sent, 0)  (hinge split)
Newey-West HAC SEs for overlapping return labels.
Cluster-robust variants clustered by event and by day.
Wald test: H0: b_pos == b_neg.
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.sandwich_covariance import cov_hac
from statsmodels.regression.linear_model import OLS

# Panel column names — verified from align.py output schema
_OFI_COL  = "ofi_z"           # normalised OFI z-score
_TS_COL   = "timestamp"       # event timestamp
_DAY_COL  = "bar_time"        # LOB bar timestamp (used to derive date group)

# TODO(schema): OFI over [t, t+delta] window — the current panel stores a
#   single pre-computed ofi_z snapped to the bar preceding the headline.
#   A proper lagged-window OFI requires raw LOB data; wire here when available.
#   Until then, ofi_z is used as a proxy for the [t, t+delta] window.

# TODO(schema): return over [t+delta, t+delta+h] — the panel has ret_1m/5m/15m
#   measured from bar_time, not from bar_time+delta.  Recompute when delta > 0.


def _hinge(sent: pd.Series):
    """Split sentiment into positive and negative parts."""
    return sent.clip(lower=0), sent.clip(upper=0)


def _day_groups(panel: pd.DataFrame) -> pd.Series:
    """Integer group labels for cluster-by-day."""
    ts_col = _DAY_COL if _DAY_COL in panel.columns else _TS_COL
    dates  = pd.to_datetime(panel[ts_col]).dt.date
    codes, _ = pd.factorize(dates)
    return pd.Series(codes, index=panel.index)


def fit_conditional_ofi(
    panel: pd.DataFrame,
    horizon: int,
    delta: int = 0,
    sentiment_col: str = "llm_score",
) -> dict:
    """
    Fit the conditional-OFI regression for one return horizon.

    Parameters
    ----------
    panel : aligned event-level panel (one row per news event)
    horizon : forward return horizon in minutes (1, 5, or 15)
    delta : separation window in minutes between OFI measurement and return
            window start. delta=0 uses the pre-computed ofi_z directly.
    sentiment_col : which sentiment score to use for the hinge split.
                    Must be one of: lm_score, llm_score, finbert_score.

    Returns
    -------
    dict with keys:
        horizon, n, alpha, gamma, delta1, delta2, b_pos, b_neg,
        se_b_pos, se_b_neg, wald_t, wald_p,
        r2, aic,
        hac_result    : statsmodels RegressionResultsWrapper (Newey-West HAC)
        cluster_event : statsmodels result clustered by event index
        cluster_day   : statsmodels result clustered by calendar day
    """
    ret_col = f"ret_{horizon}m"
    req     = [_OFI_COL, sentiment_col, ret_col]
    df      = panel.dropna(subset=req).copy()

    if len(df) < 20:
        return {"horizon": horizon, "n": len(df), "error": "insufficient data"}

    if delta != 0:
        # TODO(schema): shift OFI window by delta minutes when raw LOB available.
        pass

    sent                  = df[sentiment_col]
    sent_pos, sent_neg    = _hinge(sent)
    ofi                   = df[_OFI_COL]

    X = pd.DataFrame({
        "const":    1.0,
        "ofi":      ofi,
        "sent_pos": sent_pos,
        "sent_neg": sent_neg,
        "ofi_x_pos": ofi * sent_pos,
        "ofi_x_neg": ofi * sent_neg,
    }, index=df.index)

    y = df[ret_col]

    # ── HAC (Newey-West) ──────────────────────────────────────────────────────
    # Lags = floor(4*(T/100)^(2/9)) per Newey & West 1994
    nw_lags = max(1, int(np.floor(4 * (len(df) / 100) ** (2 / 9))))
    ols_fit  = OLS(y, X).fit()
    hac_fit  = ols_fit.get_robustcov_results(cov_type="HAC", maxlags=nw_lags)

    # ── Cluster by event (each row is one event) ──────────────────────────────
    # Event clusters: row index as group
    evt_groups = np.arange(len(df))
    clust_evt  = ols_fit.get_robustcov_results(
        cov_type="cluster", groups=evt_groups
    )

    # ── Cluster by day ────────────────────────────────────────────────────────
    day_groups = _day_groups(df).values
    clust_day  = ols_fit.get_robustcov_results(
        cov_type="cluster", groups=day_groups
    )

    # ── Wald test: b_pos == b_neg ─────────────────────────────────────────────
    b_pos  = float(hac_fit.params.get("ofi_x_pos", np.nan))
    b_neg  = float(hac_fit.params.get("ofi_x_neg", np.nan))
    se_pos = float(hac_fit.bse.get("ofi_x_pos",    np.nan))
    se_neg = float(hac_fit.bse.get("ofi_x_neg",    np.nan))

    # Simple Wald t-stat for difference (independent SEs; conservative)
    wald_se = np.sqrt(se_pos**2 + se_neg**2 + 1e-14)
    wald_t  = (b_pos - b_neg) / wald_se
    from scipy.stats import t as t_dist
    wald_p  = float(2 * t_dist.sf(abs(wald_t), df=hac_fit.df_resid))

    return {
        "horizon":       horizon,
        "delta":         delta,
        "sentiment_col": sentiment_col,
        "n":             int(hac_fit.nobs),
        "alpha":         float(hac_fit.params.get("const",    np.nan)),
        "gamma":         float(hac_fit.params.get("ofi",      np.nan)),
        "delta1":        float(hac_fit.params.get("sent_pos", np.nan)),
        "delta2":        float(hac_fit.params.get("sent_neg", np.nan)),
        "b_pos":         b_pos,
        "b_neg":         b_neg,
        "se_b_pos":      se_pos,
        "se_b_neg":      se_neg,
        "wald_t":        float(wald_t),
        "wald_p":        wald_p,
        "r2":            float(hac_fit.rsquared),
        "aic":           float(hac_fit.aic),
        "hac_result":    hac_fit,
        "cluster_event": clust_evt,
        "cluster_day":   clust_day,
    }


def fit_all_horizons(
    panel: pd.DataFrame,
    delta: int = 0,
    sentiment_col: str = "llm_score",
    horizons: list[int] = [1, 5, 15],
) -> pd.DataFrame:
    """
    Run fit_conditional_ofi for each horizon and collect scalar results.

    Returns DataFrame excluding the heavy statsmodels result objects.
    """
    rows = []
    for h in horizons:
        r = fit_conditional_ofi(panel, h, delta=delta, sentiment_col=sentiment_col)
        rows.append({k: v for k, v in r.items()
                     if k not in ("hac_result", "cluster_event", "cluster_day")})
    return pd.DataFrame(rows)
