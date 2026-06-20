"""
regression.py — Respecified conditional-OFI regression.
LNA-Agent: Lobster News Alpha

Model:
  r_{t+h} = a + g*OFI + d1*sent_pos + d2*sent_neg
            + b_pos*(OFI*sent_pos) + b_neg*(OFI*sent_neg) + e

Resolved column mapping (from inspection of aligned panel):
  OFI z-score column     : "ofi_z"
  Return columns         : "ret_1m", "ret_5m", "ret_15m"  (suffix = {h}m)
  Primary scorer         : "llm_score"  (robustness: "lm_score", "finbert_score")
  Day grouping column    : "bar_time"   (falls back to "timestamp")
  Ticker column          : "stock"      (not "ticker")

delta=0: ofi_z is pre-computed from the bar immediately preceding the headline,
so the OFI and return windows do not overlap by construction.  No raw-LOB
recompute is needed at delta=0.

delta>0: return window shifts to [bar_time+delta, bar_time+delta+h]; ofi_z
remains the pre-headline bar.  This requires recomputing returns from the raw LOB
midprice — wired when available.
TODO(schema): delta>0 return shift requires raw LOB midprice; not yet wired.

Newey-West HAC SEs (lags per Newey-West 1994).
Cluster-robust variants: by event (each row) and by calendar day.
Wald test: H0: b_pos == b_neg  (tone symmetry).
"""

import numpy as np
import pandas as pd
from scipy.stats import t as t_dist
from statsmodels.regression.linear_model import OLS

# Verified panel column names
_OFI_COL = "ofi_z"
_TS_COL  = "timestamp"
_DAY_COL = "bar_time"


def _hinge(sent: pd.Series) -> tuple[pd.Series, pd.Series]:
    """sent_pos = max(sent, 0), sent_neg = min(sent, 0)."""
    return sent.clip(lower=0), sent.clip(upper=0)


def _day_groups(df: pd.DataFrame) -> np.ndarray:
    """Integer group labels (0-based) for each unique calendar day."""
    col   = _DAY_COL if _DAY_COL in df.columns else _TS_COL
    dates = pd.to_datetime(df[col]).dt.date
    codes, _ = pd.factorize(dates)
    return codes


def fit_conditional_ofi(
    panel: pd.DataFrame,
    horizon: int,
    delta: int = 0,
    scorer: str = "llm_score",
) -> dict:
    """
    Fit the conditional-OFI regression for one return horizon.

    Parameters
    ----------
    panel   : aligned event-level panel (one row per news event).
              Required columns: ofi_z, {scorer}, ret_{horizon}m, bar_time.
    horizon : forward return horizon in minutes (1, 5, or 15).
    delta   : separation window in minutes between OFI measurement and the
              start of the return window.
              delta=0 — uses panel["ofi_z"] and panel["ret_{h}m"] directly;
                        the pre-headline OFI bar and the h-minute return
                        starting at bar_time are non-overlapping by construction.
              delta>0 — return window shifts to [bar_time+delta, bar_time+delta+h].
                        TODO(schema): requires raw LOB midprice recompute.
    scorer  : sentiment column for the hinge split.
              Primary: "llm_score".  Robustness: "lm_score", "finbert_score".

    Returns
    -------
    dict with scalar keys:
        horizon, delta, scorer, n,
        alpha, gamma, delta1, delta2, b_pos, b_neg,
        se_b_pos (HAC), se_b_neg (HAC), wald_t, wald_p,
        r2, aic
    and heavy objects:
        hac_result, cluster_event, cluster_day
        (excluded from fit_all_horizons DataFrame output)
    """
    ret_col = f"ret_{horizon}m"
    req     = [_OFI_COL, scorer, ret_col]
    df      = panel.dropna(subset=req).copy()

    if len(df) < 20:
        return {"horizon": horizon, "delta": delta, "scorer": scorer,
                "n": len(df), "error": "insufficient data"}

    # ── delta handling ────────────────────────────────────────────────────────
    if delta != 0:
        # TODO(schema): shift return window by delta minutes.
        # Requires recomputing log(mid_{t+delta+h} / mid_{t+delta}) from the
        # raw LOB midprice series.  Currently unavailable; delta=0 only.
        raise NotImplementedError(
            f"delta={delta} requires raw LOB midprice recompute — not yet wired. "
            "Use delta=0 with the pre-computed ret_{h}m columns."
        )

    # ── Feature construction ──────────────────────────────────────────────────
    ofi              = df[_OFI_COL]
    sent             = df[scorer]
    sent_pos, sent_neg = _hinge(sent)

    X = pd.DataFrame({
        "const":     1.0,
        "ofi":       ofi,
        "sent_pos":  sent_pos,
        "sent_neg":  sent_neg,
        "ofi_x_pos": ofi * sent_pos,
        "ofi_x_neg": ofi * sent_neg,
    }, index=df.index)

    y = df[ret_col]

    # ── OLS base fit ──────────────────────────────────────────────────────────
    ols_base = OLS(y, X).fit()
    feat     = list(X.columns)

    def _as_series(arr, fallback_index=feat) -> pd.Series:
        """statsmodels may return ndarray or Series; normalise to Series."""
        if isinstance(arr, pd.Series):
            return arr
        return pd.Series(arr, index=fallback_index)

    # ── HAC (Newey-West) — lags per NW 1994: floor(4*(T/100)^(2/9)) ──────────
    nw_lags = max(1, int(np.floor(4 * (len(df) / 100) ** (2 / 9))))
    hac_fit = ols_base.get_robustcov_results(cov_type="HAC", maxlags=nw_lags)
    hac_p   = _as_series(hac_fit.params)
    hac_se  = _as_series(hac_fit.bse)

    # ── Cluster by event ──────────────────────────────────────────────────────
    clust_evt = ols_base.get_robustcov_results(
        cov_type="cluster", groups=np.arange(len(df))
    )

    # ── Cluster by calendar day ───────────────────────────────────────────────
    clust_day = ols_base.get_robustcov_results(
        cov_type="cluster", groups=_day_groups(df)
    )

    # ── Wald test: H0: b_pos == b_neg ─────────────────────────────────────────
    b_pos  = float(hac_p.get("ofi_x_pos", np.nan))
    b_neg  = float(hac_p.get("ofi_x_neg", np.nan))
    se_pos = float(hac_se.get("ofi_x_pos", np.nan))
    se_neg = float(hac_se.get("ofi_x_neg", np.nan))
    wald_t = (b_pos - b_neg) / np.sqrt(se_pos**2 + se_neg**2 + 1e-14)
    wald_p = float(2 * t_dist.sf(abs(wald_t), df=hac_fit.df_resid))

    return {
        "horizon":       horizon,
        "delta":         delta,
        "scorer":        scorer,
        "n":             int(hac_fit.nobs),
        "alpha":         float(hac_p.get("const",    np.nan)),
        "gamma":         float(hac_p.get("ofi",      np.nan)),
        "delta1":        float(hac_p.get("sent_pos", np.nan)),
        "delta2":        float(hac_p.get("sent_neg", np.nan)),
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
    scorer: str = "llm_score",
    horizons: list[int] = [1, 5, 15],
) -> pd.DataFrame:
    """
    Run fit_conditional_ofi for each horizon and return scalar results only.
    """
    rows = []
    for h in horizons:
        r = fit_conditional_ofi(panel, h, delta=delta, scorer=scorer)
        rows.append({k: v for k, v in r.items()
                     if k not in ("hac_result", "cluster_event", "cluster_day")})
    return pd.DataFrame(rows)
