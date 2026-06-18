"""
baselines.py — Classical ML baselines for direction prediction.
LNA-Agent: Lobster News Alpha

fit_baseline(features, labels, model) trains logistic or gradient-boosted
classifiers on the same feature set used by the LLM agent. Trivial anchors
(always-long, random-direction) are included as sanity checks.

Features: headline embedding (sentence-level) + ofi_z + lm_score.
Labels: sign of forward return (1 = up, 0 = flat/down) at a given horizon.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.dummy import DummyClassifier


# ── Embedding ──────────────────────────────────────────────────────────────────

def _embed_headlines(headlines: list[str]) -> np.ndarray:
    """
    Bag-of-characters TF-IDF embedding as a fast, dependency-free fallback.

    For production, swap in a sentence-transformer or OpenAI embedding.
    TODO(schema): wire real embedding once sentence-transformers is added to deps.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer
    vec = TfidfVectorizer(max_features=200, analyzer="word",
                          ngram_range=(1, 2), sublinear_tf=True)
    return vec.fit_transform(headlines).toarray()


def build_features(panel: pd.DataFrame) -> np.ndarray:
    """
    Construct feature matrix: [headline_embedding | ofi_z | lm_score].

    Rows with missing ofi_z or lm_score are returned as NaN rows so the
    caller can align with labels before dropping.
    """
    headlines = panel["headline"].fillna("").tolist()
    emb       = _embed_headlines(headlines)

    ofi_z    = panel["ofi_z"].fillna(0.0).values.reshape(-1, 1)
    lm_score = panel["lm_score"].fillna(0.0).values.reshape(-1, 1)

    return np.hstack([emb, ofi_z, lm_score])


def build_labels(panel: pd.DataFrame, horizon: int = 1) -> np.ndarray:
    """Binary direction label: 1 if ret_{horizon}m > 0, else 0."""
    ret = panel[f"ret_{horizon}m"].fillna(0.0).values
    return (ret > 0).astype(int)


# ── Baselines ─────────────────────────────────────────────────────────────────

_MODELS = {
    "logistic": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(max_iter=1000, C=0.1, random_state=0)),
    ]),
    "gbm": Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    GradientBoostingClassifier(
            n_estimators=100, max_depth=3, learning_rate=0.05, random_state=0
        )),
    ]),
    "always_long": DummyClassifier(strategy="most_frequent"),
    "random":      DummyClassifier(strategy="uniform", random_state=0),
}


def fit_baseline(
    features: np.ndarray,
    labels: np.ndarray,
    model: str = "logistic",
    cv_folds: int = 5,
) -> dict:
    """
    Fit one classifier and return cross-validated accuracy plus IC proxy.

    Parameters
    ----------
    features : (n, d) float array
    labels   : (n,) binary int array
    model    : one of "logistic", "gbm", "always_long", "random"
    cv_folds : number of CV folds

    Returns
    -------
    dict: model, cv_accuracy_mean, cv_accuracy_std, ic_proxy, n
    ic_proxy = Spearman IC of predict_proba[:, 1] vs raw returns (if computable)
    """
    if model not in _MODELS:
        raise ValueError(f"Unknown model '{model}'. Choose from {list(_MODELS)}")

    clf = _MODELS[model]
    mask = ~np.isnan(features).any(axis=1) & ~np.isnan(labels)
    X, y = features[mask], labels[mask]
    n    = int(mask.sum())

    if n < cv_folds * 2:
        return {"model": model, "n": n, "error": "insufficient data"}

    scores = cross_val_score(clf, X, y, cv=cv_folds, scoring="accuracy")
    clf.fit(X, y)

    # IC proxy: predict_proba[:, 1] treated as a signal vs labels
    try:
        proba    = clf.predict_proba(X)[:, 1]
        ic_proxy = float(spearmanr(proba, y)[0])
    except Exception:
        ic_proxy = float("nan")

    return {
        "model":               model,
        "n":                   n,
        "cv_accuracy_mean":    float(scores.mean()),
        "cv_accuracy_std":     float(scores.std()),
        "ic_proxy":            ic_proxy,
    }


def run_all_baselines(
    panel: pd.DataFrame,
    horizon: int = 1,
) -> pd.DataFrame:
    """
    Run all four baselines and return a summary DataFrame.
    """
    features = build_features(panel)
    labels   = build_labels(panel, horizon=horizon)

    rows = []
    for model_name in _MODELS:
        result = fit_baseline(features, labels, model=model_name)
        rows.append(result)

    df = pd.DataFrame(rows)
    print(f"\n  Baseline results (ret_{horizon}m):")
    print(df[["model", "n", "cv_accuracy_mean", "cv_accuracy_std",
              "ic_proxy"]].to_string(index=False))
    return df
