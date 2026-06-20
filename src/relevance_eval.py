"""
relevance_eval.py — Human-label inter-rater agreement for relevance buckets.

export_label_sample(panel, k=50) writes a stratified sample to
  results/tables/relevance_sample_to_label.csv
  with columns [event_id, stock, headline, model_relevance, human_relevance]
  where human_relevance is left blank for manual annotation.

cohen_kappa(path) computes Cohen's kappa once the human column is filled.
Do NOT fabricate labels.
"""

from pathlib import Path
import numpy as np
import pandas as pd

_OUT = Path("results/tables/relevance_sample_to_label.csv")
_BUCKETS = ["direct", "sector", "macro", "unrelated"]


def export_label_sample(panel: pd.DataFrame, k: int = 50) -> Path:
    """
    Write a stratified sample (approx k/4 per bucket) to CSV for hand-labelling.

    panel must have columns: stock, headline, agent_relevance.
    If agent_relevance is absent, writes an empty-column stub.
    Returns the path written.
    """
    _OUT.parent.mkdir(parents=True, exist_ok=True)

    if "agent_relevance" not in panel.columns:
        print("  [relevance] 'agent_relevance' column absent — writing stub CSV.")
        stub = panel[["stock", "headline"]].head(k).copy()
        stub.insert(0, "event_id", range(len(stub)))
        stub["model_relevance"] = "unknown"
        stub["human_relevance"] = ""
        stub.to_csv(_OUT, index=False)
        print(f"  Saved stub → {_OUT}")
        return _OUT

    per_bucket = max(1, k // len(_BUCKETS))
    frames = []
    for bucket in _BUCKETS:
        sub = panel[panel["agent_relevance"] == bucket]
        n   = min(per_bucket, len(sub))
        if n > 0:
            frames.append(sub.sample(n=n, random_state=42))

    sampled = pd.concat(frames, ignore_index=True).sample(
        frac=1, random_state=0
    ).reset_index(drop=True)

    sampled.insert(0, "event_id", range(len(sampled)))
    out = sampled[["event_id", "stock", "headline", "agent_relevance"]].copy()
    out = out.rename(columns={"agent_relevance": "model_relevance"})
    out["human_relevance"] = ""

    out.to_csv(_OUT, index=False)
    counts = sampled["agent_relevance"].value_counts().to_dict()
    print(f"  Saved {len(out)}-event label sample → {_OUT}")
    print(f"  Bucket counts: {counts}")
    return _OUT


def cohen_kappa(path: str | Path = _OUT) -> float:
    """
    Compute Cohen's kappa between model_relevance and human_relevance.

    Returns NaN if the human_relevance column is not yet filled, with a
    message telling the analyst what to do next.
    """
    df = pd.read_csv(path)
    if "human_relevance" not in df.columns or df["human_relevance"].isna().all():
        print("  [relevance] human_relevance column is empty.")
        print(f"  Fill {path} with labels (direct/sector/macro/unrelated), then re-run.")
        return float("nan")

    filled = df.dropna(subset=["human_relevance"])
    filled = filled[filled["human_relevance"].str.strip() != ""]
    if len(filled) < 5:
        print(f"  Only {len(filled)} human labels — too few to compute kappa.")
        return float("nan")

    model = filled["model_relevance"].values
    human = filled["human_relevance"].values

    labels = sorted(set(model) | set(human))
    n = len(filled)
    label_idx = {l: i for i, l in enumerate(labels)}
    K = len(labels)

    # Confusion matrix
    cm = np.zeros((K, K), dtype=float)
    for m, h in zip(model, human):
        cm[label_idx[m], label_idx[h]] += 1

    p_o = cm.diagonal().sum() / n
    row_sums = cm.sum(axis=1) / n
    col_sums = cm.sum(axis=0) / n
    p_e = float(np.dot(row_sums, col_sums))

    kappa = (p_o - p_e) / (1 - p_e) if (1 - p_e) > 0 else float("nan")
    print(f"  Cohen's κ = {kappa:.3f}  (n={len(filled)})")
    return kappa
