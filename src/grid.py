"""
grid.py — Accumulate secondary IC cells and apply pooled BH FDR correction.

append_cells(rows)  — appends to results/tables/secondary_grid.csv
correct_grid()      — loads, applies BH, writes secondary_grid_fdr.csv
"""

from pathlib import Path
import pandas as pd
from src.stats_rigor import fdr_table

_GRID_PATH = Path("results/tables/secondary_grid.csv")
_FDR_PATH  = Path("results/tables/secondary_grid_fdr.csv")
_COLS      = ["notebook", "cell_id", "stock", "scorer", "horizon",
              "n", "ic", "ci_lo", "ci_hi", "p"]


def append_cells(rows: list[dict]) -> None:
    """
    Append secondary IC cells to the accumulating grid CSV.

    Each row dict should have keys matching _COLS.  Missing keys are filled
    with None.  Duplicate (notebook, cell_id, stock, scorer, horizon) keys
    are silently overwritten on the next correct_grid() call.
    """
    _GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    new_df = pd.DataFrame([{c: r.get(c) for c in _COLS} for r in rows])

    if _GRID_PATH.exists():
        existing = pd.read_csv(_GRID_PATH)
        combined = pd.concat([existing, new_df], ignore_index=True)
    else:
        combined = new_df

    combined.to_csv(_GRID_PATH, index=False)


def correct_grid(q: float = 0.05) -> pd.DataFrame:
    """
    Load the accumulated secondary grid, apply BH FDR at level q.

    Writes secondary_grid_fdr.csv.  Returns the annotated DataFrame.
    Prints which cells survive correction.
    """
    if not _GRID_PATH.exists():
        print("  [grid] No secondary grid found — run notebooks 01/02/04 first.")
        return pd.DataFrame(columns=_COLS + ["p_adj", "reject_fdr"])

    df = pd.read_csv(_GRID_PATH).dropna(subset=["p"])
    df = fdr_table(df, p_col="p", q=q)
    df.to_csv(_FDR_PATH, index=False)

    survivors = df[df["reject_fdr"]]
    print(f"\n  BH FDR correction (q={q}): {len(df)} cells, "
          f"{len(survivors)} survive.")

    if survivors.empty:
        print("  No secondary cells survive FDR correction.")
    else:
        print(survivors[["notebook", "cell_id", "stock", "scorer",
                          "horizon", "ic", "p", "p_adj"]].to_string(index=False))

    return df
