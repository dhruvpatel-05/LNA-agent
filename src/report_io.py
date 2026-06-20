"""
report_io.py — Unified figure and table saving for the LNA-Agent report.

save_fig(fig, name)     → results/figures/{name}.pdf
save_table(df, name)    → results/tables/{name}.csv + results/tables/{name}.tex
"""

from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd

_FIGURES = Path("results/figures")
_TABLES  = Path("results/tables")

# Navy / steel palette matching the beamer slides
_PALETTE = ["#1A3A52", "#5E8FAA", "#2E86AB", "#A23B72", "#F18F01", "#C73E1D"]


def setup_style() -> None:
    """Apply the report's minimal matplotlib style (call once per notebook)."""
    plt.rcParams.update({
        "figure.dpi":         120,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.prop_cycle":    plt.cycler("color", _PALETTE),
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.labelsize":     10,
        "legend.fontsize":    9,
        "figure.autolayout":  True,
    })


def save_fig(fig: "plt.Figure", name: str) -> Path:
    """Save figure as PDF to results/figures/{name}.pdf."""
    _FIGURES.mkdir(parents=True, exist_ok=True)
    path = _FIGURES / f"{name}.pdf"
    fig.savefig(path, bbox_inches="tight")
    print(f"  Saved figure → {path}")
    return path


def save_table(df: pd.DataFrame, name: str, caption: str = "", label: str = "") -> Path:
    """
    Save DataFrame to CSV and a booktabs LaTeX table.

    Returns the CSV path.
    """
    _TABLES.mkdir(parents=True, exist_ok=True)
    csv_path = _TABLES / f"{name}.csv"
    tex_path = _TABLES / f"{name}.tex"

    df.to_csv(csv_path, index=False)

    label_str   = label or f"tab:{name}"
    caption_str = caption or name.replace("_", " ").title()

    tex = df.to_latex(
        index=False,
        float_format="%.4f",
        escape=True,
        longtable=False,
        column_format="l" + "r" * (len(df.columns) - 1),
    )
    # Inject booktabs and caption
    tex = tex.replace(r"\begin{tabular}", r"\begin{table}[htbp]" + "\n"
                      + r"\centering" + "\n"
                      + f"\\caption{{{caption_str}}}" + "\n"
                      + f"\\label{{{label_str}}}" + "\n"
                      + r"\begin{tabular}")
    tex = tex.replace(r"\end{tabular}", r"\end{tabular}" + "\n" + r"\end{table}")
    tex_path.write_text(tex)

    print(f"  Saved table  → {csv_path} + {tex_path}")
    return csv_path
