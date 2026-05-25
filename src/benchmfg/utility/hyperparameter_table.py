"""Generate a hyperparameter search-space table for the paper.

Outputs:
  results/hyperparameter_table.tex   – LaTeX tabular (include directly in paper source)
  results/hyperparameter_table.pdf   – matplotlib-rendered figure

Usage:
    benchmfg hyperparameter_table [--out-dir results/]
"""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# ---------------------------------------------------------------------------
# Table data
# ---------------------------------------------------------------------------

_ALGORITHMS = [
    "PSO",
    "OMD",
    "Damped FP",
    "Fict. Play",
    "FP",
    "PI",
    "Smooth PI",
    "Boltzmann PI",
]

# LaTeX strings for the "Hyperparameter Space" column
_SPACE_LATEX = {
    "PSO": r"$w \in \mathbb{R}^+,\ c_1 \in \mathbb{R}^+,\ c_2 \in \mathbb{R}^+,\ \tau \in \mathbb{R}^+$",
    "OMD": r"$\eta \in \mathbb{R}^+,\ \tau \in \mathbb{R}^+$",
    "Damped FP": r"$\lambda \in [0,1]$",
    "Fict. Play": r"---",
    "FP": r"---",
    "PI": r"---",
    "Smooth PI": r"$\lambda \in [0,1]$",
    "Boltzmann PI": r"$\lambda \in [0,1],\ \tau \in \mathbb{R}^+$",
}

# Plain-text strings for the matplotlib figure column
_SPACE_TEXT = {
    "PSO": "w ∈ ℝ⁺,  c₁ ∈ ℝ⁺,  c₂ ∈ ℝ⁺,  τ ∈ ℝ⁺",
    "OMD": "η ∈ ℝ⁺,  τ ∈ ℝ⁺",
    "Damped FP": "λ ∈ [0,1]",
    "Fict. Play": "—",
    "FP": "—",
    "PI": "—",
    "Smooth PI": "λ ∈ [0,1]",
    "Boltzmann PI": "λ ∈ [0,1],  τ ∈ ℝ⁺",
}

# LaTeX strings for the "Grid Size" column
_GRID_LATEX = {
    "PSO": r"$2 \times 3 \times 3 \times 2 = 36$",
    "OMD": r"$3 \times 3 = 9$",
    "Damped FP": r"$3$",
    "Fict. Play": r"$1$",
    "FP": r"$1$",
    "PI": r"$1$",
    "Smooth PI": r"$3$",
    "Boltzmann PI": r"$3 \times 3 = 9$",
}

_GRID_TEXT = {
    "PSO": "2×3×3×2 = 36",
    "OMD": "3×3 = 9",
    "Damped FP": "3",
    "Fict. Play": "1",
    "FP": "1",
    "PI": "1",
    "Smooth PI": "3",
    "Boltzmann PI": "3×3 = 9",
}

_NO_HP = {"Fict. Play", "FP", "PI"}


# ---------------------------------------------------------------------------
# LaTeX export
# ---------------------------------------------------------------------------


def _build_latex_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "Hyperparameter Space": {a: _SPACE_LATEX[a] for a in _ALGORITHMS},
        }
    )


def save_latex(out_path: Path) -> None:
    df = _build_latex_df()
    df.index.name = "Algorithm"
    latex_str = df.to_latex(escape=False, column_format="ll", index_names=False)
    # Patch: pandas leaves the index column header blank; fill it in
    latex_str = latex_str.replace(
        " & Hyperparameter Space",
        "Algorithm & Hyperparameter Space",
        1,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(latex_str)
    print(f"Saved LaTeX  → {out_path}")


# ---------------------------------------------------------------------------
# Matplotlib figure export
# ---------------------------------------------------------------------------

_COL_HEADER = "#D0D8E8"  # light blue-grey for header
_ROW_DEFAULT = "#FFFFFF"  # white for normal rows
_ROW_NO_HP = "#F5F5F5"  # very light grey for no-hyperparameter rows


def save_figure(out_path: Path) -> None:
    col_labels = ["Algorithm", "Hyperparameter Space"]
    cell_text = [[a, _SPACE_TEXT[a]] for a in _ALGORITHMS]

    fig, ax = plt.subplots(figsize=(13, 3.6))
    ax.axis("off")
    fig.patch.set_facecolor("white")

    tbl = ax.table(
        cellText=cell_text,
        colLabels=col_labels,
        loc="center",
        cellLoc="left",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 1.6)

    # Style header row
    for col in range(len(col_labels)):
        cell = tbl[0, col]
        cell.set_facecolor(_COL_HEADER)
        cell.set_text_props(fontweight="bold", ha="center")

    # Style body rows
    for row_idx, algo in enumerate(_ALGORITHMS, start=1):
        bg = _ROW_NO_HP if algo in _NO_HP else _ROW_DEFAULT
        for col in range(len(col_labels)):
            cell = tbl[row_idx, col]
            cell.set_facecolor(bg)
            cell.set_edgecolor("#CCCCCC")

    # Widen columns: algo name narrow, space wide
    col_widths = [0.18, 0.65]
    for col, w in enumerate(col_widths):
        for row in range(len(_ALGORITHMS) + 1):
            tbl[row, col].set_width(w)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", dpi=300, facecolor="white")
    plt.close(fig)
    print(f"Saved figure → {out_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=str,
        default="results",
        help="Output directory (default: results/)",
    )
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    save_latex(out_dir / "hyperparameter_table.tex")
    save_figure(out_dir / "hyperparameter_table.pdf")
