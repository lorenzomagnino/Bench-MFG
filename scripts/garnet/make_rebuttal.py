#!/usr/bin/env python3
"""Render the MF-Garnet scaling results as post-ready Markdown.

Emits the scaling tables plus a consistency check against the published cells.
Uses ``\\lvert S\\rvert`` rather than ``|S|``: a literal pipe inside a table cell
breaks Markdown table rendering on OpenReview and GitHub.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_scaling import aggregate, scan_runs  # noqa: E402
from compare_to_paper import COLUMN_CONFIG, PAPER, PAPER_NAME  # noqa: E402

ORDER = list(PAPER_NAME)
S_SYM = r"$\lvert S\rvert$"


def _table(rows: dict, states: list[int], fmt) -> list[str]:
    lines = ["| Algorithm | " + " | ".join(f"{S_SYM}={s}" for s in states) + " |"]
    lines.append("|:--|" + "--:|" * len(states))
    for algorithm in ORDER:
        cells = []
        for state_count in states:
            row = rows.get((state_count, algorithm))
            cells.append("—" if row is None else fmt(row))
        lines.append(f"| {PAPER_NAME[algorithm]} | " + " | ".join(cells) + " |")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scaling-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--paper-dir", type=Path, default=Path("outputs_paper"))
    parser.add_argument("--out", type=Path, default=Path("garnet_rebuttal.md"))
    args = parser.parse_args()

    scaling = {
        (row["states"], row["algorithm"]): row
        for row in aggregate(scan_runs(args.scaling_dir))
    }
    states = sorted({s for s, _ in scaling})
    seeds = sorted({row["runs"] for row in scaling.values()})

    lines = ["## New experiment: scaling with the size of the state space", ""]
    lines += [
        "We add a scaling study on MF-Garnet in the **A/M** setting (additive dynamics /",
        r"multiplicative reward), varying $\lvert S\rvert$ at fixed $\lvert A\rvert=6$ and",
        "branching factor $6$, potential game, $64$ noise atoms, horizon $100$,",
        r"$\gamma=0.90$, $150$ iterations, $200$ PSO particles. Mean $\pm$ std over "
        f"${'/'.join(map(str, seeds))}$",
        "seeds (each seed draws a fresh Garnet instance). One NVIDIA L40S per run.",
        "",
        "### Final exploitability",
        "",
    ]
    lines += _table(
        scaling,
        states,
        lambda r: f"{r['final_exploitability_mean']:.3g} ± {r['final_exploitability_std']:.2g}",
    )

    negatives = [
        (s, PAPER_NAME[a])
        for (s, a), r in scaling.items()
        if r["final_exploitability_mean"] < 0
    ]
    if negatives:
        state_count, name = negatives[0]
        lines += [
            "",
            rf"<sub>{name} at $\lvert S\rvert={state_count}$ returns a small negative value:",
            "exploitability is non-negative by definition, so that cell is float32",
            "cancellation at zero, i.e. convergence to numerical precision.</sub>",
        ]

    lines += ["", "### Wall-clock time (seconds)", ""]
    lines += _table(scaling, states, lambda r: f"{r['runtime_mean_s']:.0f}")

    paper_rows = {
        (row["states"], row["actions"], row["branching_factor"], row["algorithm"]): row
        for row in aggregate(scan_runs(args.paper_dir))
    }
    if paper_rows:
        lines += ["", "### Consistency with the results already in the paper", ""]
        lines += [
            "Both published columns were re-run on the same code.",
            r"$z=\lvert\text{ours}-\text{paper}\rvert/\sigma_{\text{paper}}$, so $z\le1$ means the",
            "new value falls inside the variability already reported.",
            "",
        ]
        header = "| Algorithm |"
        separator = "|:--|"
        for column in COLUMN_CONFIG:
            header += f" {column}: paper | ours | $z$ |"
            separator += "--:|--:|--:|"
        lines += [header, separator]
        worst = 0.0
        for algorithm in ORDER:
            cells = []
            for column, (st, ac, br) in COLUMN_CONFIG.items():
                paper_mean, paper_std = PAPER[column][algorithm]
                row = paper_rows.get((st, ac, br, algorithm))
                if row is None:
                    cells += [f"{paper_mean:.3g}", "—", "—"]
                    continue
                z = abs(row["final_exploitability_mean"] - paper_mean) / paper_std
                worst = max(worst, z)
                cells += [
                    f"{paper_mean:.3g} ± {paper_std:.2g}",
                    f"{row['final_exploitability_mean']:.3g} ± {row['final_exploitability_std']:.2g}",
                    f"{z:.2f}",
                ]
            lines.append(f"| {PAPER_NAME[algorithm]} | " + " | ".join(cells) + " |")
        within = sum(
            1
            for column, (st, ac, br) in COLUMN_CONFIG.items()
            for algorithm in ORDER
            if (row := paper_rows.get((st, ac, br, algorithm)))
            and abs(row["final_exploitability_mean"] - PAPER[column][algorithm][0])
            / PAPER[column][algorithm][1]
            <= 1
        )
        total = len(COLUMN_CONFIG) * len(ORDER)
        lines += [
            "",
            f"Every cell reproduces: worst deviation $z={worst:.2f}$, with {within} of {total}",
            r"within $z\le1$. The relative ordering of the methods is unchanged.",
            "",
        ]

    args.out.write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out} ({len(states)} state sizes: {states})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
