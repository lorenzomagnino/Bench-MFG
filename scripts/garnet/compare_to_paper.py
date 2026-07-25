#!/usr/bin/env python3
"""Compare freshly computed MF-Garnet cells against the published table.

Regression check: run the matching configuration into its own output root, then
run this to confirm the numbers still land inside the paper's seed variance.

Column labels follow ``SxAxB (dynamics/reward)``, e.g. ``5x5x5 (A/M)`` is
num_states=5, num_actions=5, branching_factor=5, additive dynamics,
multiplicative reward.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_scaling import aggregate, scan_runs  # noqa: E402

# (mean, std) as printed in the paper.
PAPER = {
    "5x5x5 (A/M)": {
        "DampedFP:pure": (1.6537, 3.2993),
        "DampedFP:damped": (1.4504, 2.8451),
        "DampedFP:fictitious_play": (0.6235, 1.6547),
        "PI:boltzmann_policy_iteration": (1.0570, 1.8618),
        "PI:smooth_policy_iteration": (2.3228, 4.8665),
        "PI:policy_iteration": (1.8707, 3.5905),
        "OMD": (2.3106, 3.0778),
        "PSO": (0.2250, 0.2115),
    },
    "25x10x10 (A/A)": {
        "DampedFP:pure": (9.50e-04, 0.0017),
        "DampedFP:damped": (8.51e-04, 0.0017),
        "DampedFP:fictitious_play": (0.0026, 0.0036),
        "PI:boltzmann_policy_iteration": (0.9508, 0.2879),
        "PI:smooth_policy_iteration": (9.80e-04, 0.0026),
        "PI:policy_iteration": (0.0030, 0.0046),
        "OMD": (1.4371, 0.3936),
        "PSO": (3.8633, 1.9042),
    },
}
# Which (states, actions, branching) each paper column corresponds to.
COLUMN_CONFIG = {"5x5x5 (A/M)": (5, 5, 5), "25x10x10 (A/A)": (25, 10, 10)}
# Paper's name for each of our algorithm keys.
PAPER_NAME = {
    "DampedFP:pure": "Fixed Point (FP)",
    "DampedFP:damped": "Damped FP",
    "DampedFP:fictitious_play": "Fictitious Play",
    "PI:boltzmann_policy_iteration": "Boltzmann PI",
    "PI:smooth_policy_iteration": "Smooth PI",
    "PI:policy_iteration": "PI",
    "OMD": "OMD",
    "PSO": "MF-PSO",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--markdown", type=Path)
    args = parser.parse_args()

    rows = {
        (row["states"], row["actions"], row["branching_factor"], row["algorithm"]): row
        for row in aggregate(scan_runs(args.outputs_dir))
    }

    lines = ["# MF-Garnet: new results vs published table", ""]
    lines.append("`z` = |ours - paper| / paper_std. z<=1 is inside the paper's own")
    lines.append("seed spread; z>2 would be a real discrepancy.")
    lines.append("")
    worst = 0.0
    for column, entries in PAPER.items():
        states, actions, branching = COLUMN_CONFIG[column]
        lines += [f"## {column}", ""]
        lines.append("| Algorithm | Paper | Ours | z | |")
        lines.append("|---|---|---|---|---|")
        for algorithm, (paper_mean, paper_std) in entries.items():
            row = rows.get((states, actions, branching, algorithm))
            if row is None:
                lines.append(
                    f"| {PAPER_NAME[algorithm]} | {paper_mean:.4g} | missing | - | |"
                )
                continue
            mean = row["final_exploitability_mean"]
            std = row["final_exploitability_std"]
            z = abs(mean - paper_mean) / paper_std if paper_std else float("inf")
            worst = max(worst, z)
            verdict = "ok" if z <= 1 else ("borderline" if z <= 2 else "**DIFFERS**")
            lines.append(
                f"| {PAPER_NAME[algorithm]} | {paper_mean:.4g} ± {paper_std:.3g} | "
                f"{mean:.4g} ± {std:.3g} (n={row['runs']}) | {z:.2f} | {verdict} |"
            )
        lines.append("")
    lines.append(f"Worst deviation: z = {worst:.2f}")
    lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.markdown:
        args.markdown.write_text(text)
        print(f"wrote {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
