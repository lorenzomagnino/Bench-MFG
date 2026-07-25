#!/usr/bin/env python3
"""Plot how MF-Garnet algorithms scale with the state-space dimension.

Produces two figures: final exploitability vs S, and wall-clock time vs S.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_scaling import aggregate, scan_runs  # noqa: E402

# Stable style per algorithm family so the two figures read the same way.
STYLE = {
    "PSO": ("#d62728", "o", "-"),
    "OMD": ("#ff7f0e", "s", "-"),
    "DampedFP:pure": ("#1f77b4", "^", "--"),
    "DampedFP:damped": ("#17becf", "v", "--"),
    "DampedFP:fictitious_play": ("#2ca02c", "D", "--"),
    "PI:policy_iteration": ("#9467bd", "P", ":"),
    "PI:smooth_policy_iteration": ("#8c564b", "X", ":"),
    "PI:boltzmann_policy_iteration": ("#e377c2", "*", ":"),
}


def _series(rows: list[dict], algorithm: str, value: str) -> tuple[list, list, list]:
    """Return (states, means, stds) for one algorithm, sorted by state count."""
    picked = sorted(
        (row for row in rows if row["algorithm"] == algorithm),
        key=lambda row: row["states"],
    )
    states = [row["states"] for row in picked]
    means = [row[f"{value}_mean{'_s' if value == 'runtime' else ''}"] for row in picked]
    stds = [row[f"{value}_std{'_s' if value == 'runtime' else ''}"] for row in picked]
    return states, means, stds


def _plot(
    rows: list[dict],
    records: list[dict],
    value: str,
    record_key: str,
    ylabel: str,
    title: str,
    path: Path,
) -> None:
    """Mean line per algorithm, with the individual seed runs overlaid.

    With only a couple of seeds per cell a std band spans decades and hides the
    data, so the raw per-seed points are shown instead.
    """
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    algorithms = sorted({row["algorithm"] for row in rows})
    masked = []

    for algorithm in algorithms:
        states, means, _ = _series(rows, algorithm, value)
        if not states:
            continue
        color, marker, linestyle = STYLE.get(algorithm, ("#444444", "o", "-"))
        means_arr = np.asarray(means, dtype=float)
        states_arr = np.asarray(states, dtype=float)

        # A log axis cannot show non-positive values; exploitability is
        # non-negative in theory, so these are float32 cancellation at ~0.
        good = means_arr > 0
        for state_count, mean in zip(states_arr[~good], means_arr[~good], strict=False):
            masked.append(f"{algorithm} S={int(state_count)} ({mean:.2g})")

        ax.plot(
            states_arr[good],
            means_arr[good],
            marker=marker,
            linestyle=linestyle,
            color=color,
            label=algorithm,
            linewidth=1.8,
            markersize=6,
        )
        seed_states = [r["states"] for r in records if r["algorithm"] == algorithm]
        seed_values = [r[record_key] for r in records if r["algorithm"] == algorithm]
        seed_states = np.asarray(seed_states, dtype=float)
        seed_values = np.asarray(seed_values, dtype=float)
        positive = seed_values > 0
        ax.scatter(
            seed_states[positive],
            seed_values[positive],
            color=color,
            s=12,
            alpha=0.45,
            linewidths=0,
            zorder=1,
        )

    ax.set_xlabel("Number of states $S$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(sorted({row["states"] for row in rows}))
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="both", alpha=0.25, linewidth=0.5)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    if masked:
        fig.text(
            0.01,
            0.01,
            "omitted (<=0, float32 noise floor): " + ", ".join(masked),
            fontsize=7,
            color="#666666",
        )
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="garnet_scaling")
    args = parser.parse_args()

    records = scan_runs(args.outputs_dir)
    rows = aggregate(records)
    if not rows:
        raise SystemExit(f"no Garnet runs found under {args.outputs_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    _plot(
        rows,
        records,
        "final_exploitability",
        "final_exploitability",
        "Final exploitability",
        "MF-Garnet: exploitability vs state-space size",
        args.out_dir / f"{args.prefix}_exploitability.png",
    )
    _plot(
        rows,
        records,
        "runtime",
        "runtime_s",
        "Wall-clock time (s)",
        "MF-Garnet: runtime vs state-space size",
        args.out_dir / f"{args.prefix}_runtime.png",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
