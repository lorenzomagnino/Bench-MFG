#!/usr/bin/env python3
"""Plot how MF-Garnet algorithms scale with the state-space dimension.

Produces two figures: final exploitability vs S, and wall-clock time vs S.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from benchmfg.garnet.aggregate import (
    EXPLOITABILITY_FLOOR,
    aggregate,
    modality,
    scan_runs,
)

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
    is_exploitability = value == "final_exploitability"
    # Values under the float32 resolution mean "solved"; clamp them onto a floor
    # line so they are visible without implying precision we do not have.
    floor = EXPLOITABILITY_FLOOR if is_exploitability else None

    for algorithm in algorithms:
        states, means, _ = _series(rows, algorithm, value)
        if not states:
            continue
        color, marker, linestyle = STYLE.get(algorithm, ("#444444", "o", "-"))
        means_arr = np.asarray(means, dtype=float)
        states_arr = np.asarray(states, dtype=float)
        if floor is not None:
            means_arr = np.maximum(means_arr, floor)
        good = means_arr > 0

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
        if floor is not None:
            seed_values = np.maximum(seed_values, floor)
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
    if floor is not None:
        ax.axhline(floor, color="#888888", linestyle="-.", linewidth=1.1, zorder=0)
        ax.text(
            0.995,
            floor,
            " float32 floor: solved exactly ",
            transform=ax.get_yaxis_transform(),
            ha="right",
            va="bottom",
            fontsize=7,
            color="#666666",
        )
        ax.set_ylim(bottom=floor / 3)
    ax.legend(fontsize=8, loc="center left", bbox_to_anchor=(1.01, 0.5))
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {path}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--out-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--prefix", default="garnet_scaling")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf"], default=["png"])
    parser.add_argument(
        "--modality",
        nargs="+",
        help='only plot these modalities, e.g. "A/M" (default: one figure pair each)',
    )
    args = parser.parse_args(argv)

    records = scan_runs(args.outputs_dir)
    rows = aggregate(records)
    if not rows:
        raise SystemExit(f"no Garnet runs found under {args.outputs_dir}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    labels = sorted({row["modality"] for row in rows})
    if args.modality:
        unknown = set(args.modality) - set(labels)
        if unknown:
            raise SystemExit(f"no runs for modality {sorted(unknown)}; have {labels}")
        labels = [label for label in labels if label in args.modality]

    # One figure pair per modality: overlaying them would put 16 series on one axis.
    for label in labels:
        subset_rows = [row for row in rows if row["modality"] == label]
        subset_records = [rec for rec in records if modality(rec) == label]
        slug = label.replace("/", "")
        for value, record_key, ylabel, what in (
            (
                "final_exploitability",
                "final_exploitability",
                "Final exploitability",
                "exploitability",
            ),
            ("runtime", "runtime_s", "Wall-clock time (s)", "runtime"),
        ):
            for fmt in args.formats:
                _plot(
                    subset_rows,
                    subset_records,
                    value,
                    record_key,
                    ylabel,
                    f"MF-Garnet ({label}): {what} vs state-space size",
                    args.out_dir / f"{args.prefix}_{slug}_{what}.{fmt}",
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
