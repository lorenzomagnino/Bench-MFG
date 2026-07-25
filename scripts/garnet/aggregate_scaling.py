#!/usr/bin/env python3
"""Aggregate MF-Garnet run metrics into scaling tables (Markdown + optional CSV).

Every saved run directory is a completed table cell, so this module is also the
source of truth for what the matrix launcher still has to run.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
import csv
from pathlib import Path

import numpy as np
import yaml

# Exploitability is a difference of two value functions of order 1, computed in
# float32, so its absolute resolution is a few ULPs of 1.0 (2^-24 = 6.0e-08).
# Every observed sub-1e-6 result is an exact small multiple of that ULP, i.e. it
# means "zero", not a small number. Report such cells as a floor instead of
# printing digits that carry no information.
EXPLOITABILITY_FLOOR = 1e-6
FLOOR_LABEL = "<1e-6"


def format_exploitability(mean: float, std: float = 0.0) -> str:
    """Render an exploitability, collapsing anything at the float32 floor."""
    if abs(mean) < EXPLOITABILITY_FLOOR:
        return FLOOR_LABEL
    return f"{mean:.3g} ± {std:.2g}"


def _algorithm_name(config: dict) -> str:
    algorithm = config["algorithm"]
    target = algorithm["_target_"]
    if target == "DampedFP":
        return f"DampedFP:{algorithm['dampedfp']['lambda_schedule']}"
    if target == "PI":
        return f"PI:{algorithm['pi']['variant']}"
    return target


def scan_runs(outputs_dir: Path) -> list[dict]:
    """Return one record per completed run found under ``outputs_dir``."""
    records = []
    for metrics_path in Path(outputs_dir).rglob("metrics.npz"):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        exploitability_path = run_dir / "exploitabilities.npz"
        if not config_path.exists() or not exploitability_path.exists():
            continue
        try:
            with config_path.open() as file:
                config = yaml.safe_load(file)
            metrics = np.load(metrics_path)
            exploitabilities = np.load(exploitability_path)["exploitabilities"]
        except Exception:
            # A partially written run directory is simply not a finished cell.
            continue
        if len(exploitabilities) == 0:
            continue
        # outputs/ also holds runs from other environments; only Garnet has mfgarnet.
        env = config.get("environment") or {}
        garnet = (env.get("reward") or {}).get("mfgarnet")
        if garnet is None or "num_states" not in env:
            continue
        records.append(
            {
                "states": env["num_states"],
                "actions": env["num_actions"],
                "branching_factor": garnet["branching_factor"],
                "dynamics_structure": garnet["dynamics_structure"],
                "reward_structure": garnet["reward_structure"],
                "garnet_seed": garnet["seed"],
                "random_seed": config["experiment"]["random_seed"],
                "algorithm": _algorithm_name(config),
                "runtime_s": float(metrics["runtime_s"]),
                "final_exploitability": float(exploitabilities[-1]),
                "run_dir": str(run_dir),
            }
        )
    return records


def modality(record: dict) -> str:
    """Short label for the coupling structures, e.g. "A/M"."""
    return (
        f"{record['dynamics_structure'][0].upper()}/"
        f"{record['reward_structure'][0].upper()}"
    )


def completed_cells(outputs_dir: Path) -> set[tuple]:
    """Return keys of finished cells, for resuming an interrupted sweep.

    The coupling structures are part of the key: an A/M run is not a finished
    M/A cell, even at the same size, algorithm and seed.
    """
    return {
        (
            record["states"],
            record["actions"],
            record["branching_factor"],
            record["dynamics_structure"],
            record["reward_structure"],
            record["algorithm"],
            record["garnet_seed"],
            record["random_seed"],
        )
        for record in scan_runs(outputs_dir)
    }


def aggregate(records: list[dict]) -> list[dict]:
    """Average per-run records over seeds, keeping modalities separate."""
    groups = defaultdict(list)
    for record in records:
        key = (
            record["states"],
            record["actions"],
            record["branching_factor"],
            record["dynamics_structure"],
            record["reward_structure"],
            record["algorithm"],
        )
        groups[key].append((record["runtime_s"], record["final_exploitability"]))

    rows = []
    for key, values in sorted(groups.items()):
        states, actions, branching, dynamics, reward, algorithm = key
        runtimes, final_values = np.asarray(values).T
        rows.append(
            {
                "states": states,
                "actions": actions,
                "branching_factor": branching,
                "dynamics_structure": dynamics,
                "reward_structure": reward,
                "modality": modality(
                    {"dynamics_structure": dynamics, "reward_structure": reward}
                ),
                "algorithm": algorithm,
                "runs": len(values),
                # How many seeds converged to the float32 floor. The distribution is
                # bimodal (solved / not solved), so the mean alone is misleading.
                "seeds_at_floor": int(
                    (np.abs(final_values) < EXPLOITABILITY_FLOOR).sum()
                ),
                "final_exploitability_median": float(np.median(final_values)),
                "runtime_mean_s": runtimes.mean(),
                "runtime_std_s": runtimes.std(ddof=1) if len(values) > 1 else 0.0,
                "final_exploitability_mean": final_values.mean(),
                "final_exploitability_std": final_values.std(ddof=1)
                if len(values) > 1
                else 0.0,
            }
        )
    return rows


def collect(outputs_dir: Path) -> list[dict]:
    """Scan and aggregate in one step."""
    return aggregate(scan_runs(outputs_dir))


def _pivot_lines(rows: list[dict], value: str, fmt: str, title: str) -> list[str]:
    """Render an algorithm x states table for one metric."""
    algorithms = sorted({row["algorithm"] for row in rows})
    states = sorted({row["states"] for row in rows})
    cells = {(row["algorithm"], row["states"]): row for row in rows}

    lines = [f"### {title}", ""]
    header = ["Algorithm"] + [f"S={s}" for s in states]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for algorithm in algorithms:
        parts = [algorithm]
        for state_count in states:
            row = cells.get((algorithm, state_count))
            if row is None:
                parts.append("-")
                continue
            if value == "runtime":
                parts.append(
                    f"{row['runtime_mean_s']:{fmt}} ± {row['runtime_std_s']:{fmt}} "
                    f"({row['runs']})"
                )
                continue
            cell = format_exploitability(
                row["final_exploitability_mean"], row["final_exploitability_std"]
            )
            if row["seeds_at_floor"]:
                cell += f" [{row['seeds_at_floor']}/{row['runs']} at floor]"
            parts.append(f"{cell} ({row['runs']})")
        lines.append("| " + " | ".join(parts) + " |")
    lines.append("")
    return lines


def _combined_pivot_lines(rows: list[dict]) -> list[str]:
    """Algorithm x states table carrying both exploitability and wall-clock time."""
    algorithms = sorted({row["algorithm"] for row in rows})
    states = sorted({row["states"] for row in rows})
    cells = {(row["algorithm"], row["states"]): row for row in rows}

    lines = [
        "### Exploitability and wall-clock time",
        "",
        "Each cell: final exploitability (mean ± std) / wall-clock seconds (mean).",
        f"`{FLOOR_LABEL}` marks cells at the float32 resolution of exploitability, i.e."
        " solved exactly.",
        "`[k/n at floor]` counts how many seeds reached it.",
        "",
    ]
    header = ["Algorithm"] + [f"S={s}" for s in states]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("|" + "---|" * len(header))
    for algorithm in algorithms:
        parts = [algorithm]
        for state_count in states:
            row = cells.get((algorithm, state_count))
            if row is None:
                parts.append("-")
                continue
            cell = format_exploitability(
                row["final_exploitability_mean"], row["final_exploitability_std"]
            )
            if row["seeds_at_floor"]:
                cell += f" [{row['seeds_at_floor']}/{row['runs']}]"
            parts.append(f"{cell} / {row['runtime_mean_s']:.0f}s")
        lines.append("| " + " | ".join(parts) + " |")
    lines.append("")
    return lines


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write per-modality scaling pivots followed by the full per-cell table."""
    lines = ["# MF-Garnet Scaling", ""]
    # One section per modality: the pivots key on (algorithm, states), so mixing
    # A/M and M/A rows in one table would drop cells.
    for label in sorted({row["modality"] for row in rows}):
        subset = [row for row in rows if row["modality"] == label]
        actions = sorted({row["actions"] for row in subset})
        branching = sorted({row["branching_factor"] for row in subset})
        seeds = sorted({row["runs"] for row in subset})
        lines.append(f"## Modality {label}")
        lines.append("")
        lines.append(
            f"Actions: {', '.join(map(str, actions))} | "
            f"Branching factor: {', '.join(map(str, branching))} | "
            f"Seeds per cell: {', '.join(map(str, seeds))}"
        )
        lines.append("")
        lines += _combined_pivot_lines(subset)
        lines += _pivot_lines(
            subset,
            "final_exploitability",
            ".4g",
            "Final exploitability (mean ± std, n)",
        )
        lines += _pivot_lines(
            subset, "runtime", ".1f", "Runtime in seconds (mean ± std, n)"
        )

    headers = [
        "Modality",
        "States",
        "Actions",
        "Branching",
        "Algorithm",
        "Runs",
        "At floor",
        "Runtime (s)",
        "Final exploitability",
        "Median",
    ]
    lines += ["## All cells", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for row in rows:
        lines.append(
            f"| {row['modality']} | {row['states']} | {row['actions']} | {row['branching_factor']} | {row['algorithm']} | "
            f"{row['runs']} | {row['seeds_at_floor']}/{row['runs']} | "
            f"{row['runtime_mean_s']:.3f} +/- {row['runtime_std_s']:.3f} | "
            f"{row['final_exploitability_mean']:.6g} +/- {row['final_exploitability_std']:.6g} | "
            f"{row['final_exploitability_median']:.6g} |"
        )
    path.write_text("\n".join(lines) + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("outputs_dir", type=Path)
    parser.add_argument("--markdown", type=Path, default=Path("garnet_scaling.md"))
    parser.add_argument("--csv", type=Path)
    parser.add_argument(
        "--print", action="store_true", help="also echo the tables to stdout"
    )
    args = parser.parse_args()
    rows = collect(args.outputs_dir)
    write_markdown(rows, args.markdown)
    if args.csv:
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} aggregate rows to {args.markdown}")
    if args.print:
        print()
        print(args.markdown.read_text())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
