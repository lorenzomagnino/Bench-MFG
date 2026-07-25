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
                "garnet_seed": garnet["seed"],
                "random_seed": config["experiment"]["random_seed"],
                "algorithm": _algorithm_name(config),
                "runtime_s": float(metrics["runtime_s"]),
                "final_exploitability": float(exploitabilities[-1]),
                "run_dir": str(run_dir),
            }
        )
    return records


def completed_cells(outputs_dir: Path) -> set[tuple]:
    """Return keys of finished cells, for resuming an interrupted sweep."""
    return {
        (
            record["states"],
            record["actions"],
            record["branching_factor"],
            record["algorithm"],
            record["garnet_seed"],
            record["random_seed"],
        )
        for record in scan_runs(outputs_dir)
    }


def aggregate(records: list[dict]) -> list[dict]:
    """Average per-run records over seeds."""
    groups = defaultdict(list)
    for record in records:
        key = (
            record["states"],
            record["actions"],
            record["branching_factor"],
            record["algorithm"],
        )
        groups[key].append((record["runtime_s"], record["final_exploitability"]))

    rows = []
    for (states, actions, branching, algorithm), values in sorted(groups.items()):
        runtimes, final_values = np.asarray(values).T
        rows.append(
            {
                "states": states,
                "actions": actions,
                "branching_factor": branching,
                "algorithm": algorithm,
                "runs": len(values),
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
            mean = row[f"{value}_mean" if value != "runtime" else "runtime_mean_s"]
            std = row[f"{value}_std" if value != "runtime" else "runtime_std_s"]
            parts.append(f"{mean:{fmt}} ± {std:{fmt}} ({row['runs']})")
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
            parts.append(
                f"{row['final_exploitability_mean']:.3g} ± "
                f"{row['final_exploitability_std']:.3g} / "
                f"{row['runtime_mean_s']:.0f}s"
            )
        lines.append("| " + " | ".join(parts) + " |")
    lines.append("")
    return lines


def write_markdown(rows: list[dict], path: Path) -> None:
    """Write the scaling pivots followed by the full per-cell table."""
    lines = ["# MF-Garnet Scaling", ""]
    if rows:
        actions = sorted({row["actions"] for row in rows})
        branching = sorted({row["branching_factor"] for row in rows})
        seeds = sorted({row["runs"] for row in rows})
        lines.append(
            f"Actions: {', '.join(map(str, actions))} | "
            f"Branching factor: {', '.join(map(str, branching))} | "
            f"Seeds per cell: {', '.join(map(str, seeds))}"
        )
        lines.append("")
        lines += _combined_pivot_lines(rows)
        lines += _pivot_lines(
            rows, "final_exploitability", ".4g", "Final exploitability (mean ± std, n)"
        )
        lines += _pivot_lines(
            rows, "runtime", ".1f", "Runtime in seconds (mean ± std, n)"
        )

    headers = [
        "States",
        "Actions",
        "Branching",
        "Algorithm",
        "Runs",
        "Runtime (s)",
        "Final exploitability",
    ]
    lines += ["### All cells", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for row in rows:
        lines.append(
            f"| {row['states']} | {row['actions']} | {row['branching_factor']} | {row['algorithm']} | "
            f"{row['runs']} | {row['runtime_mean_s']:.3f} +/- {row['runtime_std_s']:.3f} | "
            f"{row['final_exploitability_mean']:.6g} +/- {row['final_exploitability_std']:.6g} |"
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
