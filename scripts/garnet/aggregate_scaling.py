#!/usr/bin/env python3
"""Aggregate MF-Garnet run metrics into a Markdown and optional CSV table."""

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


def collect(outputs_dir: Path) -> list[dict]:
    groups = defaultdict(list)
    for metrics_path in outputs_dir.rglob("metrics.npz"):
        run_dir = metrics_path.parent
        config_path = run_dir / "config.yaml"
        exploitability_path = run_dir / "exploitabilities.npz"
        if not config_path.exists() or not exploitability_path.exists():
            continue
        with config_path.open() as file:
            config = yaml.safe_load(file)
        metrics = np.load(metrics_path)
        exploitabilities = np.load(exploitability_path)["exploitabilities"]
        env = config["environment"]
        garnet = env["reward"]["mfgarnet"]
        key = (
            env["num_states"],
            env["num_actions"],
            garnet["branching_factor"],
            _algorithm_name(config),
        )
        groups[key].append((float(metrics["runtime_s"]), float(exploitabilities[-1])))

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


def write_markdown(rows: list[dict], path: Path) -> None:
    headers = [
        "States",
        "Actions",
        "Branching",
        "Algorithm",
        "Runs",
        "Runtime (s)",
        "Final exploitability",
    ]
    lines = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
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
    args = parser.parse_args()
    rows = collect(args.outputs_dir)
    write_markdown(rows, args.markdown)
    if args.csv:
        with args.csv.open("w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys() if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    print(f"wrote {len(rows)} aggregate rows to {args.markdown}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
