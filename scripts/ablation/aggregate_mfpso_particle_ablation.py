#!/usr/bin/env python3
"""Aggregate MF-PSO particle ablation runs and write summary + plots.

Usage:
  python scripts/ablation/aggregate_mfpso_particle_ablation.py
  python scripts/ablation/aggregate_mfpso_particle_ablation.py \
      --ablation-root outputs/mfpso_particle_ablation_four_rooms
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


DEFAULT_PARTICLES = (50, 100, 200, 400, 800)
DEFAULT_SEEDS = (42, 10, 111, 1032)
# Prefer a100 (working partition); keep rtx6000 for optional future runs.
DEVICE_DIRS = ("cpu", "gpu_a100", "gpu_rtx6000")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ablation-root",
        type=Path,
        default=Path("outputs/mfpso_particle_ablation_four_rooms"),
    )
    p.add_argument(
        "--particles",
        type=str,
        default=",".join(str(x) for x in DEFAULT_PARTICLES),
    )
    p.add_argument(
        "--seeds",
        type=str,
        default=",".join(str(x) for x in DEFAULT_SEEDS),
    )
    return p.parse_args()


def _latest_run_dir(device_root: Path, seed: int, particles: int) -> Path | None:
    pattern = (
        f"FourRoomsAversion2D/PSO/seed_{seed}/"
        f"pso_particle_ablation_*_particles{particles}"
    )
    matches = sorted(device_root.glob(pattern))
    if not matches:
        return None
    exp_dir = matches[-1]
    run_dirs = sorted([d for d in exp_dir.iterdir() if d.is_dir()])
    for run_dir in reversed(run_dirs):
        if (run_dir / "metrics.npz").exists() and (
            run_dir / "exploitabilities.npz"
        ).exists():
            return run_dir
    return None


def load_cell(run_dir: Path) -> dict:
    metrics = np.load(run_dir / "metrics.npz")
    expl = np.load(run_dir / "exploitabilities.npz")["exploitabilities"]
    expl = np.asarray(expl, dtype=np.float64).reshape(-1)
    return {
        "path": str(run_dir),
        "runtime_s": float(metrics["runtime_s"]),
        "exploitabilities": expl,
        "final_exploitability": float(expl[-1]),
    }


def discover_cells(
    ablation_root: Path, particles: list[int], seeds: list[int]
) -> list[dict]:
    rows: list[dict] = []
    for device_dir in DEVICE_DIRS:
        device_root = ablation_root / device_dir
        if not device_root.exists():
            continue
        device = "cuda" if device_dir.startswith("gpu") else "cpu"
        for p in particles:
            for seed in seeds:
                run_dir = _latest_run_dir(device_root, seed, p)
                if run_dir is None:
                    rows.append(
                        {
                            "status": "missing",
                            "device": device,
                            "device_dir": device_dir,
                            "num_particles": p,
                            "seed": seed,
                            "path": None,
                            "runtime_s": None,
                            "final_exploitability": None,
                            "exploitabilities": None,
                        }
                    )
                    continue
                cell = load_cell(run_dir)
                rows.append(
                    {
                        "status": "ok",
                        "device": device,
                        "device_dir": device_dir,
                        "num_particles": p,
                        "seed": seed,
                        **cell,
                    }
                )
    return rows


def summarize(rows: list[dict], particles: list[int]) -> list[dict]:
    grouped: dict[tuple[str, int], list[dict]] = defaultdict(list)
    for row in rows:
        if row["status"] != "ok":
            continue
        grouped[(row["device"], row["num_particles"])].append(row)

    summary = []
    for device in ("cpu", "cuda"):
        for p in particles:
            cells = grouped.get((device, p), [])
            if not cells:
                summary.append(
                    {
                        "device": device,
                        "num_particles": p,
                        "n_seeds": 0,
                        "final_exploitability_mean": None,
                        "final_exploitability_std": None,
                        "runtime_s_mean": None,
                        "runtime_s_std": None,
                    }
                )
                continue
            finals = np.array([c["final_exploitability"] for c in cells], dtype=float)
            runtimes = np.array([c["runtime_s"] for c in cells], dtype=float)
            summary.append(
                {
                    "device": device,
                    "num_particles": p,
                    "n_seeds": int(len(cells)),
                    "final_exploitability_mean": float(finals.mean()),
                    "final_exploitability_std": float(finals.std(ddof=0)),
                    "runtime_s_mean": float(runtimes.mean()),
                    "runtime_s_std": float(runtimes.std(ddof=0)),
                    "seeds": sorted(int(c["seed"]) for c in cells),
                }
            )
    return summary


def write_tables(summary_dir: Path, rows: list[dict], summary: list[dict]) -> None:
    summary_dir.mkdir(parents=True, exist_ok=True)

    cells_csv = summary_dir / "cells.csv"
    with cells_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "status",
                "device",
                "device_dir",
                "num_particles",
                "seed",
                "runtime_s",
                "final_exploitability",
                "path",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "status": row["status"],
                    "device": row["device"],
                    "device_dir": row["device_dir"],
                    "num_particles": row["num_particles"],
                    "seed": row["seed"],
                    "runtime_s": row["runtime_s"],
                    "final_exploitability": row["final_exploitability"],
                    "path": row["path"],
                }
            )

    summary_csv = summary_dir / "summary.csv"
    with summary_csv.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "device",
                "num_particles",
                "n_seeds",
                "final_exploitability_mean",
                "final_exploitability_std",
                "runtime_s_mean",
                "runtime_s_std",
            ],
        )
        writer.writeheader()
        for row in summary:
            writer.writerow({k: row.get(k) for k in writer.fieldnames})

    with (summary_dir / "summary.yaml").open("w") as f:
        yaml.safe_dump({"summary": summary, "n_cells": len(rows)}, f, sort_keys=False)


def plot_process_log(rows: list[dict], particles: list[int], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=True)
    for ax, device in zip(axes, ("cpu", "cuda")):
        for p in particles:
            curves = [
                r["exploitabilities"]
                for r in rows
                if r["status"] == "ok"
                and r["device"] == device
                and r["num_particles"] == p
            ]
            if not curves:
                continue
            min_len = min(len(c) for c in curves)
            arr = np.stack([c[:min_len] for c in curves], axis=0)
            # numerical floor for log scale
            arr = np.maximum(arr, 1e-12)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0, ddof=0)
            xs = np.arange(1, min_len + 1)
            ax.plot(xs, mean, label=f"P={p}")
            ax.fill_between(xs, np.maximum(mean - std, 1e-12), mean + std, alpha=0.15)
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_title(f"Process exploitability ({device})")
        ax.grid(True, which="both", alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel(r"Exploitability $\mathcal{E}$ (log)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_final_and_runtime(
    summary: list[dict], particles: list[int], out_final: Path, out_runtime: Path
) -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for device, marker in (("cpu", "o"), ("cuda", "s")):
        xs, ys, yerr = [], [], []
        for p in particles:
            row = next(
                (
                    r
                    for r in summary
                    if r["device"] == device and r["num_particles"] == p
                ),
                None,
            )
            if row is None or row["final_exploitability_mean"] is None:
                continue
            xs.append(p)
            ys.append(max(row["final_exploitability_mean"], 1e-12))
            yerr.append(row["final_exploitability_std"] or 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, capsize=3, label=device)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(particles)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Number of particles P")
    ax.set_ylabel(r"Final exploitability $\mathcal{E}$ (log)")
    ax.set_title("Final exploitability vs particles")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_final, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for device, marker in (("cpu", "o"), ("cuda", "s")):
        xs, ys, yerr = [], [], []
        for p in particles:
            row = next(
                (
                    r
                    for r in summary
                    if r["device"] == device and r["num_particles"] == p
                ),
                None,
            )
            if row is None or row["runtime_s_mean"] is None:
                continue
            xs.append(p)
            ys.append(row["runtime_s_mean"])
            yerr.append(row["runtime_s_std"] or 0.0)
        if xs:
            ax.errorbar(xs, ys, yerr=yerr, marker=marker, capsize=3, label=device)
    ax.set_xscale("log", base=2)
    ax.set_xticks(particles)
    ax.get_xaxis().set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))
    ax.set_xlabel("Number of particles P")
    ax.set_ylabel("Wall-clock runtime (s)")
    ax.set_title("Runtime vs particles")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_runtime, bbox_inches="tight")
    plt.close(fig)


def completeness_report(
    rows: list[dict], particles: list[int], seeds: list[int]
) -> dict:
    # Expect one CPU device_dir and one GPU device_dir (whichever exists).
    present_dirs = sorted({r["device_dir"] for r in rows})
    devices = sorted({r["device"] for r in rows})
    expected = len(devices) * len(particles) * len(seeds)
    ok = sum(1 for r in rows if r["status"] == "ok")
    missing = [
        {
            "device": r["device"],
            "device_dir": r["device_dir"],
            "num_particles": r["num_particles"],
            "seed": r["seed"],
        }
        for r in rows
        if r["status"] != "ok"
    ]
    return {
        "device_dirs": present_dirs,
        "devices": devices,
        "expected_cells": expected,
        "completed_cells": ok,
        "missing_cells": missing,
        "complete": ok == expected and not missing,
    }


def main() -> None:
    args = parse_args()
    particles = [int(x) for x in args.particles.split(",") if x.strip()]
    seeds = [int(x) for x in args.seeds.split(",") if x.strip()]
    ablation_root = args.ablation_root.resolve()
    summary_dir = ablation_root / "summary"

    rows = discover_cells(ablation_root, particles, seeds)
    summary = summarize(rows, particles)
    report = completeness_report(rows, particles, seeds)

    write_tables(summary_dir, rows, summary)
    plot_process_log(rows, particles, summary_dir / "process_exploitability_log.pdf")
    plot_final_and_runtime(
        summary,
        particles,
        summary_dir / "final_exploitability_vs_P.pdf",
        summary_dir / "runtime_vs_P.pdf",
    )

    with (summary_dir / "completeness.json").open("w") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"Wrote summary artifacts under {summary_dir}")
    if not report["complete"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
