#!/usr/bin/env python3
"""Run a reproducible MF-Garnet algorithm/seed/state matrix."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import os
from pathlib import Path
import shlex
import subprocess
import sys

ALGORITHM_SEEDS = (42, 10, 111, 1032, 999, 1234, 5678, 9012, 3456, 7890)
ALGORITHMS = (
    "pso",
    "omd",
    "dampedfp:pure",
    "dampedfp:damped",
    "dampedfp:fictitious_play",
    "pi:policy_iteration",
    "pi:smooth_policy_iteration",
    "pi:boltzmann_policy_iteration",
)


@dataclass(frozen=True)
class Job:
    index: int
    states: int
    algorithm: str
    garnet_seed: int
    algorithm_seed: int


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", nargs="+", type=int, default=[100])
    parser.add_argument("--actions", type=int, default=6)
    parser.add_argument("--branching-factor", type=int, default=6)
    parser.add_argument("--num-seeds", type=int, default=10)
    parser.add_argument(
        "--algorithm-seeds", nargs="+", type=int, default=ALGORITHM_SEEDS
    )
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--dynamics-structure", default="additive")
    parser.add_argument("--reward-structure", default="multiplicative")
    parser.add_argument("--game-type", default="potential")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--log-dir", type=Path, default=Path("garnet_logs"))
    return parser


def build_jobs(args: argparse.Namespace) -> list[Job]:
    if args.num_seeds < 1:
        raise ValueError("--num-seeds must be positive")
    if args.num_seeds > len(args.algorithm_seeds):
        raise ValueError("--num-seeds exceeds the number of --algorithm-seeds")
    if args.parallel < 1:
        raise ValueError("--parallel must be positive")
    jobs = []
    for states in args.states:
        for algorithm in args.algorithms:
            for seed in range(args.num_seeds):
                jobs.append(
                    Job(len(jobs), states, algorithm, seed, args.algorithm_seeds[seed])
                )
    return jobs


def command_for(job: Job, args: argparse.Namespace) -> list[str]:
    algorithm, variant = (job.algorithm.split(":", 1) + [None])[:2]
    target = {"dampedfp": "damped_fixed_point", "pi": "pi"}.get(algorithm, algorithm)
    device = "cuda" if args.gpu_id is not None else args.device
    overrides = [
        "train",
        "environment=mf_garnet",
        f"algorithm={target}",
        f"device={device}",
        f"environment.num_states={job.states}",
        f"environment.num_actions={args.actions}",
        f"environment.reward.mfgarnet.branching_factor={args.branching_factor}",
        f"environment.reward.mfgarnet.seed={job.garnet_seed}",
        f"environment.reward.mfgarnet.dynamics_structure={args.dynamics_structure}",
        f"environment.reward.mfgarnet.reward_structure={args.reward_structure}",
        f"environment.reward.mfgarnet.game_type={args.game_type}",
        f"experiment.random_seed={job.algorithm_seed}",
        f"experiment.name=garnet_s{job.states}_a{args.actions}_b{args.branching_factor}_{job.algorithm.replace(':', '_')}",
    ]
    if variant is not None:
        overrides.append(
            f"algorithm.{algorithm}.{'lambda_schedule' if algorithm == 'dampedfp' else 'variant'}={variant}"
        )
    if algorithm == "dampedfp":
        overrides.append("algorithm.dampedfp.damped_constant=0.2")
    elif algorithm == "pi":
        overrides.append("algorithm.pi.damped_constant=0.4")
    if args.no_plots:
        overrides.append("experiment.make_plots=false")
    return [sys.executable, "-m", "benchmfg.cli", *overrides]


def _run(job: Job, args: argparse.Namespace) -> None:
    env = os.environ.copy()
    env["BENCH_MFG_OUTPUT_ROOT"] = str(args.output_root.resolve())
    if args.gpu_id is not None:
        if not args.device.startswith("cuda"):
            raise ValueError("--gpu-id requires --device=cuda or --device=cuda:N")
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"job_{job.index:04d}.log"
    with log_path.open("w") as log:
        result = subprocess.run(
            command_for(job, args), env=env, stdout=log, stderr=subprocess.STDOUT
        )
    if result.returncode:
        raise RuntimeError(f"job {job.index} failed; see {log_path}")


def main() -> int:
    args = _parser().parse_args()
    jobs = build_jobs(args)
    if args.job_index is not None:
        if not 0 <= args.job_index < len(jobs):
            raise SystemExit(f"--job-index must be in [0, {len(jobs) - 1}]")
        jobs = [jobs[args.job_index]]

    if args.dry_run:
        for job in jobs:
            print(f"[{job.index}] {shlex.join(command_for(job, args))}")
        print(f"jobs={len(jobs)}")
        return 0

    if args.parallel == 1 or len(jobs) == 1:
        for job in jobs:
            _run(job, args)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(_run, job, args) for job in jobs]
            for future in as_completed(futures):
                future.result()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
