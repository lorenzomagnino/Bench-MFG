#!/usr/bin/env python3
"""Run a reproducible MF-Garnet algorithm/seed/state matrix.

Each job is one cell of the scaling table. Finished runs are detected on startup,
so an interrupted sweep resumes instead of recomputing. Use --no-resume to force
a full re-run.
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
import os
from pathlib import Path
import queue
import shlex
import subprocess
import sys
import threading
import time

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aggregate_scaling import collect, completed_cells, write_markdown  # noqa: E402

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
# Table label for each launcher algorithm key, matching aggregate_scaling naming.
TABLE_NAMES = {
    "pso": "PSO",
    "omd": "OMD",
    "dampedfp:pure": "DampedFP:pure",
    "dampedfp:damped": "DampedFP:damped",
    "dampedfp:fictitious_play": "DampedFP:fictitious_play",
    "pi:policy_iteration": "PI:policy_iteration",
    "pi:smooth_policy_iteration": "PI:smooth_policy_iteration",
    "pi:boltzmann_policy_iteration": "PI:boltzmann_policy_iteration",
}

_print_lock = threading.Lock()
# Populated from --gpu-ids: a queue of GPU indices, one held per running job.
_GPU_POOL: queue.Queue | None = None


def log(message: str) -> None:
    """Timestamped, thread-safe progress line."""
    with _print_lock:
        print(f"[{datetime.now():%H:%M:%S}] {message}", flush=True)


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
    parser.add_argument("--num-seeds", type=int, default=2)
    parser.add_argument(
        "--algorithm-seeds", nargs="+", type=int, default=ALGORITHM_SEEDS
    )
    parser.add_argument("--algorithms", nargs="+", default=ALGORITHMS)
    parser.add_argument("--dynamics-structure", default="additive")
    parser.add_argument("--reward-structure", default="multiplicative")
    parser.add_argument("--game-type", default="potential")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--gpu-id", type=int)
    parser.add_argument(
        "--gpu-ids",
        nargs="+",
        type=int,
        help="run one job per listed GPU (implies --device=cuda and --parallel=len)",
    )
    parser.add_argument("--parallel", type=int, default=1)
    parser.add_argument("--job-index", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--output-root", type=Path, default=Path("outputs"))
    parser.add_argument("--log-dir", type=Path, default=Path("garnet_logs"))
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="re-run cells that already have saved results",
    )
    parser.add_argument(
        "--job-timeout",
        type=float,
        default=3600.0,
        help="kill a cell after this many seconds and treat it as failed "
        "(JAX can deadlock on a loaded host; without this one hung cell "
        "blocks the whole sweep). 0 disables.",
    )
    parser.add_argument(
        "--table",
        type=Path,
        default=Path("garnet_scaling.md"),
        help="scaling table refreshed after each finished cell",
    )
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


def cell_key(job: Job, args: argparse.Namespace) -> tuple:
    """Key matching aggregate_scaling.completed_cells for resume checks.

    Includes the coupling structures, so a finished A/M cell does not make the
    launcher skip the corresponding M/A job.
    """
    return (
        job.states,
        args.actions,
        args.branching_factor,
        args.dynamics_structure,
        args.reward_structure,
        TABLE_NAMES.get(job.algorithm, job.algorithm),
        job.garnet_seed,
        job.algorithm_seed,
    )


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


def _run(job: Job, args: argparse.Namespace, total: int) -> float:
    env = os.environ.copy()
    env["BENCH_MFG_OUTPUT_ROOT"] = str(args.output_root.resolve())
    if args.gpu_id is not None:
        if not args.device.startswith("cuda"):
            raise ValueError("--gpu-id requires --device=cuda or --device=cuda:N")
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)

    # One job per GPU: a worker holds its GPU for the duration of the job. At S=400
    # a single job reserves ~40 GB of a 46 GB card, so they cannot be packed.
    gpu = _GPU_POOL.get() if _GPU_POOL is not None else None
    if gpu is not None:
        env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.log_dir / f"job_{job.index:04d}.log"

    where = f" gpu={gpu}" if gpu is not None else ""
    log(
        f"start  {job.index + 1}/{total}  S={job.states} {job.algorithm} "
        f"seed={job.garnet_seed}{where}"
    )
    started = time.perf_counter()
    timeout = args.job_timeout or None
    try:
        with log_path.open("w") as handle:
            try:
                result = subprocess.run(
                    command_for(job, args),
                    env=env,
                    stdout=handle,
                    stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            except subprocess.TimeoutExpired:
                elapsed = time.perf_counter() - started
                log(
                    f"TIMEOUT {job.index + 1}/{total}  S={job.states} {job.algorithm} "
                    f"seed={job.garnet_seed} after {elapsed:.0f}s -- killed, will retry"
                )
                raise RuntimeError(
                    f"job {job.index} timed out; see {log_path}"
                ) from None
    finally:
        if gpu is not None:
            _GPU_POOL.put(gpu)
    elapsed = time.perf_counter() - started
    if result.returncode:
        log(
            f"FAILED {job.index + 1}/{total}  S={job.states} {job.algorithm} "
            f"seed={job.garnet_seed} after {elapsed:.0f}s -- see {log_path}"
        )
        raise RuntimeError(f"job {job.index} failed; see {log_path}")
    log(
        f"done   {job.index + 1}/{total}  S={job.states} {job.algorithm} "
        f"seed={job.garnet_seed} in {elapsed:.0f}s"
    )
    return elapsed


def refresh_table(args: argparse.Namespace) -> int:
    """Rewrite the scaling table from whatever cells exist so far."""
    rows = collect(args.output_root)
    if rows:
        write_markdown(rows, args.table)
    return len(rows)


def main() -> int:
    global _GPU_POOL
    args = _parser().parse_args()
    if args.gpu_ids:
        # Each worker owns one GPU, so concurrency is fixed by the GPU count.
        args.device = "cuda"
        args.parallel = len(args.gpu_ids)
        _GPU_POOL = queue.Queue()
        for gpu in args.gpu_ids:
            _GPU_POOL.put(gpu)
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

    total = len(jobs)
    if not args.no_resume:
        done = completed_cells(args.output_root)
        pending = [job for job in jobs if cell_key(job, args) not in done]
        if len(pending) != total:
            log(f"resume: {total - len(pending)}/{total} cells already saved")
        jobs = pending

    if not jobs:
        log("nothing to run; all cells present")
        log(f"table has {refresh_table(args)} rows -> {args.table}")
        return 0

    log(f"running {len(jobs)} cells, parallel={args.parallel}, device={args.device}")
    started = time.perf_counter()
    failures = 0

    if args.parallel == 1 or len(jobs) == 1:
        for job in jobs:
            try:
                _run(job, args, total)
            except RuntimeError:
                failures += 1
            refresh_table(args)
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as pool:
            futures = [pool.submit(_run, job, args, total) for job in jobs]
            for future in as_completed(futures):
                try:
                    future.result()
                except RuntimeError:
                    failures += 1
                refresh_table(args)

    elapsed = time.perf_counter() - started
    rows = refresh_table(args)
    log(
        f"sweep finished in {elapsed / 60:.1f} min; {failures} failed; "
        f"table has {rows} rows -> {args.table}"
    )
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
