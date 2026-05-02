"""Plot all results from a single experiment run.

Usage:
    PYTHONPATH=src python -m utility.plot_single_run <run_dir> [options]
    PYTHONPATH=src python src/utility/plot_single_run.py <run_dir> [options]

Given a run directory (the timestamped folder inside outputs/), this script plots:
  - Exploitability curve
  - Final mean field distribution
  - Final policy
  - Prints wall-clock runtime

Example:
    PYTHONPATH=src python -m utility.plot_single_run \\
        outputs/LasryLionsChain/OMD/seed_42/bench_mfg_lr0p0050_temp0p20/20260502_175255_575
"""

import argparse
from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from utility.plot_loaders import load_runtime  # noqa: E402
from utility.plot_primitives import (  # noqa: E402
    plot_exploitability_from_npz,
    plot_mean_field_from_npz,
    plot_policy_from_npz,
)


def plot_run(
    run_dir: str | Path,
    is_grid: bool = False,
    grid_dim: tuple | None = None,
    walls=None,
) -> None:
    """Plot all result artifacts from a single run directory.

    Args:
        run_dir: Path to the timestamped run directory containing NPZ files.
        is_grid: Whether the environment is a 2D grid.
        grid_dim: (n_rows, n_cols) for grid environments.
        walls: Optional wall mask array for grid environments.
        Saves both linear and log-scale exploitability plots.
    """
    run_dir = Path(run_dir)

    exploitability_npz = run_dir / "exploitabilities.npz"
    if exploitability_npz.exists():
        plot_exploitability_from_npz(
            exploitability_npz,
            fn=run_dir / "plots" / "exploitability.pdf",
            log_scale=False,
        )
        plot_exploitability_from_npz(
            exploitability_npz,
            fn=run_dir / "plots" / "exploitability_log.pdf",
            log_scale=True,
        )
        print(f"Exploitability → {run_dir / 'plots' / 'exploitability.pdf'}")
        print(f"Exploitability → {run_dir / 'plots' / 'exploitability_log.pdf'}")
    else:
        print(f"Skipping exploitability: {exploitability_npz} not found")

    mean_field_npz = run_dir / "final_mean_field.npz"
    if mean_field_npz.exists():
        plot_mean_field_from_npz(
            mean_field_npz, is_grid=is_grid, grid_dim=grid_dim, walls=walls
        )
        print(f"Mean field     → {run_dir / 'plots' / 'mean_field.pdf'}")
    else:
        print(f"Skipping mean field: {mean_field_npz} not found")

    policy_npz = run_dir / "final_policy.npz"
    if policy_npz.exists():
        plot_policy_from_npz(
            policy_npz, is_grid=is_grid, grid_dim=grid_dim, walls=walls
        )
        print(f"Policy         → {run_dir / 'plots' / 'policy.pdf'}")
    else:
        print(f"Skipping policy: {policy_npz} not found")

    metrics_npz = run_dir / "metrics.npz"
    if metrics_npz.exists():
        runtime = load_runtime(metrics_npz)
        print(f"Runtime        → {runtime:.3f}s")
    else:
        print(f"Skipping runtime: {metrics_npz} not found")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot exploitability, mean field, and policy from a single run directory."
    )
    parser.add_argument(
        "run_dir", type=str, help="Path to the timestamped run directory"
    )
    parser.add_argument(
        "--is-grid", action="store_true", help="Environment is a 2D grid"
    )
    parser.add_argument(
        "--grid-rows", type=int, default=None, help="Number of grid rows (2D only)"
    )
    parser.add_argument(
        "--grid-cols", type=int, default=None, help="Number of grid cols (2D only)"
    )
    args = parser.parse_args()

    grid_dim = (
        (args.grid_rows, args.grid_cols)
        if (args.is_grid and args.grid_rows and args.grid_cols)
        else None
    )

    plot_run(
        run_dir=args.run_dir,
        is_grid=args.is_grid,
        grid_dim=grid_dim,
    )
