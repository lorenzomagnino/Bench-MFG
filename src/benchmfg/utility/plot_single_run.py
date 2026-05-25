"""Plot mean field and policy from a single run or from a results/{env}/{algo} directory.

Two usage modes:

  1. Direct run directory (original behaviour):
       benchmfg plot single-run <run_dir> [options]

  2. Environment + algorithm from results/:
       benchmfg plot single-run --env <ENV> --algo <ALGO> [options]

     Reads  results/{ENV}/{ALGO}/final_mean_field.npz
            results/{ENV}/{ALGO}/final_policy.npz
     Writes results/{ENV}/{ALGO}/mean_field.pdf
            results/{ENV}/{ALGO}/policy.pdf

Examples:
    benchmfg plot single-run --env LasryLionsChain --algo OMD
    benchmfg plot single-run --env FourRoomsAversion2D --algo PSO
    benchmfg plot single-run \\
        outputs/LasryLionsChain/OMD/seed_42/omd_sweep_lr0p0500_temp0p80/20260503_133412_762_job5
"""

import argparse
from pathlib import Path
import sys

import numpy as np

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from benchmfg.utility.plot_loaders import load_runtime  # noqa: E402
from benchmfg.utility.plot_primitives import (  # noqa: E402
    plot_exploitability_from_npz,
    plot_mean_field_from_npz,
    plot_policy_from_npz,
)

# Grid metadata for known 2-D environments
_GRID_ENVS: dict[str, tuple[int, int]] = {
    "FourRoomsAversion2D": (11, 11),
    "KineticCongestion": (5, 5),
}


def _build_four_rooms_walls(
    n_rows: int = 11,
    n_cols: int = 11,
    doors: tuple = ((2, 5), (8, 5), (5, 8), (5, 2)),
) -> np.ndarray:
    """Return flat walls array (1=free, 0=wall) for FourRoomsAversion2D."""
    mask = np.ones(n_rows * n_cols, dtype=float)
    mid_row, mid_col = n_rows // 2, n_cols // 2
    door_set = set(doors)
    for row in range(n_rows):
        if (row, mid_col) not in door_set:
            mask[row * n_cols + mid_col] = 0.0
    for col in range(n_cols):
        if (mid_row, col) not in door_set:
            mask[mid_row * n_cols + col] = 0.0
    for row, col in door_set:
        if 0 <= row < n_rows and 0 <= col < n_cols:
            mask[row * n_cols + col] = 1.0
    return mask


def _get_walls(environment: str) -> np.ndarray | None:
    if environment == "FourRoomsAversion2D":
        return _build_four_rooms_walls()
    return None


def _grid_meta(
    environment: str, is_grid: bool, grid_rows: int | None, grid_cols: int | None
):
    """Return (is_grid, grid_dim) for an environment, auto-filling known 2-D environments."""
    if is_grid and grid_rows and grid_cols:
        return True, (grid_rows, grid_cols)
    if environment in _GRID_ENVS:
        return True, _GRID_ENVS[environment]
    return is_grid, None


def plot_run(
    run_dir: str | Path,
    is_grid: bool = False,
    grid_dim: tuple | None = None,
    walls=None,
) -> None:
    """Plot all result artifacts from a single timestamped run directory."""
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


def plot_best_for_algo(
    environment: str,
    algo: str,
    is_grid: bool = False,
    grid_dim: tuple | None = None,
    walls=None,
) -> None:
    """Plot mean field and policy from results/{environment}/{algo}/."""
    project_root = Path.cwd()
    algo_dir = project_root / "results" / environment / algo

    if not algo_dir.exists():
        available = [
            d.name
            for d in (project_root / "results" / environment).iterdir()
            if d.is_dir()
        ]
        print(f"Directory not found: {algo_dir}")
        print(f"Available algorithms for {environment}: {sorted(available)}")
        sys.exit(1)

    for npz_name in ["final_mean_field.npz", "final_policy.npz"]:
        if not (algo_dir / npz_name).exists():
            print(
                f"Missing {npz_name} in {algo_dir} — run collect_best_artifacts first."
            )
            sys.exit(1)

    mf_out = algo_dir / "mean_field.pdf"
    plot_mean_field_from_npz(
        npz_path=algo_dir / "final_mean_field.npz",
        is_grid=is_grid,
        grid_dim=grid_dim,
        walls=walls,
        fn=mf_out,
    )
    print(f"Mean field → {mf_out}")

    pol_out = algo_dir / "policy.pdf"
    plot_policy_from_npz(
        npz_path=algo_dir / "final_policy.npz",
        is_grid=is_grid,
        grid_dim=grid_dim,
        walls=walls,
        fn=pol_out,
    )
    print(f"Policy     → {pol_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot mean field and policy from a run directory or results/{env}/{algo}."
    )

    # Mode 1: direct run directory
    parser.add_argument(
        "run_dir",
        type=str,
        nargs="?",
        default=None,
        help="Path to a timestamped run directory (outputs/.../YYYYMMDD_...)",
    )

    # Mode 2: results directory
    parser.add_argument("--env", type=str, default=None, help="Environment name")
    parser.add_argument(
        "--algo",
        type=str,
        default=None,
        help="Algorithm directory name (e.g. OMD, PSO, DampedFP_damped)",
    )

    # Grid options (auto-filled for known 2-D environments)
    parser.add_argument(
        "--is-grid", action="store_true", help="Override: treat as 2D grid"
    )
    parser.add_argument("--grid-rows", type=int, default=None)
    parser.add_argument("--grid-cols", type=int, default=None)

    args = parser.parse_args()

    if args.run_dir is None and args.env is None:
        parser.print_help()
        sys.exit(1)

    if args.run_dir is not None:
        env_name = (
            Path(args.run_dir).parts[1] if len(Path(args.run_dir).parts) > 1 else ""
        )
        is_grid, grid_dim = _grid_meta(
            env_name, args.is_grid, args.grid_rows, args.grid_cols
        )
        walls = _get_walls(env_name)
        plot_run(run_dir=args.run_dir, is_grid=is_grid, grid_dim=grid_dim, walls=walls)
    else:
        if args.algo is None:
            project_root = Path.cwd()
            env_path = project_root / "results" / args.env
            if env_path.exists():
                available = sorted(d.name for d in env_path.iterdir() if d.is_dir())
                print(f"Available algorithms for {args.env}: {available}")
            else:
                print(f"Environment not found in results/: {args.env}")
            sys.exit(1)

        is_grid, grid_dim = _grid_meta(
            args.env, args.is_grid, args.grid_rows, args.grid_cols
        )
        walls = _get_walls(args.env)
        plot_best_for_algo(
            environment=args.env,
            algo=args.algo,
            is_grid=is_grid,
            grid_dim=grid_dim,
            walls=walls,
        )
