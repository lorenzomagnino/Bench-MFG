"""
Main entry point for Bench-MFG experiments using Hydra for configuration management.
"""

import logging
from pathlib import Path
import sys
import time

# Ensure src/ packages are importable when running directly (without pip install -e .)
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

from conf.config_schema import MFGConfig  # noqa: E402
from conf.config_utils import print_config_table  # noqa: E402
import hydra  # noqa: E402 we need to set the level of the JAX TPU backend warning to WARNING before importing JAX
from hydra.core.hydra_config import HydraConfig  # noqa: E402
from hydra.types import RunMode  # noqa: E402
import numpy as np  # noqa: E402
from utility.create_environment import create_environment  # noqa: E402
from utility.create_solver import create_solver  # noqa: E402
from utility.path_utils import (  # noqa: E402
    get_algorithm_name_with_variant,
    get_output_directory,
)
from utility.plot_results import plot_results  # noqa: E402
from utility.run_training import run_training  # noqa: E402
from utility.save_results import save_results  # noqa: E402
from utility.wandb_logger import (  # noqa: E402
    upload_mean_field_plot,
    upload_policy_plot,
)

log = logging.getLogger(__name__)


def train_model(
    solver, cfg: MFGConfig, initial_policy=None, initial_mean_field=None
) -> None:
    """Handle training mode execution for both single-seed and multi-seed scenarios.

    Args:
        solver: The optimization solver
        cfg: MFGConfig
        initial_policy: Initial policy to save
        initial_mean_field: Initial mean field to save
    """
    log.info(f"Solver created: {cfg.algorithm._target_}")
    t0 = time.perf_counter()
    results = run_training(solver, cfg)
    runtime_s = time.perf_counter() - t0
    optimal_policy, mean_field, exploitabilities, logger = results

    run_id = None
    if cfg.experiment.is_saved:
        run_id = save_results(
            (optimal_policy, mean_field, exploitabilities),
            cfg,
            initial_policy=initial_policy,
            initial_mean_field=initial_mean_field,
            runtime_s=runtime_s,
        )
        log.info(f"Results saved with run ID: {run_id}")

    final_exploitability = exploitabilities[-1] if len(exploitabilities) > 0 else "N/A"
    log.info(f"Final exploitability: {final_exploitability}")
    mean_field_fig, policy_fig = plot_results(
        (optimal_policy, mean_field, exploitabilities), cfg, run_id=run_id
    )
    upload_mean_field_plot(logger, cfg, mean_field_fig, run_id=run_id)
    upload_policy_plot(logger, cfg, policy_fig, run_id=run_id)

    if logger is not None:
        logger.finish()

    if run_id is not None:
        _print_plot_commands(cfg, run_id)


def _print_plot_commands(cfg: MFGConfig, run_id: str) -> None:
    """Print the exact command to visualise the saved results of this run."""
    try:
        is_sweep = HydraConfig.get().mode == RunMode.MULTIRUN
    except Exception:
        is_sweep = False

    sep = "-" * 64

    if is_sweep:
        env_name = cfg.environment.name
        algo_dir = get_algorithm_name_with_variant(cfg)
        print(
            f"\n{sep}\n"
            f"Plot sweep results:\n"
            f"  PYTHONPATH=src python -m utility.plot_sweep {env_name} {algo_dir}\n"
            f"{sep}\n"
        )
    else:
        run_dir = Path(get_output_directory(cfg)) / run_id
        is_grid = cfg.environment.grid.is_grid
        grid_flags = ""
        if is_grid:
            rows, cols = cfg.environment.grid.dimension
            grid_flags = f" --is-grid --grid-rows {rows} --grid-cols {cols}"
        print(
            f"\n{sep}\n"
            f"Plot this run:\n"
            f"  PYTHONPATH=src python -m utility.plot_single_run {run_dir}{grid_flags}\n"
            f"{sep}\n"
        )


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: MFGConfig) -> None:
    """Main execution function with Hydra configuration management."""
    print_config_table(cfg, style="tree")
    np.random.seed(cfg.experiment.random_seed)
    log.info("Using DEVICE: %s", cfg.device)
    environment, initial_policy = create_environment(cfg)

    initial_mean_field = environment.mean_field_by_transition_kernel(
        initial_policy, num_transition_steps=20
    )
    initial_mean_field = initial_mean_field / initial_mean_field.sum()
    solver = create_solver(environment, initial_policy, cfg)
    if cfg.experiment.mode == 1:
        train_model(solver, cfg, initial_policy, initial_mean_field)
    else:
        log.info("Rollout mode not implemented yet")

    log.info("Experiment completed successfully✅")


if __name__ == "__main__":
    main()
