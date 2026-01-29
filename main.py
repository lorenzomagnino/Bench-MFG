"""
Main entry point for Zero-order MFG experiments using Hydra for configuration management.
"""

import logging

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

import hydra  # noqa: E402 we need to set the level of the JAX TPU backend warning to WARNING before importing JAX
import numpy as np  # noqa: E402

from conf.config_schema import MFGConfig  # noqa: E402
from conf.config_utils import print_config_table  # noqa: E402
from utility.create_environment import create_environment  # noqa: E402
from utility.create_solver import create_solver  # noqa: E402
from utility.plot_results import plot_results  # noqa: E402
from utility.run_training import run_training  # noqa: E402
from utility.save_results import save_results  # noqa: E402
from utility.wandb_logger import (  # noqa: E402
    upload_mean_field_plot,
    upload_policy_plot,
)

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="conf", config_name="defaults")
def main(cfg: MFGConfig) -> None:
    """Main execution function with Hydra configuration management."""
    print_config_table(cfg, style="table")
    np.random.seed(cfg.experiment.random_seed)
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
    results = run_training(solver, cfg)
    optimal_policy, mean_field, exploitabilities, logger = results

    run_id = None
    if cfg.experiment.is_saved:
        run_id = save_results(
            (optimal_policy, mean_field, exploitabilities),
            cfg,
            initial_policy=initial_policy,
            initial_mean_field=initial_mean_field,
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


if __name__ == "__main__":
    main()
