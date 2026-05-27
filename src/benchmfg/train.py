"""Packaged Hydra entrypoint for BenchMFG experiments."""

from __future__ import annotations

import logging

import hydra
import numpy as np

from benchmfg.conf.config_schema import MFGConfig
from benchmfg.conf.config_utils import print_config_table
from benchmfg.utility.create_environment import create_environment
from benchmfg.utility.create_solver import create_solver
from benchmfg.utility.main_utils import create_initial_mean_field, train_model

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

log = logging.getLogger(__name__)


def run(cfg: MFGConfig) -> None:
    """Execute one composed BenchMFG experiment configuration."""
    print_config_table(cfg, style="tree")
    np.random.seed(cfg.experiment.random_seed)
    log.info("Using DEVICE: %s", cfg.device)
    environment, initial_policy = create_environment(cfg)
    initial_mean_field = create_initial_mean_field(environment, initial_policy, cfg)
    log.info("Creating solver...")
    solver = create_solver(environment, initial_policy, cfg)
    log.info("Let's train the model...")
    if cfg.experiment.mode == 1:
        train_model(solver, cfg, initial_policy, initial_mean_field)
    else:
        log.info("Rollout mode not implemented yet")

    log.info("Experiment completed successfully")


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: MFGConfig) -> None:
    """Hydra command entrypoint."""
    run(cfg)


if __name__ == "__main__":
    main()
