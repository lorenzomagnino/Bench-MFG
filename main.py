"""
Main entry point for Bench-MFG experiments using Hydra for configuration management.
"""

import logging
from pathlib import Path
import sys

# Ensure src/ packages are importable when running directly (without pip install -e .)
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.getLogger("jax._src.xla_bridge").setLevel(logging.WARNING)

from conf.config_schema import MFGConfig  # noqa: E402
from conf.config_utils import print_config_table  # noqa: E402
import hydra  # noqa: E402 we need to set the level of the JAX TPU backend warning to WARNING before importing JAX
import numpy as np  # noqa: E402
from utility.create_environment import create_environment  # noqa: E402
from utility.create_solver import create_solver  # noqa: E402
from utility.main_utils import create_initial_mean_field, train_model  # noqa: E402

log = logging.getLogger(__name__)


@hydra.main(version_base=None, config_path="config", config_name="defaults")
def main(cfg: MFGConfig) -> None:
    """Main execution function with Hydra configuration management."""
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

    log.info("Experiment completed successfully✅")


if __name__ == "__main__":
    main()
