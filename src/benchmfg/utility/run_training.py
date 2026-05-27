"""Run the training process."""

from benchmfg.conf.config_schema import MFGConfig
from benchmfg.utility.wandb_logger import WandbLogger

_KNOWN_TARGETS = {"DampedFP", "PSO", "OMD", "PI"}


def run_training(solver, cfg: MFGConfig, logger=None):
    """Run the training process.

    Args:
        solver: The optimization solver.
        cfg: MFGConfig
        logger: Optional wandb logger (if None and wandb_enabled, creates new logger)

    Returns:
        Tuple of (optimal_policy, mean_field, exploitabilities, logger)
    """
    if logger is None and cfg.logging.wandb_enabled:
        logger = WandbLogger(cfg)

    algo_target = cfg.algorithm._target_
    if algo_target not in _KNOWN_TARGETS:
        raise ValueError(f"Unknown algorithm target: {algo_target!r}")

    optimal_policy, mean_field, exploitabilities = solver.eval(logger=logger)
    return optimal_policy, mean_field, exploitabilities, logger
