"""Run the training process."""

from conf.config_schema import AlgorithmConfig, MFGConfig
from utility.wandb_logger import WandbLogger


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

    algo_cfg: AlgorithmConfig = cfg.algorithm
    algo_target = algo_cfg._target_
    if (
        algo_target == "DampedFP"
        or algo_target == "PSO"
        or algo_target == "OMD"
        or algo_target == "PI"
    ):
        optimal_policy, mean_field, exploitabilities = solver.eval(logger=logger)

    return optimal_policy, mean_field, exploitabilities, logger
