"""Helpers used by the main experiment entrypoint."""

import logging
from pathlib import Path
import time

from conf.config_schema import MFGConfig
from envs.mfg_model_class_jit import (
    get_jax_device,
    mean_field_by_transition_kernel_multi_jax,
)
from hydra.core.hydra_config import HydraConfig
from hydra.types import RunMode
import jax
import numpy as np
from utility.create_solver import ENV_JIT_FUNCTIONS, get_env_spec
from utility.path_utils import get_algorithm_name_with_variant, get_output_directory
from utility.plot_results import plot_results
from utility.run_training import run_training
from utility.save_results import save_results
from utility.wandb_logger import upload_mean_field_plot, upload_policy_plot

log = logging.getLogger(__name__)

_JAX_ALGORITHM_TARGETS = {"OMD", "DampedFP", "PI", "PSO"}


def create_initial_mean_field(
    environment,
    initial_policy: np.ndarray,
    cfg: MFGConfig,
) -> np.ndarray:
    """Create the initial mean field saved alongside a run.

    JAX-backed algorithms use the existing JAX transition-kernel helper to avoid
    the Python multiprocessing startup cost, while preserving the current
    20-transition-step semantics.
    """
    log.info("Creating initial mean field...")
    if (
        cfg.algorithm._target_ in _JAX_ALGORITHM_TARGETS
        and cfg.environment.name in ENV_JIT_FUNCTIONS
    ):
        env_spec = get_env_spec(environment, cfg.environment.name)
        jax_device = get_jax_device(cfg.device)
        initial_mean_field = mean_field_by_transition_kernel_multi_jax(
            jax.device_put(initial_policy, jax_device),
            env_spec,
            num_iterations=20,
            initial_mean_field=jax.device_put(
                environment.stationary_mean_field,
                jax_device,
            ),
        )
        return np.asarray(initial_mean_field) / np.asarray(initial_mean_field).sum()

    initial_mean_field = environment.mean_field_by_transition_kernel(
        initial_policy,
        num_transition_steps=20,
    )
    return initial_mean_field / initial_mean_field.sum()


def train_model(
    solver,
    cfg: MFGConfig,
    initial_policy=None,
    initial_mean_field=None,
) -> None:
    """Handle training mode execution for both single-seed and multi-seed scenarios."""
    log.info("Solver created: %s", cfg.algorithm._target_)
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
        log.info("Results saved with run ID: %s", run_id)

    final_exploitability = exploitabilities[-1] if len(exploitabilities) > 0 else "N/A"
    log.info("Final exploitability: %s", final_exploitability)
    mean_field_fig, policy_fig = plot_results(
        (optimal_policy, mean_field, exploitabilities),
        cfg,
        run_id=run_id,
    )
    upload_mean_field_plot(logger, cfg, mean_field_fig, run_id=run_id)
    upload_policy_plot(logger, cfg, policy_fig, run_id=run_id)

    if logger is not None:
        logger.finish()

    if run_id is not None:
        print_plot_commands(cfg, run_id)


def print_plot_commands(cfg: MFGConfig, run_id: str) -> None:
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
        return

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
