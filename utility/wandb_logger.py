"""Wandb logging utilities for Zero-order MFG experiments."""

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from omegaconf import OmegaConf

from conf.config_schema import MFGConfig
from utility.path_utils import get_output_directory

log = logging.getLogger(__name__)


def compute_mean_field_stats(mean_field: np.ndarray) -> dict:
    """Compute statistics for mean field distribution.

    Args:
        mean_field: Mean field distribution array

    Returns:
        Dictionary with mean field statistics
    """
    mean_field = np.asarray(mean_field)
    mean_field_flat = mean_field.flatten()

    epsilon = 1e-12
    safe_mean_field = np.clip(mean_field_flat, epsilon, 1.0)
    entropy = -np.sum(safe_mean_field * np.log(safe_mean_field))

    stats = {
        "mean_field_entropy": float(entropy),
        "mean_field_max": float(np.max(mean_field_flat)),
        "mean_field_min": float(np.min(mean_field_flat)),
        "mean_field_std": float(np.std(mean_field_flat)),
    }
    return stats


class WandbLogger:
    """Wandb logger for tracking MFG experiment metrics."""

    def __init__(
        self, cfg: MFGConfig, seed: Optional[int] = None, is_summary: bool = False
    ) -> None:
        """Initialize wandb logger.

        Args:
            cfg: MFG configuration
            seed: Optional seed value for multi-seed runs
            is_summary: Whether this is a summary run for multi-seed experiments
        """
        self.cfg = cfg
        self.seed = seed
        self.is_summary = is_summary
        self.enabled = cfg.logging.wandb_enabled
        self.log_interval = cfg.logging.wandb_log_interval

        if not self.enabled:
            return

        try:
            import os

            import wandb

            api_key = os.environ.get("WANDB_API_KEY")
            if api_key:
                api_key_stripped = api_key.strip()
                if api_key != api_key_stripped:
                    os.environ["WANDB_API_KEY"] = api_key_stripped
                    log.info("Stripped whitespace from WANDB_API_KEY")
            run_name = f"{cfg.environment.name}_{cfg.experiment.name}_{cfg.algorithm._target_.lower()}"
            algo_target = cfg.algorithm._target_.lower()
            hyperparams = []

            if algo_target == "omd":
                lr = cfg.algorithm.omd.learning_rate
                temp = cfg.algorithm.omd.temperature
                num_iter = cfg.algorithm.omd.num_iterations
                hyperparams.append(f"lr{lr:.4f}".replace(".", "p"))
                hyperparams.append(f"temp{temp:.2f}".replace(".", "p"))
                hyperparams.append(f"iter{num_iter}")
            elif algo_target == "pso":
                w = cfg.algorithm.pso.w
                c1 = cfg.algorithm.pso.c1
                c2 = cfg.algorithm.pso.c2
                temp = cfg.algorithm.pso.temperature
                num_particles = cfg.algorithm.pso.num_particles
                num_iter = cfg.algorithm.pso.num_iterations
                policy_type = cfg.algorithm.pso.policy_type
                hyperparams.append(f"w{w:.2f}".replace(".", "p"))
                hyperparams.append(f"c1{c1:.2f}".replace(".", "p"))
                hyperparams.append(f"c2{c2:.2f}".replace(".", "p"))
                hyperparams.append(f"temp{temp:.2f}".replace(".", "p"))
                hyperparams.append(f"particles{num_particles}")
                hyperparams.append(f"iter{num_iter}")
                hyperparams.append(policy_type)
            elif algo_target == "dampedfp":
                num_iter = cfg.algorithm.dampedfp.num_iterations
                variant = cfg.algorithm.dampedfp.lambda_schedule
                hyperparams.append(f"iter{num_iter}")
                hyperparams.append(variant)
                if cfg.algorithm.dampedfp.damped_constant is not None:
                    damped = cfg.algorithm.dampedfp.damped_constant
                    hyperparams.append(f"damped{damped:.2f}".replace(".", "p"))
            elif algo_target == "pi":
                num_iter = cfg.algorithm.pi.num_iterations
                temp = cfg.algorithm.pi.temperature
                variant = cfg.algorithm.pi.variant
                hyperparams.append(f"iter{num_iter}")
                hyperparams.append(f"temp{temp:.2f}".replace(".", "p"))
                hyperparams.append(variant)
                if cfg.algorithm.pi.damped_constant is not None:
                    damped = cfg.algorithm.pi.damped_constant
                    hyperparams.append(f"damped{damped:.2f}".replace(".", "p"))

            if hyperparams:
                run_name = f"{run_name}_{'_'.join(hyperparams)}"

            if seed is not None:
                run_name = f"{run_name}_seed_{seed}"
            elif self.is_summary:
                run_name = f"{run_name}_summary"
            else:
                run_name = f"{run_name}_seed_{cfg.experiment.random_seed}"

            init_kwargs = {
                "project": cfg.logging.wandb_project,
                "name": run_name,
                "config": self._config_to_dict(cfg),
                "tags": [cfg.algorithm._target_, cfg.environment.name],
                "mode": "online",
            }

            if cfg.logging.wandb_entity is not None:
                init_kwargs["entity"] = cfg.logging.wandb_entity
                log.info(f"Using WandB entity: {cfg.logging.wandb_entity}")

            if seed is not None:
                init_kwargs["tags"].append(f"seed_{seed}")
            elif self.is_summary:
                init_kwargs["tags"].append("summary")

            log.info(f"Initializing WandB with project: {cfg.logging.wandb_project}")
            wandb.init(**init_kwargs)

            if wandb.run is None:
                raise RuntimeError("wandb.init() completed but wandb.run is None")

            self.wandb = wandb
            run_url = wandb.run.url if wandb.run else "N/A"
            run_id = wandb.run.id if wandb.run else "N/A"
            log.info("Wandb logging initialized successfully")
            log.info(f"  Run name: {run_name}")
            log.info(f"  Run ID: {run_id}")
            log.info(f"  Entity: {wandb.run.entity if wandb.run else 'N/A'}")
            log.info(f"  Project: {wandb.run.project if wandb.run else 'N/A'}")
            log.info(f"  View run at: {run_url}")

        except ImportError:
            log.warning("wandb not installed, logging disabled")
            self.enabled = False
        except Exception as e:
            log.error(f"Failed to initialize wandb: {e}", exc_info=True)
            log.error(f"  Project: {cfg.logging.wandb_project}")
            log.error(f"  Entity: {cfg.logging.wandb_entity}")
            log.error("  Check that:")
            log.error("    1. WANDB_API_KEY environment variable is set")
            log.error("    2. You have access to the entity/team")
            log.error("    3. The entity name is correct")
            self.enabled = False

    def _config_to_dict(self, cfg: MFGConfig) -> dict:
        """Convert Hydra config to dictionary for wandb.

        Args:
            cfg: MFG configuration

        Returns:
            Dictionary representation of config
        """
        config_dict = OmegaConf.to_container(cfg, resolve=True)
        if isinstance(config_dict, dict):
            return config_dict
        return {}

    def log_iteration(
        self,
        iteration: int,
        exploitability: float,
        mean_field: Optional[np.ndarray] = None,
    ) -> None:
        """Log metrics at each iteration.

        Args:
            iteration: Current iteration number
            exploitability: Current exploitability value
            mean_field: Current mean field distribution (not used, kept for compatibility)
        """
        if not self.enabled:
            return

        if iteration != 0 and iteration % self.log_interval != 0:
            return

        metrics = {
            "train/exploitability": float(exploitability),
        }

        self.wandb.log(metrics, step=iteration)

    def log_config(self, cfg: MFGConfig) -> None:
        """Log configuration to wandb.

        Args:
            cfg: MFG configuration
        """
        if not self.enabled:
            return

        config_dict = self._config_to_dict(cfg)
        self.wandb.config.update(config_dict)

    def finish(self) -> None:
        """Finish wandb run."""
        if not self.enabled:
            return

        try:
            self.wandb.finish()
            log.info("Wandb run finished")
        except Exception as e:
            log.warning(f"Error finishing wandb run: {e}")


def upload_mean_field_plot(
    logger: Optional[WandbLogger],
    cfg: MFGConfig,
    fig=None,
    run_id: Optional[str] = None,
) -> None:
    """Upload mean field plot to wandb if logger is enabled.

    Args:
        logger: Optional wandb logger instance
        cfg: MFG configuration
        fig: Optional matplotlib figure object. If provided, will upload this figure.
             Otherwise, will look for saved PDF file.
        run_id: Unique run identifier. If provided, looks for plots in run_id subdirectory.
    """
    if logger is None or not logger.enabled:
        return

    try:
        if fig is not None:
            base_dir = Path(get_output_directory(cfg))
            output_dir = base_dir / run_id if run_id is not None else base_dir
            temp_plot_path = output_dir / "plots" / "mean_field_distribution.pdf"
            temp_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(temp_plot_path), bbox_inches="tight", pad_inches=0.1, format="pdf"
            )

            logger.wandb.save(str(temp_plot_path))
            logger.wandb.log({"mean_field_plot": logger.wandb.Image(fig)})
            log.info("Mean field plot uploaded to wandb from figure object")
        else:
            base_dir = Path(get_output_directory(cfg))
            if run_id is not None:
                plots_dir = base_dir / run_id / "plots"
            else:
                plots_dir = base_dir / "plots"
            plot_filename_mean_field = (
                f"{cfg.experiment.name}_{cfg.algorithm._target_.lower()}_mean_field.pdf"
            )
            plot_path_mean_field = plots_dir / plot_filename_mean_field

            if plot_path_mean_field.exists():
                logger.wandb.save(str(plot_path_mean_field))
                log.info(f"Mean field plot uploaded to wandb: {plot_path_mean_field}")
    except Exception as e:
        log.warning(f"Failed to upload mean field plot to wandb: {e}")


def upload_policy_plot(
    logger: Optional[WandbLogger],
    cfg: MFGConfig,
    fig=None,
    run_id: Optional[str] = None,
) -> None:
    """Upload policy plot to wandb if logger is enabled.

    Args:
        logger: Optional wandb logger instance
        cfg: MFG configuration
        fig: Optional matplotlib figure object. If provided, will upload this figure.
             Otherwise, will look for saved PDF file.
        run_id: Unique run identifier. If provided, looks for plots in run_id subdirectory.
    """
    if logger is None or not logger.enabled:
        return

    try:
        if fig is not None:
            base_dir = Path(get_output_directory(cfg))
            output_dir = base_dir / run_id if run_id is not None else base_dir
            temp_plot_path = output_dir / "plots" / "policy_distribution.pdf"
            temp_plot_path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(
                str(temp_plot_path), bbox_inches="tight", pad_inches=0.1, format="pdf"
            )

            logger.wandb.save(str(temp_plot_path))
            logger.wandb.log({"policy_plot": logger.wandb.Image(fig)})
            log.info("Policy plot uploaded to wandb from figure object")
        else:
            base_dir = Path(get_output_directory(cfg))
            if run_id is not None:
                plots_dir = base_dir / run_id / "plots"
            else:
                plots_dir = base_dir / "plots"
            plot_filename_policy = (
                f"{cfg.experiment.name}_{cfg.algorithm._target_.lower()}_policy.pdf"
            )
            plot_path_policy = plots_dir / plot_filename_policy

            if plot_path_policy.exists():
                logger.wandb.save(str(plot_path_policy))
                log.info(f"Policy plot uploaded to wandb: {plot_path_policy}")
    except Exception as e:
        log.warning(f"Failed to upload policy plot to wandb: {e}")
