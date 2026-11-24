"""Programmatic helpers for configuring and running BenchMFG experiments."""

from __future__ import annotations

from collections.abc import Sequence

from hydra import compose, initialize_config_module
import numpy as np
from omegaconf import DictConfig, OmegaConf

from benchmfg.rl import FixedMeanFieldEnv
from benchmfg.train import run
from benchmfg.utility.animation import plot_gif, show_gifs  # noqa: F401
from benchmfg.utility.create_environment import create_environment
from benchmfg.utility.create_solver import create_solver
from benchmfg.utility.MFGPlots import plot_mean_field, plot_policy  # noqa: F401
from benchmfg.utility.plot_primitives import plot_exploitability  # noqa: F401


def load_config(overrides: Sequence[str] | None = None) -> DictConfig:
    """Load the packaged Hydra config.

    Args:
        overrides: Hydra-style overrides, for example
            ``["algorithm=pso", "device=cpu"]``.

    Returns:
        A composed OmegaConf config.
    """
    with initialize_config_module(
        version_base=None,
        config_module="benchmfg.config",
    ):
        return compose(config_name="defaults", overrides=list(overrides or ()))


def make_environment(cfg_or_overrides: DictConfig | Sequence[str] | None = None):
    """Create a BenchMFG environment from a config or override list.

    Seeds NumPy with ``experiment.random_seed`` (as the CLI does), so the
    random initial policy is reproducible.
    """
    cfg = (
        cfg_or_overrides
        if isinstance(cfg_or_overrides, DictConfig)
        else load_config(cfg_or_overrides)
    )
    np.random.seed(cfg.experiment.random_seed)
    return create_environment(cfg)


def make_solver(
    cfg_or_overrides: DictConfig | Sequence[str] | None = None,
    *,
    environment=None,
    initial_policy=None,
):
    """Create a solver from a config or override list."""
    cfg = (
        cfg_or_overrides
        if isinstance(cfg_or_overrides, DictConfig)
        else load_config(cfg_or_overrides)
    )
    if environment is None or initial_policy is None:
        environment, initial_policy = make_environment(cfg)
    return create_solver(environment, initial_policy, cfg)


def make_fixed_mean_field_env(environment, mean_field):
    """Create a Gymnasium env for the fixed-mean-field MDP."""
    return FixedMeanFieldEnv(environment, mean_field)


def run_experiment(cfg_or_overrides: DictConfig | Sequence[str] | None = None) -> None:
    """Run an experiment from a config or override list."""
    cfg = (
        cfg_or_overrides
        if isinstance(cfg_or_overrides, DictConfig)
        else load_config(cfg_or_overrides)
    )
    run(cfg)


def _as_config(cfg_or_overrides: DictConfig | Sequence[str] | None) -> DictConfig:
    if isinstance(cfg_or_overrides, DictConfig):
        return cfg_or_overrides
    return load_config(cfg_or_overrides)


def print_config_algo(
    cfg_or_overrides: DictConfig | Sequence[str] | None = None,
) -> None:
    """Print the algorithm section of a config as YAML."""
    print(OmegaConf.to_yaml(_as_config(cfg_or_overrides).algorithm))


def print_config_env(
    cfg_or_overrides: DictConfig | Sequence[str] | None = None,
) -> None:
    """Print the environment section of a config as YAML."""
    print(OmegaConf.to_yaml(_as_config(cfg_or_overrides).environment))


class History:
    """Record per-iteration solver outputs. Pass as ``solver.eval(logger=...)``.

    Attributes:
        exploitabilities: One float per iteration.
        mean_fields: One ``(N_states,)`` array per iteration.
        policies: One ``(N_states, N_actions)`` array per iteration
            (empty for solvers that do not report it, e.g. PSO).
    """

    def __init__(self) -> None:
        self.exploitabilities: list[float] = []
        self.mean_fields: list[np.ndarray] = []
        self.policies: list[np.ndarray] = []

    def log_iteration(
        self,
        iteration: int,
        exploitability: float,
        mean_field: np.ndarray | None = None,
        policy: np.ndarray | None = None,
    ) -> None:
        self.exploitabilities.append(float(exploitability))
        if mean_field is not None:
            self.mean_fields.append(np.array(mean_field))
        if policy is not None:
            self.policies.append(np.array(policy))
