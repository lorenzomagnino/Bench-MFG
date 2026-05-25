"""Programmatic helpers for configuring and running BenchMFG experiments."""

from __future__ import annotations

from collections.abc import Sequence

from hydra import compose, initialize_config_module
from omegaconf import DictConfig

from benchmfg.train import run
from benchmfg.utility.create_environment import create_environment
from benchmfg.utility.create_solver import create_solver


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
    """Create a BenchMFG environment from a config or override list."""
    cfg = (
        cfg_or_overrides
        if isinstance(cfg_or_overrides, DictConfig)
        else load_config(cfg_or_overrides)
    )
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
        environment, initial_policy = create_environment(cfg)
    return create_solver(environment, initial_policy, cfg)


def run_experiment(cfg_or_overrides: DictConfig | Sequence[str] | None = None) -> None:
    """Run an experiment from a config or override list."""
    cfg = (
        cfg_or_overrides
        if isinstance(cfg_or_overrides, DictConfig)
        else load_config(cfg_or_overrides)
    )
    run(cfg)
