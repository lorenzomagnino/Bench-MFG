"""Public API for BenchMFG."""

from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

from benchmfg.api import (
    History,
    load_config,
    make_environment,
    make_fixed_mean_field_env,
    make_solver,
    plot_exploitability,
    plot_gif,
    plot_mean_field,
    plot_policy,
    print_config_algo,
    print_config_env,
    run_experiment,
)

try:
    __version__ = version("bench-mfg-suite")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "__version__",
    "History",
    "load_config",
    "make_environment",
    "make_fixed_mean_field_env",
    "make_solver",
    "plot_exploitability",
    "plot_gif",
    "plot_mean_field",
    "plot_policy",
    "print_config_algo",
    "print_config_env",
    "run_experiment",
]
