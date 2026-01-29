"""Structured configuration schemas for Zero-order MFG project using Hydra."""

from dataclasses import dataclass


@dataclass
class RunConfig:
    """Hydra run output configuration."""

    dir: str = "outputs/${environment.name}/${algorithm._target_}/seed_${experiment.random_seed}/${experiment.name}"


@dataclass
class SweepConfig:
    """Hydra sweep output configuration."""

    dir: str = "outputs/${environment.name}/${algorithm._target_}/seed_${experiment.random_seed}/${experiment.name}"
    subdir: str = "${hydra:job.num}"
