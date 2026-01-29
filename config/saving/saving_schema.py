"""Structured configuration schemas for Zero-order MFG project using Hydra."""

from dataclasses import dataclass, field

from hydra.core.config_store import ConfigStore


@dataclass
class RunConfig:
    """Hydra run output configuration."""

    dir: str = "outputs/${now:%Y-%m-%d}/${experiment.name}"


@dataclass
class SweepConfig:
    """Hydra sweep output configuration."""

    dir: str = "multirun/${now:%Y-%m-%d}/${experiment.name}"
    subdir: str = "${hydra:job.num}"


@dataclass
class SavingConfig:
    """Saving configuration."""

    name: str = "default"
    is_saved: bool = True
    description: str = "Default experiment"
    run: RunConfig = field(default_factory=RunConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


ConfigStore.instance().store(name="run", node=RunConfig)
ConfigStore.instance().store(name="sweep", node=SweepConfig)
