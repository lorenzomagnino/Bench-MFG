"""Structured configuration schemas for Bench-MFG project using Hydra."""

from dataclasses import dataclass, field
from typing import Optional

from conf.algorithm.algorithm_schema import (
    DampedFPConfig,
    OMDConfig,
    PIConfig,
    PSOConfig,
)
from conf.environment.environemnt_schema import (
    DynamicsConfig,
    GridConfig,
    InitialDistributionConfig,
    ObstaclesConfig,
    RewardConfig,
)
from conf.experiment.experiment_scheme import RunConfig, SweepConfig
from conf.visualization.visualization_schema import ColorsConfig


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    name: str = "default"
    grid: GridConfig = field(default_factory=GridConfig)
    dynamics: DynamicsConfig = field(default_factory=DynamicsConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    initial_distribution: InitialDistributionConfig = field(
        default_factory=InitialDistributionConfig
    )
    obstacles: ObstaclesConfig = field(default_factory=ObstaclesConfig)
    num_actions: int = 3
    num_noises: int = 3
    num_states: int = 10
    horizon: int = 100
    gamma: float = 0.90


@dataclass
class AlgorithmConfig:
    """Algorithm configuration."""

    _target_: str = "PSO"
    pso: PSOConfig = field(default_factory=PSOConfig)
    dampedfp: DampedFPConfig = field(default_factory=DampedFPConfig)
    omd: OMDConfig = field(default_factory=OMDConfig)
    pi: PIConfig = field(default_factory=PIConfig)


@dataclass
class JobConfig:
    """Hydra job configuration."""

    name: str = "mfg_experiment"


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""

    name: str = "default_experiment"
    mode: int = 1
    is_saved: bool = False
    random_seed: int = 42
    description: str = "Default configuration for Bench-MFG experiments"
    run: RunConfig = field(default_factory=RunConfig)
    sweep: SweepConfig = field(default_factory=SweepConfig)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    level: str = "INFO"
    wandb_enabled: bool = False
    wandb_project: str = "bench-mfg"
    wandb_entity: Optional[str] = None
    wandb_log_interval: int = 1  # Log every N iterations (1 = every iteration)


@dataclass
class VisualizationConfig:
    """Visualization settings."""

    show_mean_field_evolution: bool = True
    show_policy_evolution: bool = True
    colors: ColorsConfig = field(default_factory=ColorsConfig)


@dataclass
class InitializationConfig:
    """Policy initialization configuration."""

    initialization_type: str = "PSO_uniform"
    init_policy_temp: float = 0.2


@dataclass
class MFGConfig:
    """
    Main configuration for Bench-MFG experiments.
    This configuration class aggregates all other configuration classes and provides a unified interface for the Bench-MFG project.
    Args:
        environment: Environment configuration.
        algorithm: Algorithm configuration.
        experiment: Experiment configuration.
        logging: Logging configuration.
        visualization: Visualization configuration.
        initialization: Policy initialization configuration.
    """

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    initialization: InitializationConfig = field(default_factory=InitializationConfig)
