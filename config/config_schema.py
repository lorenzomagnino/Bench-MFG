"""Configuration schemas using dataclasses for type safety."""

from dataclasses import dataclass, field

from config.environment.environment_schema import EnvironmentConfig
from config.logging.logging_schema import LoggingConfig
from config.model.model_schema import ModelConfig
from config.saving.saving_schema import SavingConfig
from config.training.training_schema import TrainingConfig


@dataclass
class Config:
    """Root configuration combining all sub-configs."""

    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    saving: SavingConfig = field(default_factory=SavingConfig)
    experiment_name: str = "default_experiment"
    seed: int = 42
    device: str = "cpu"  # cpu, cuda
