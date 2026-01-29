from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Training hyperparameters configuration."""

    learning_rate: float = 1e-3
    batch_size: int = 32
    num_epochs: int = 100
    optimizer: str = "adam"
    weight_decay: float = 1e-5
