from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Model architecture configuration."""

    name: str = "default_model"
    hidden_dim: int = 128
    num_layers: int = 2
    activation: str = "relu"
