from dataclasses import dataclass


@dataclass
class EnvironmentConfig:
    """Environment configuration."""

    name: str = "default_environment"
