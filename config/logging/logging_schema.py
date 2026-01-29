from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class WandbConfig:
    """Wandb configuration."""

    project: str = "roboml-research"
    entity: Optional[str] = None
    mode: str = "online"  # online, offline, disabled
    tags: List[str] = field(default_factory=list)


@dataclass
class LoggingConfig:
    """Logging configuration."""

    wandb: WandbConfig = field(default_factory=WandbConfig)
