"""Structured configuration schemas for Zero-order MFG project using Hydra."""

from dataclasses import dataclass, field
from typing import List


@dataclass
class AversionConfig:
    alpha: float = 1.0
    epsilon: float = 1e-12


@dataclass
class GridConfig:
    """Grid environment configuration."""

    is_grid: bool = False
    dimension: list[int] = field(default_factory=lambda: [5, 5])
    walls_enabled: bool = False


@dataclass
class WallsConfig:
    """Wall configuration for obstacle layouts."""

    row: int = 5
    col: int = 5


@dataclass
class ObstaclesConfig:
    type: str = "four_rooms_cross"
    walls: WallsConfig = field(default_factory=WallsConfig)
    doors: List[List[int]] = field(
        default_factory=lambda: [[2, 5], [7, 5], [5, 7], [5, 2]]
    )


@dataclass
class DynamicsConfig:
    """Dynamics configuration for chain environments."""

    is_noisy: bool = True
    noise_probabilities: List[float] = field(default_factory=lambda: [0.2, 0.6, 0.2])


@dataclass
class LasryLionsConfig:
    """Lasry-Lion configuration."""

    crowd_penalty_coefficient: float = 2.0
    movement_penalty: float = 0.1
    center_attraction: float = 0.5


@dataclass
class NoInteractionConfig:
    """No Interaction configuration."""

    movement_penalty: float = 0.1


@dataclass
class StrictContractionConfig:
    alpha: float = 1.0
    beta: float = 1.0
    movement_penalty: float = 1.0
    targets: List[int] = field(default_factory=lambda: [2, 8])


@dataclass
class MFGarnetConfig:
    seed: int = 0
    branching_factor: int = 5

    dynamics_structure: str = "additive"  # "additive" | "multiplicative"
    cp: float = 0.5
    rho_p: float = 0.5

    reward_structure: str = "additive"  # "additive" | "multiplicative"
    cr: float = 0.5
    rho_r: float = 0.5
    game_type: str = "potential"  # "potential" | "cyclic"

    reward_scale: float = 1.0
    eps: float = 1e-12
    relu_eps: float = 0.0


@dataclass
class SISEpidemicConfig:
    """SIS Epidemic Model configuration."""

    beta: float = 0.5  # Transmission rate (β > 0)
    nu: float = 0.1  # Recovery rate (0 < ν < 1)
    cost_infection: float = 1.0  # Cost of being infected (C > 0)


@dataclass
class KineticCongestionConfig:
    """Kinetic Congestion / Crowd Dynamics configuration."""

    target_state: int = 0  # Target state index (0-indexed)
    movement_cost: float = 0.1  # Cost for moving (c_move > 0)
    capacity_threshold: float = (
        0.4  # Maximum allowed density per cell (0 < threshold <= 1)
    )


@dataclass
class ContractionGameConfig:
    """Contraction Game configuration.

    Reward: r(x, a, μ) = -C · I(a=Switch) - α μ(x)
    If C > α/(1-γ), the best response is unique (always Stay).
    """

    switching_cost: float = 10.0  # C: cost for switching states
    congestion_coefficient: float = 1.0  # α: congestion penalty coefficient


@dataclass
class RewardConfig:
    lasry_lions: LasryLionsConfig = field(default_factory=LasryLionsConfig)
    no_interaction: NoInteractionConfig = field(default_factory=NoInteractionConfig)
    aversion: AversionConfig = field(default_factory=AversionConfig)
    strict_contraction: StrictContractionConfig = field(
        default_factory=StrictContractionConfig
    )
    mfgarnet: MFGarnetConfig = field(default_factory=MFGarnetConfig)
    sis_epidemic: SISEpidemicConfig = field(default_factory=SISEpidemicConfig)
    kinetic_congestion: KineticCongestionConfig = field(
        default_factory=KineticCongestionConfig
    )
    contraction_game: ContractionGameConfig = field(
        default_factory=ContractionGameConfig
    )


@dataclass
class InitialDistributionConfig:
    """Initial distribution configuration."""

    type: str = "uniform"  # uniform, concentrated, custom
    concentration_state: int = 5
    concentration_ratio: float = 0.8
    custom_values: list[int] | None = None
