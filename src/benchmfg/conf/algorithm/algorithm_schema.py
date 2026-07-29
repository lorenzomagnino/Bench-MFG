"""Structured configuration schemas for Bench-MFG project using Hydra."""

from dataclasses import dataclass
from typing import Literal

from hydra.core.config_store import ConfigStore

# Type aliases for type checking (not used in dataclass fields due to OmegaConf limitations)
LambdaSchedule = Literal["pure", "damped", "fictitious_play"]
PIVariant = Literal[
    "policy_iteration", "smooth_policy_iteration", "boltzmann_policy_iteration"
]


@dataclass
class PSOConfig:
    """PSO algorithm configuration."""

    num_particles: int = 100
    num_iterations: int = 300
    init_policy_temp: float = 1.0  # temperature for initial policy conversion to logits (if init_solution provided)
    temperature: float = (
        1.0  # temperature for boltzmann policy conversion during algorithm iterations
    )
    w: float = 0.4
    c1: float = 0.5
    c2: float = 1.5
    policy_type: str = "boltzmann"
    initialization_type: str = (
        "PSO_uniform"  # "PSO_uniform" or "one_uniform" or "dirichlet"
    )


@dataclass
class OMDConfig:
    """On-line Mirror Descent algorithm configuration."""

    use_python: bool = False
    num_iterations: int = 100
    learning_rate: float = 0.1
    verbose: bool = False
    early_stopping_enabled: bool = False
    temperature: float = 0.2  # temperature for OMD softmax
    init_policy_temp: float = 0.2  # temperature for initial policy
    initialization_type: str = "uniform"  # "uniform" or "logits"


@dataclass
class DampedFPConfig:
    """Fictitious Play algorithm configuration."""

    use_python: bool = False
    num_iterations: int = 100
    convergence_tolerance: float = 1e-6
    verbose: bool = False
    initialization_type: str = (
        "PSO_uniform"  # "PSO_uniform" or "one_uniform" or "dirichlet"
    )
    early_stopping_enabled: bool = False
    lambda_schedule: str = "fictitious_play"  # "pure" | "damped" | "fictitious_play"
    damped_constant: float | None = None  # for damped_fixed_point variant
    num_transition_steps: int = 20
    init_policy_temp: float = 0.1  # temperature for initial policy
    best_response: str = "exact"  # "exact" | "ppo" | "dqn"
    ppo_total_timesteps: int = 10000
    ppo_n_steps: int = 128
    ppo_batch_size: int = 64
    ppo_learning_rate: float = 3e-4
    ppo_n_epochs: int = 10
    ppo_seed: int = 0
    ppo_reset_uniform_eps: float = 0.1
    ppo_episode_length: int | None = None
    ppo_normalize_reward: bool = True
    ppo_device: str = "cpu"
    ppo_n_envs: int = 1
    ppo_warm_start: bool = False
    dqn_total_timesteps: int = 10000
    dqn_learning_rate: float = 1e-3
    dqn_buffer_size: int = 10000
    dqn_learning_starts: int = 100
    dqn_batch_size: int = 32
    dqn_train_freq: int = 1
    dqn_gradient_steps: int = 1
    dqn_exploration_fraction: float = 0.1
    dqn_exploration_final_eps: float = 0.05
    dqn_seed: int = 0
    dqn_reset_uniform_eps: float = 0.1
    dqn_episode_length: int | None = None
    dqn_normalize_reward: bool = False
    dqn_device: str = "cpu"
    dqn_n_envs: int = 1
    dqn_warm_start: bool = True


@dataclass
class PIConfig:
    """Policy Iteration algorithm configuration."""

    use_python: bool = False
    num_iterations: int = 100
    verbose: bool = False
    early_stopping_enabled: bool = False
    initialization_type: str = "uniform"
    variant: str = "policy_iteration"  # "policy_iteration" | "smooth_policy_iteration" | "boltzmann_policy_iteration"
    init_policy_temp: float = 0.5  # temperature for initial policy creation
    temperature: float = 0.5  # temperature for boltzmann_policy_iteration (used during algorithm iterations)
    damped_constant: float | None = (
        None  # for smooth_policy_iteration: None = 1/(k+1), else constant
    )


ConfigStore.instance().store(name="pso", node=PSOConfig)
ConfigStore.instance().store(name="dampedfp", node=DampedFPConfig)
ConfigStore.instance().store(name="omd", node=OMDConfig)
ConfigStore.instance().store(name="pi", node=PIConfig)
