"""
Configuration utilities for creating MFG environments from Hydra configs.

This module provides helper functions to instantiate environments from
configuration objects, handling parameter mapping and validation.
"""

import logging

import numpy as np

from conf.config_schema import EnvironmentConfig, InitialDistributionConfig
from envs.contraction_game.contraction_game import ContractionGame
from envs.four_rooms_obstacles.four_rooms_obstacles import FourRoomsAversion2D
from envs.kinetic_congestion.kinetic_congestion import KineticCongestion
from envs.lasry_lions_chain.lasry_lions_chain import LasryLionsChain
from envs.mf_garnet.mf_garnet import MFGarnet, MFGarnetSampling
from envs.multiple_equilibria.multiple_equilibria import MultipleEquilibria1DGame
from envs.no_interaction.no_interaction import NoInteractionChain
from envs.rock_paper_scissors.rock_paper_scissors import RockPaperScissors
from envs.sis_epidemic.sis_epidemic import SISEpidemic
from learner.jax.pso_jax import boltzmann_policy


def create_initial_distribution(
    config: InitialDistributionConfig, num_states: int
) -> np.ndarray:
    """
    Create initial population distribution from configuration.

    Parameters:
        config (InitialDistributionConfig): Initial distribution configuration
        num_states (int): Number of states in the environment

    Returns:
        np.ndarray: Initial population distribution
    """
    if config.type == "uniform":
        return np.ones(num_states) / num_states

    elif config.type == "concentrated":
        mu = np.zeros(num_states)
        concentration_state = min(config.concentration_state, num_states - 1)
        mu[concentration_state] = config.concentration_ratio

        remaining_prob = 1.0 - config.concentration_ratio
        remaining_states = num_states - 1
        if remaining_states > 0:
            for i in range(num_states):
                if i != concentration_state:
                    mu[i] = remaining_prob / remaining_states

        return mu

    elif config.type == "custom":
        custom_values = np.array(config.custom_values)
        if len(custom_values) != num_states:
            raise ValueError(
                f"Custom distribution length ({len(custom_values)}) "
                f"doesn't match number of states ({num_states})"
            )

        custom_values = custom_values / np.sum(custom_values)
        return custom_values

    else:
        raise ValueError(f"Unknown initial distribution type: {config.type}")


def initialize_policy_from_logits(
    environment, initialization_type: str, temperature: float
) -> np.ndarray:
    """
    Initialize a policy using logits similar to PSO initialization approach.

    Args:
        environment: The MFG environment
        initialization_type: Type of initialization ("dirichlet", "PSO_uniform", "one_uniform")
        temperature: Temperature for Boltzmann conversion

    Returns:
        initial_policy: Policy as probability distribution
    """
    N_states = environment.N_states
    N_actions = environment.N_actions

    if initialization_type == "dirichlet":
        logging.info("Using dirichlet initialization for policy")
        policies = np.random.dirichlet(np.ones(N_actions), size=(N_states,))
        logits = np.log(policies + 1e-12)
        initial_policy = boltzmann_policy(logits, temperature)

    elif initialization_type == "PSO_uniform":
        logging.info("Using PSO uniform initialization for policy")
        logits = np.random.uniform(-1.0, 1.0, (N_states, N_actions))
        initial_policy = boltzmann_policy(logits, temperature)

    elif initialization_type == "one_uniform":
        logging.info("Using one uniform initialization for policy")
        logits = np.zeros((N_states, N_actions))
        initial_policy = boltzmann_policy(logits, temperature)

    else:
        raise ValueError(f"Unknown initialization_type: {initialization_type}")

    return initial_policy


def validate_environment_config(config: EnvironmentConfig) -> None:
    """
    Validate environment configuration parameters.

    Parameters:
        config (EnvironmentConfig): Environment configuration to validate

    Raises:
        ValueError: If configuration is invalid
    """
    if config.num_states <= 0:
        raise ValueError("Number of states must be positive")

    if config.horizon <= 0:
        raise ValueError("Horizon must be positive")

    if hasattr(config, "gamma") and not (0 <= config.gamma <= 1):
        raise ValueError("Gamma must be between 0 and 1")

    if config.num_noises != len(config.dynamics.noise_probabilities):
        raise ValueError("Number of noise values must match number of probabilities")

    if config.reward.lasry_lions.crowd_penalty_coefficient < 0:
        raise ValueError("Alpha must be positive")

    if config.reward.lasry_lions.movement_penalty < 0:
        raise ValueError("Movement penalty must be non-negative")

    if (
        hasattr(config.reward, "center_attraction")
        and config.reward.lasry_lions.center_attraction < 0
    ):
        raise ValueError("Center attraction must be non-negative")

    if config.initial_distribution.type == "concentrated":
        if not (
            0 <= config.initial_distribution.concentration_state < config.num_states
        ):
            raise ValueError("Concentration state must be within valid range")
        if not (0 <= config.initial_distribution.concentration_ratio <= 1):
            raise ValueError("Concentration ratio must be between 0 and 1")

    elif config.initial_distribution.type == "custom":
        if config.initial_distribution.custom_values != config.num_states:
            raise ValueError("Custom distribution length must match number of states")


def get_environment_info(cfg: EnvironmentConfig) -> dict:
    """
    Get summary information about the environment.

    Parameters:
        env (LasryLionsChain): Environment instance

    Returns:
        dict: Environment information
    """
    return {
        "environment_type": cfg.name,
        "num_states": cfg.num_states,
        "num_actions": cfg.num_actions,
        "horizon": cfg.horizon,
        "reward": cfg.reward,
        "initial_distribution": cfg.initial_distribution.type,
    }


def create_lasry_lions_chain_from_config(config: EnvironmentConfig) -> LasryLionsChain:
    """
    Create a LasryLionsChain environment from Hydra configuration.

    Parameters:
        config (EnvironmentConfig): Environment configuration

    Returns:
        LasryLionsChain: Configured environment instance
    """
    num_states = config.num_states
    horizon = config.horizon
    gamma = config.gamma

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)

    # Create noise probability matrix
    noise_prob = np.array(config.dynamics.noise_probabilities)
    if config.reward.lasry_lions.crowd_penalty_coefficient < 0:
        raise ValueError("Crowd penalty coefficient must be non-negative")

    center_attraction = config.reward.lasry_lions.center_attraction

    env = LasryLionsChain(
        N_states=num_states,
        N_actions=config.num_actions,
        N_noises=config.num_noises,
        horizon=horizon,
        mean_field=mu_0,
        noise_prob=noise_prob,
        crowd_penalty_coefficient=config.reward.lasry_lions.crowd_penalty_coefficient,
        movement_penalty=config.reward.lasry_lions.movement_penalty,
        center_attraction=center_attraction,
        gamma=gamma,
        is_noisy=config.dynamics.is_noisy,
    )

    return env


def create_no_interaction_chain_from_config(
    config: EnvironmentConfig,
) -> NoInteractionChain:
    """
    Create a LasryLionsChain environment from Hydra configuration.

    Parameters:
        config (EnvironmentConfig): Environment configuration

    Returns:
        LasryLionsChain: Configured environment instance
    """
    num_states = config.num_states
    horizon = config.horizon
    gamma = config.gamma

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)

    noise_prob = np.array(config.dynamics.noise_probabilities)
    if config.reward.no_interaction.movement_penalty < 0:
        raise ValueError("Movement penalty must be non-negative")

    env = NoInteractionChain(
        N_states=num_states,
        N_actions=config.num_actions,
        N_noises=config.num_noises,
        horizon=horizon,
        mean_field=mu_0,
        noise_prob=noise_prob,
        movement_penalty=config.reward.no_interaction.movement_penalty,
        gamma=gamma,
        is_noisy=config.dynamics.is_noisy,
    )
    return env


def create_four_rooms_aversion2d_from_config(
    config: EnvironmentConfig,
) -> FourRoomsAversion2D:
    """
    Create a FourRoomsAversion2D environment from Hydra configuration.
    """
    num_states = config.num_states
    horizon = config.horizon
    gamma = config.gamma

    if not config.grid.is_grid:
        raise ValueError("FourRoomsAversion2D requires grid configuration enabled")
    grid_dim_list = config.grid.dimension

    grid_dim = np.array(grid_dim_list, dtype=int)
    if grid_dim.shape != (2,):
        raise ValueError("grid.dimension must be [rows, cols]")
    if int(grid_dim[0]) != 11 or int(grid_dim[1]) != 11:
        raise ValueError("FourRoomsAversion2D expects grid.dimension == [11, 11]")

    if config.num_actions != 5:
        raise ValueError("FourRoomsAversion2D requires num_actions == 5")
    if config.num_noises != 5:
        raise ValueError("FourRoomsAversion2D requires num_noises == 5")

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)
    noise_prob = np.array(config.dynamics.noise_probabilities, dtype=float)

    alpha = getattr(getattr(config.reward, "aversion", None), "alpha", 1.0)
    eps = getattr(getattr(config.reward, "aversion", None), "epsilon", 1e-12)

    doors_cfg = getattr(getattr(config, "obstacles", None), "doors", None)
    doors = None
    if doors_cfg is not None:
        doors = tuple((int(d[0]), int(d[1])) for d in doors_cfg)

    return FourRoomsAversion2D(
        horizon=horizon,
        mean_field=mu_0,
        noise_prob=noise_prob,
        gamma=gamma,
        alpha=float(alpha),
        epsilon=float(eps),
        grid_dim=grid_dim,
        doors=doors,
    )


def create_strict_contraction_game_from_config(
    config: EnvironmentConfig,
) -> MultipleEquilibria1DGame:
    num_states = config.num_states
    horizon = config.horizon
    gamma = config.gamma

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)
    noise_prob = np.array(config.dynamics.noise_probabilities, dtype=float)

    r_cfg = config.reward.strict_contraction

    env = MultipleEquilibria1DGame(
        N_states=num_states,
        N_actions=config.num_actions,
        N_noises=config.num_noises,
        horizon=horizon,
        mean_field=mu_0,
        noise_prob=noise_prob,
        alpha=r_cfg.alpha,
        beta=r_cfg.beta,
        targets=r_cfg.targets,
        movement_penalty=r_cfg.movement_penalty,
        gamma=gamma,
        is_noisy=config.dynamics.is_noisy,
    )
    return env


def create_multiple_equilibria_from_config(
    config: EnvironmentConfig,
) -> MultipleEquilibria1DGame:
    """Create MultipleEquilibria1DGame environment from configuration.

    This is an alias for create_strict_contraction_game_from_config to support
    the "MultipleEquilibriaGame" environment name.
    """
    return create_strict_contraction_game_from_config(config)


def create_mf_garnet_from_config(config: EnvironmentConfig) -> MFGarnet:
    num_states = config.num_states
    horizon = config.horizon
    gamma = config.gamma

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)

    gcfg = config.reward.mfgarnet
    sampling = MFGarnetSampling(
        seed=gcfg.seed,
        branching_factor=gcfg.branching_factor,
        dynamics_structure=gcfg.dynamics_structure,  # type: ignore[arg-type]
        cp=gcfg.cp,
        rho_p=gcfg.rho_p,
        reward_structure=gcfg.reward_structure,  # type: ignore[arg-type]
        cr=gcfg.cr,
        rho_r=gcfg.rho_r,
        game_type=gcfg.game_type,  # type: ignore[arg-type]
        reward_scale=gcfg.reward_scale,
        eps=gcfg.eps,
        relu_eps=gcfg.relu_eps,
    )

    return MFGarnet(
        N_states=num_states,
        N_actions=config.num_actions,
        N_noises=config.num_noises,
        horizon=horizon,
        mean_field=mu_0,
        gamma=gamma,
        sampling=sampling,
    )


def create_rock_paper_scissors_from_config(
    config: EnvironmentConfig,
) -> RockPaperScissors:
    """
    Create a RockPaperScissors environment from Hydra configuration.

    Parameters:
        config (EnvironmentConfig): Environment configuration

    Returns:
        RockPaperScissors: Configured environment instance
    """
    num_states = 3
    num_actions = 3
    horizon = config.horizon
    gamma = config.gamma

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)

    if (
        hasattr(config.dynamics, "noise_probabilities")
        and config.dynamics.noise_probabilities
    ):
        noise_prob = np.array(config.dynamics.noise_probabilities)
    else:
        noise_prob = np.array([1.0])

    env = RockPaperScissors(
        N_states=num_states,
        N_actions=num_actions,
        N_noises=len(noise_prob),
        horizon=horizon,
        mean_field=mu_0,
        noise_prob=noise_prob,
        gamma=gamma,
    )

    return env


def create_sis_epidemic_from_config(
    config: EnvironmentConfig,
) -> SISEpidemic:
    """
    Create a SISEpidemic environment from Hydra configuration.

    Parameters:
        config (EnvironmentConfig): Environment configuration

    Returns:
        SISEpidemic: Configured environment instance
    """
    num_actions = config.num_actions
    horizon = config.horizon
    gamma = config.gamma

    mu_0 = create_initial_distribution(config.initial_distribution, num_states=2)

    beta = 0.5
    nu = 0.1
    cost_infection = 1.0

    if hasattr(config.reward, "sis_epidemic"):
        sis_cfg = config.reward.sis_epidemic
        if hasattr(sis_cfg, "beta"):
            beta = float(sis_cfg.beta)
        if hasattr(sis_cfg, "nu"):
            nu = float(sis_cfg.nu)
        if hasattr(sis_cfg, "cost_infection"):
            cost_infection = float(sis_cfg.cost_infection)

    env = SISEpidemic(
        N_actions=num_actions,
        horizon=horizon,
        mean_field=mu_0,
        beta=beta,
        nu=nu,
        cost_infection=cost_infection,
        gamma=gamma,
    )

    return env


def create_kinetic_congestion_from_config(
    config: EnvironmentConfig,
) -> KineticCongestion:
    """
    Create a KineticCongestion environment from Hydra configuration.

    Parameters:
        config (EnvironmentConfig): Environment configuration

    Returns:
        KineticCongestion: Configured environment instance
    """
    horizon = config.horizon
    gamma = config.gamma

    grid_height = config.grid.dimension[0] if len(config.grid.dimension) > 0 else 5
    grid_width = config.grid.dimension[1] if len(config.grid.dimension) > 1 else 5

    num_states = grid_height * grid_width

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)

    target_state = 0
    movement_cost = 0.1
    capacity_threshold = 0.4

    if hasattr(config.reward, "kinetic_congestion"):
        kc_cfg = config.reward.kinetic_congestion
        if hasattr(kc_cfg, "target_state"):
            target_state = int(kc_cfg.target_state)
        if hasattr(kc_cfg, "movement_cost"):
            movement_cost = float(kc_cfg.movement_cost)
        if hasattr(kc_cfg, "capacity_threshold"):
            capacity_threshold = float(kc_cfg.capacity_threshold)

    if target_state < 0 or target_state >= num_states:
        raise ValueError(
            f"target_state ({target_state}) must be in [0, {num_states - 1}]"
        )

    env = KineticCongestion(
        grid_height=grid_height,
        grid_width=grid_width,
        horizon=horizon,
        mean_field=mu_0,
        target_state=target_state,
        movement_cost=movement_cost,
        capacity_threshold=capacity_threshold,
        gamma=gamma,
    )

    return env


def create_contraction_game_from_config(
    config: EnvironmentConfig,
) -> ContractionGame:
    """
    Create a ContractionGame environment from Hydra configuration.

    Parameters:
        config (EnvironmentConfig): Environment configuration

    Returns:
        ContractionGame: Configured environment instance
    """
    num_states = config.num_states
    horizon = config.horizon
    gamma = config.gamma

    mu_0 = create_initial_distribution(config.initial_distribution, num_states)
    noise_prob = np.array(config.dynamics.noise_probabilities, dtype=float)

    r_cfg = config.reward.contraction_game

    env = ContractionGame(
        N_states=num_states,
        N_actions=config.num_actions,
        N_noises=config.num_noises,
        horizon=horizon,
        mean_field=mu_0,
        noise_prob=noise_prob,
        switching_cost=r_cfg.switching_cost,
        congestion_coefficient=r_cfg.congestion_coefficient,
        gamma=gamma,
        is_noisy=config.dynamics.is_noisy,
    )
    return env
