"""Create the optimization solver based on configuration."""

from typing import cast

import numpy as np

from conf.algorithm.algorithm_schema import LambdaSchedule, PIVariant
from conf.config_schema import AlgorithmConfig, MFGConfig
from envs.contraction_game.contraction_game_jit import (
    reward_contraction_game,
    transition_contraction_game,
)
from envs.four_rooms_obstacles.four_rooms_obstacles_jit import (
    reward_four_rooms_obstacles,
    transition_four_rooms_obstacles,
)
from envs.kinetic_congestion.kinetic_congestion_jit import (
    reward_kinetic_congestion,
    transition_kinetic_congestion,
)
from envs.lasry_lions_chain.lasry_lions_chain_jit import (
    reward_lasry_lions_chain,
    transition_lasry_lions_chain,
)
from envs.mf_garnet.mf_garnet_jit import (
    reward_mf_garnet,
    transition_mf_garnet,
)
from envs.mfg_model_class import MFGStationary
from envs.mfg_model_class_jit import EnvSpec
from envs.multiple_equilibria.multiple_equilibria_jit import (
    reward_multiple_equilibria,
    transition_multiple_equilibria,
)
from envs.no_interaction.no_interaction_jit import (
    reward_no_interaction_chain,
    transition_no_interaction_chain,
)
from envs.rock_paper_scissors.rock_paper_scissors_jit import (
    reward_rock_paper_scissors,
    transition_rock_paper_scissors,
)
from envs.sis_epidemic.sis_epidemic_jit import (
    reward_sis_epidemic,
    transition_sis_epidemic,
)
from learner.jax.fp_jax import DampedFP_jax
from learner.jax.omd_jax import OMD_jax
from learner.jax.pi_jax import PI_jax
from learner.jax.pso_jax import PSO_jax
from learner.python.fp_py import DampedFP_python
from learner.python.omd_py import OMD_python
from learner.python.pi_py import PI_python

# =============================================================================
# Environment to JIT functions mapping
# =============================================================================

ENV_JIT_FUNCTIONS = {
    "LasryLionsChain": (transition_lasry_lions_chain, reward_lasry_lions_chain),
    "NoInteractionChain": (
        transition_no_interaction_chain,
        reward_no_interaction_chain,
    ),
    "FourRoomsAversion2D": (
        transition_four_rooms_obstacles,
        reward_four_rooms_obstacles,
    ),
    "RockPaperScissors": (transition_rock_paper_scissors, reward_rock_paper_scissors),
    "KineticCongestion": (transition_kinetic_congestion, reward_kinetic_congestion),
    "SISEpidemic": (transition_sis_epidemic, reward_sis_epidemic),
    "StrictContractionGame": (
        transition_multiple_equilibria,
        reward_multiple_equilibria,
    ),
    "MultipleEquilibriaGame": (
        transition_multiple_equilibria,
        reward_multiple_equilibria,
    ),
    "ContractionGame": (transition_contraction_game, reward_contraction_game),
    "MFGarnet": (transition_mf_garnet, reward_mf_garnet),
}


def _get_env_spec(environment: MFGStationary, env_name: str) -> EnvSpec:
    """Create EnvSpec for the given environment."""
    if env_name not in ENV_JIT_FUNCTIONS:
        raise ValueError(f"Unknown environment: {env_name}")

    transition_fn, reward_fn = ENV_JIT_FUNCTIONS[env_name]
    return EnvSpec(
        environment=environment,
        transition=transition_fn,
        reward=reward_fn,
    )


# =============================================================================
# Generic Python solver creators (work with any MFGStationary environment)
# =============================================================================


def _create_omd_solver_python(
    environment: MFGStationary,
    initial_policy: np.ndarray,
    algo_cfg: AlgorithmConfig,
) -> OMD_python:
    """Create a Python OMD solver for any environment."""
    return OMD_python(
        model=environment,
        initial_policy=initial_policy,
        learning_rate=algo_cfg.omd.learning_rate,
        num_iterations=algo_cfg.omd.num_iterations,
        early_stopping_enabled=algo_cfg.omd.early_stopping_enabled,
        temperature=algo_cfg.omd.temperature,
    )


def _create_fp_solver_python(
    environment: MFGStationary,
    initial_policy: np.ndarray,
    algo_cfg: AlgorithmConfig,
) -> DampedFP_python:
    """Create a Python DampedFP solver for any environment."""
    damped_constant = (
        algo_cfg.dampedfp.damped_constant
        if algo_cfg.dampedfp.damped_constant is not None
        else 0.2
    )
    return DampedFP_python(
        model=environment,
        initial_policy=initial_policy,
        num_iterations=algo_cfg.dampedfp.num_iterations,
        early_stopping_enabled=algo_cfg.dampedfp.early_stopping_enabled,
        lambda_schedule=cast(LambdaSchedule, algo_cfg.dampedfp.lambda_schedule),
        damped_constant=damped_constant,
        num_transition_steps=algo_cfg.dampedfp.num_transition_steps,
    )


def _create_pi_solver_python(
    environment: MFGStationary,
    initial_policy: np.ndarray,
    algo_cfg: AlgorithmConfig,
) -> PI_python:
    """Create a Python PI solver for any environment."""
    return PI_python(
        model=environment,
        initial_policy=initial_policy,
        num_iterations=algo_cfg.pi.num_iterations,
        early_stopping_enabled=algo_cfg.pi.early_stopping_enabled,
        variant=cast(PIVariant, algo_cfg.pi.variant),
        temperature=algo_cfg.pi.temperature,
        damped_constant=algo_cfg.pi.damped_constant,
    )


# =============================================================================
# Generic JAX solver creators (work with any environment via EnvSpec)
# =============================================================================


def _create_pso_solver_jax(
    environment: MFGStationary,
    env_name: str,
    algo_cfg: AlgorithmConfig,
) -> PSO_jax:
    """Create a JAX PSO solver for any environment."""
    env_spec = _get_env_spec(environment, env_name)
    return PSO_jax(
        env_spec=env_spec,
        num_particles=algo_cfg.pso.num_particles,
        num_iterations=algo_cfg.pso.num_iterations,
        w=algo_cfg.pso.w,
        c1=algo_cfg.pso.c1,
        c2=algo_cfg.pso.c2,
        temperature=algo_cfg.pso.temperature,
        policy_type=algo_cfg.pso.policy_type,
        initialization_type=algo_cfg.pso.initialization_type,
        init_policy_temp=algo_cfg.pso.init_policy_temp,
    )


def _create_fp_solver_jax(
    environment: MFGStationary,
    env_name: str,
    initial_policy: np.ndarray,
    algo_cfg: AlgorithmConfig,
) -> DampedFP_jax:
    """Create a JAX DampedFP solver for any environment."""
    env_spec = _get_env_spec(environment, env_name)
    damped_constant = (
        algo_cfg.dampedfp.damped_constant
        if algo_cfg.dampedfp.damped_constant is not None
        else 0.2
    )
    return DampedFP_jax(
        env_spec=env_spec,
        initial_policy=initial_policy,
        num_iterations=algo_cfg.dampedfp.num_iterations,
        early_stopping_enabled=algo_cfg.dampedfp.early_stopping_enabled,
        lambda_schedule=cast(LambdaSchedule, algo_cfg.dampedfp.lambda_schedule),
        damped_constant=damped_constant,
        num_transition_steps=algo_cfg.dampedfp.num_transition_steps,
    )


def _create_omd_solver_jax(
    environment: MFGStationary,
    env_name: str,
    initial_policy: np.ndarray,
    algo_cfg: AlgorithmConfig,
) -> OMD_jax:
    """Create a JAX OMD solver for any environment."""
    env_spec = _get_env_spec(environment, env_name)
    return OMD_jax(
        env_spec=env_spec,
        initial_policy=initial_policy,
        learning_rate=algo_cfg.omd.learning_rate,
        num_iterations=algo_cfg.omd.num_iterations,
        early_stopping_enabled=algo_cfg.omd.early_stopping_enabled,
        temperature=algo_cfg.omd.temperature,
    )


def _create_pi_solver_jax(
    environment: MFGStationary,
    env_name: str,
    initial_policy: np.ndarray,
    algo_cfg: AlgorithmConfig,
) -> PI_jax:
    """Create a JAX PI solver for any environment."""
    env_spec = _get_env_spec(environment, env_name)
    return PI_jax(
        env_spec=env_spec,
        initial_policy=initial_policy,
        num_iterations=algo_cfg.pi.num_iterations,
        early_stopping_enabled=algo_cfg.pi.early_stopping_enabled,
        variant=cast(PIVariant, algo_cfg.pi.variant),
        temperature=algo_cfg.pi.temperature,
        damped_constant=algo_cfg.pi.damped_constant,
    )


def create_solver(
    environment: MFGStationary,
    initial_policy: np.ndarray,
    cfg: MFGConfig,
) -> (
    PSO_jax | DampedFP_jax | DampedFP_python | OMD_jax | OMD_python | PI_jax | PI_python
):
    """Create the optimization solver based on configuration.

    Args:
        environment: The MFG environment.
        initial_policy: The initialized policy from create_environment.
        cfg: MFGConfig containing algorithm and initialization settings.

    Returns:
        The solver instance (PSO, DampedFP, OMD, or PI).
    """
    algo_cfg: AlgorithmConfig = cfg.algorithm
    env_name = cfg.environment.name

    if algo_cfg._target_ == "PSO":
        solver = _create_pso_solver_jax(environment, env_name, algo_cfg)
    elif algo_cfg._target_ == "DampedFP":
        if algo_cfg.dampedfp.use_python:
            solver = _create_fp_solver_python(environment, initial_policy, algo_cfg)
        else:
            solver = _create_fp_solver_jax(
                environment, env_name, initial_policy, algo_cfg
            )
    elif algo_cfg._target_ == "OMD":
        if algo_cfg.omd.use_python:
            solver = _create_omd_solver_python(environment, initial_policy, algo_cfg)
        else:
            solver = _create_omd_solver_jax(
                environment, env_name, initial_policy, algo_cfg
            )
    elif algo_cfg._target_ == "PI":
        if algo_cfg.pi.use_python:
            solver = _create_pi_solver_python(environment, initial_policy, algo_cfg)
        else:
            solver = _create_pi_solver_jax(
                environment, env_name, initial_policy, algo_cfg
            )
    else:
        raise ValueError(f"Unknown algorithm: {cfg.algorithm._target_}")
    return solver
