from typing import Optional

import jax.numpy as jnp

from envs.mfg_model_class import MFGStationary
from envs.multiple_equilibria.multiple_equilibria import MultipleEquilibria1DGame


def transition_multiple_equilibria(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Transition function for MultipleEquilibria1DGame environment."""
    assert environment is not None, "Environment must be provided"
    assert isinstance(
        environment, MultipleEquilibria1DGame
    ), "Environment must be a MultipleEquilibria1DGame"

    # Convert action {0,1,2} to {-1,0,1}
    a_move = action - 1
    # Convert noise {0,1,2} to {-1,0,1}
    eps = jnp.where(environment.is_noisy, noise - 1, 0)

    next_state = state + a_move + eps
    return jnp.clip(next_state, 0, environment.N_states - 1).astype(jnp.int32)


def reward_multiple_equilibria(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Reward function for MultipleEquilibria1DGame environment.

    Implements: r(x,a,μ) = -c₁|a| - c₂ min{|x-x_L|, |x-x_R|} + α μ(x)
    where:
        - c₁ = movement_penalty (penalizes motion)
        - c₂ = beta (attraction to beach bars)
        - α = alpha (positive mean-field interaction, favors crowded states)
        - x_L, x_R = targets (symmetric target locations)
    """
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(
        environment, MultipleEquilibria1DGame
    ), "Environment must be a MultipleEquilibria1DGame"

    # Convert action {0,1,2} to {-1,0,1}
    a_move = action - 1

    # Compute minimal absolute distance to any target
    # Convert targets tuple to JAX array for JAX operations
    targets_array = jnp.array(environment.targets)
    dist_to_targets = jnp.min(jnp.abs(state - targets_array))

    # Reward: r(x,a,μ) = -c₁|a| - c₂ min{|x-x_L|, |x-x_R|} + α μ(x)
    r = (
        -environment.movement_penalty * jnp.abs(a_move)  # -c₁|a|
        - environment.beta * dist_to_targets  # -c₂ min{|x-x_L|, |x-x_R|}
        + environment.alpha
        * mean_field[state]  # +α μ(x) - positive mean-field interaction
    )
    return r
