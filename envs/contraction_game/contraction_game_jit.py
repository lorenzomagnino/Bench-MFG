from typing import Optional

import jax.numpy as jnp

from envs.contraction_game.contraction_game import ContractionGame
from envs.mfg_model_class import MFGStationary


def transition_contraction_game(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Transition function for ContractionGame environment.

    Deterministic: x_{n+1} = x_n if Stay (action=0), x_{n+1} = 1-x_n if Switch (action=1).
    """
    assert environment is not None, "Environment must be provided"
    assert isinstance(
        environment, ContractionGame
    ), "Environment must be a ContractionGame"

    # Action 0 = Stay: x_{n+1} = x_n
    # Action 1 = Switch: x_{n+1} = 1 - x_n
    next_state = jnp.where(action == 0, state, 1 - state)
    return next_state.astype(jnp.int32)


def reward_contraction_game(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Reward function for ContractionGame environment.

    Implements: r(x, a, μ) = -C · I(a=Switch) - α μ(x)
    where:
        - C = switching_cost (cost for switching states)
        - α = congestion_coefficient (congestion penalty)
    """
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(
        environment, ContractionGame
    ), "Environment must be a ContractionGame"

    # Reward: r(x, a, μ) = -C · I(a=Switch) - α μ(x)
    # Use jnp.where to create indicator function: I(a=Switch) = 1 if action==1, else 0
    switching_indicator = jnp.where(action == 1, 1.0, 0.0)
    switching_penalty = -environment.switching_cost * switching_indicator
    congestion_penalty = -environment.congestion_coefficient * mean_field[state]

    r = switching_penalty + congestion_penalty
    return r
