from typing import Optional

import jax.numpy as jnp

from envs.mfg_model_class import MFGStationary
from envs.no_interaction.no_interaction import NoInteractionChain


def transition_no_interaction_chain(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Transition function for NoInteractionChain environment."""
    assert environment is not None, "Environment must be provided"
    assert isinstance(
        environment, NoInteractionChain
    ), "Environment must be a NoInteractionChain"
    direction = action - 1  # Convert action {0,1,2} to {-1,0,1}
    noise_direction = jnp.where(
        environment.is_noisy, noise - 1, 0
    )  # Convert noise {0,1,2} to {-1,0,1}
    next_state = state + direction + noise_direction
    return jnp.clip(next_state, 0, environment.N_states - 1).astype(jnp.int32)


def reward_no_interaction_chain(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Reward function for NoInteractionChain environment."""
    assert environment is not None, "Environment must be provided"
    assert isinstance(
        environment, NoInteractionChain
    ), "Environment must be a NoInteractionChain"
    direction = action - 1  # Convert action {0,1,2} to {-1,0,1}
    action_penalty = -environment.movement_penalty * jnp.abs(direction)
    return action_penalty + state
