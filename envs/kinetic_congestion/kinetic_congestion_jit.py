from typing import Optional

import jax.numpy as jnp

from envs.kinetic_congestion.kinetic_congestion import KineticCongestion
from envs.mfg_model_class import MFGStationary


def transition_kinetic_congestion(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Transition function for KineticCongestion environment."""
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(
        environment, KineticCongestion
    ), "Environment must be a KineticCongestion"

    random_roll = (noise + 0.5) / environment.precision_N

    # Get current position
    row = state // environment.width
    col = state % environment.width

    # Compute target coordinates based on action
    # 0: Up, 1: Right, 2: Down, 3: Left, 4: Stay
    # Row 0 is at bottom, row increases upward
    target_row = jnp.where(
        action == 0,  # Up (increase row)
        jnp.minimum(environment.height - 1, row + 1),
        jnp.where(
            action == 2,  # Down (decrease row)
            jnp.maximum(0, row - 1),
            row,  # Right, Left, or Stay
        ),
    )

    target_col = jnp.where(
        action == 1,  # Right
        jnp.minimum(environment.width - 1, col + 1),
        jnp.where(
            action == 3,  # Left
            jnp.maximum(0, col - 1),
            col,  # Up, Down, or Stay
        ),
    )

    target_state = target_row * environment.width + target_col

    # If target_state == state (happens when action == 4 or at boundaries), return state
    is_same_state = target_state == state

    # Compute congestion rejection probability
    target_density = mean_field[target_state]
    prob_rejection = jnp.where(
        target_density >= environment.capacity_threshold,
        1.0,
        jnp.minimum(1.0, target_density / environment.capacity_threshold),
    )
    prob_success = 1.0 - prob_rejection

    # If movement succeeds, go to target; otherwise stay
    movement_succeeds = random_roll < prob_success
    next_state = jnp.where(
        is_same_state,
        state,
        jnp.where(movement_succeeds, target_state, state),
    )

    return next_state.astype(jnp.int32)


def reward_kinetic_congestion(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Reward function for KineticCongestion environment."""
    assert environment is not None, "Environment must be provided"
    assert isinstance(
        environment, KineticCongestion
    ), "Environment must be a KineticCongestion"

    is_at_target = jnp.where(state == environment.target_state, 1.0, 0.0)
    move_cost = jnp.where(action == 4, 0.0, environment.movement_cost)
    dist_penalty = -1.0 * (1.0 - is_at_target)

    return dist_penalty - move_cost
