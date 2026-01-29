from typing import Optional

import jax.numpy as jnp

from envs.four_rooms_obstacles.four_rooms_obstacles import FourRoomsAversion2D
from envs.mfg_model_class import MFGStationary


def transition_four_rooms_obstacles(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Transition function for FourRoomsAversion2D environment."""
    assert environment is not None, "Environment must be provided"
    assert isinstance(environment, FourRoomsAversion2D), (
        "Environment must be a FourRoomsAversion2D"
    )

    moves = jnp.array(
        [
            [1, 0],  # up (row 0 at bottom, row increases upward)
            [0, 1],  # right
            [-1, 0],  # down
            [0, -1],  # left
            [0, 0],  # stay
        ]
    )
    cur_row, cur_col = state // environment.cols, state % environment.cols
    act_move = moves[action]
    nz_move = moves[noise]
    prop_row = cur_row + act_move[0] + nz_move[0]
    prop_col = cur_col + act_move[1] + nz_move[1]

    # Check bounds
    valid_bounds = (
        (prop_row >= 0)
        & (prop_row < environment.rows)
        & (prop_col >= 0)
        & (prop_col < environment.cols)
    )
    obstacle_mask = jnp.asarray(environment.obstacle_mask)
    safe_row = jnp.clip(prop_row, 0, environment.rows - 1)
    safe_col = jnp.clip(prop_col, 0, environment.cols - 1)
    is_obstacle = obstacle_mask[safe_row, safe_col]
    is_obstacle = jnp.where(valid_bounds, is_obstacle, True)
    valid_obstacle = ~is_obstacle

    valid = valid_bounds & valid_obstacle

    new_row = jnp.where(valid, prop_row, cur_row)
    new_col = jnp.where(valid, prop_col, cur_col)
    return (new_row * environment.cols + new_col).astype(jnp.int32)


def reward_four_rooms_obstacles(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Reward function for FourRoomsAversion2D environment."""
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(environment, FourRoomsAversion2D), (
        "Environment must be a FourRoomsAversion2D"
    )

    dens = jnp.maximum(mean_field[state], environment.epsilon)
    return -environment.alpha * jnp.log(dens)
