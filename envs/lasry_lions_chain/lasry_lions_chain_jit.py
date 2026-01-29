from typing import Optional

import jax.numpy as jnp

from envs.lasry_lions_chain.lasry_lions_chain import LasryLionsChain
from envs.mfg_model_class import MFGStationary


def transition_lasry_lions_chain(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    assert environment is not None, "Environment must be provided"
    assert isinstance(
        environment, LasryLionsChain
    ), "Environment must be a LasryLionsChain"
    direction = action - 1
    noise_dir = jnp.where(environment.is_noisy, noise - 1, 0)
    next_state = state + direction + noise_dir
    return jnp.clip(next_state, 0, environment.N_states - 1).astype(jnp.int32)


def reward_lasry_lions_chain(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(
        environment, LasryLionsChain
    ), "Environment must be a LasryLionsChain"
    direction = action - 1
    action_pen = -environment.movement_penalty * jnp.abs(direction)
    dist = jnp.abs(state - environment.center_state)
    center_pen = -environment.center_attraction * (dist**2)
    congestion = environment.crowd_penalty_coefficient * mean_field[state]
    return action_pen + center_pen - congestion
