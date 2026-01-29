from typing import Optional

import jax.numpy as jnp

from envs.mfg_model_class import MFGStationary
from envs.sis_epidemic.sis_epidemic import SISEpidemic


def transition_sis_epidemic(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Transition function for SISEpidemic environment."""
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(environment, SISEpidemic), "Environment must be a SISEpidemic"

    random_roll = (noise + 0.5) / environment.precision_N
    social_intensity = jnp.asarray(environment.action_levels)[action]

    # If susceptible (state == 0)
    prevalence = mean_field[1]
    prob_infection = environment.beta * social_intensity * prevalence
    prob_infection = jnp.clip(prob_infection, 0.0, 1.0)

    # Transition logic
    next_state = jnp.where(
        state == 0,  # Susceptible
        jnp.where(
            random_roll < prob_infection, 1, 0
        ),  # Become infected or stay susceptible
        jnp.where(
            state == 1,  # Infected
            jnp.where(random_roll < environment.nu, 0, 1),  # Recover or stay infected
            state,  # Fallback (shouldn't happen)
        ),
    )

    return next_state.astype(jnp.int32)


def reward_sis_epidemic(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Reward function for SISEpidemic environment."""
    assert environment is not None, "Environment must be provided"
    assert isinstance(environment, SISEpidemic), "Environment must be a SISEpidemic"

    social_intensity = jnp.asarray(environment.action_levels)[action]
    utility = social_intensity
    infection_cost = jnp.where(state == 1, environment.cost_infection, 0.0)

    return utility - infection_cost
