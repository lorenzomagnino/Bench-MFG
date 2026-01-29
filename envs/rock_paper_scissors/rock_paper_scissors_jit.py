from typing import Optional

import jax.numpy as jnp

from envs.mfg_model_class import MFGStationary
from envs.rock_paper_scissors.rock_paper_scissors import RockPaperScissors


def transition_rock_paper_scissors(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Transition function for RockPaperScissors environment."""
    assert environment is not None, "Environment must be provided"
    assert isinstance(
        environment, RockPaperScissors
    ), "Environment must be a RockPaperScissors"

    # Action directly determines the next state (deterministic)
    # Ensure action is valid by taking modulo
    next_state = action % environment.N_states
    return next_state


def reward_rock_paper_scissors(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """Reward function for RockPaperScissors environment."""
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(
        environment, RockPaperScissors
    ), "Environment must be a RockPaperScissors"

    # Compute g(x, μ) = [Aμ]x = Σ_{y∈𝒳} Axy μ(y)
    interaction_matrix = jnp.asarray(environment.interaction_matrix)
    reward = jnp.dot(interaction_matrix[state], mean_field)
    return reward
