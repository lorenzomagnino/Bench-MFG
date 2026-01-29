"""JAX JIT-compatible functions for MFGarnet environment."""

from typing import Optional

import jax.numpy as jnp

from envs.mf_garnet.mf_garnet import MFGarnet
from envs.mfg_model_class import MFGStationary


def _compute_transition_distribution(
    mean_field: jnp.ndarray,
    state: int,
    action: int,
    environment: MFGarnet,
) -> jnp.ndarray:
    """Compute p(.|s,a,mu) over all next-states as a dense vector."""
    mu = mean_field / jnp.maximum(mean_field.sum(), environment.cfg.eps)

    P0_dense = jnp.array(environment.P0_dense)
    C = jnp.array(environment.C)

    base = P0_dense[state, action]

    g = C[state, action] @ mu

    is_additive = environment.cfg.dynamics_structure == "additive"

    intensity_additive = environment.cfg.cp * base + environment.cfg.rho_p * g
    intensity_additive = jnp.maximum(0.0, intensity_additive)

    gate = environment.cfg.cp + environment.cfg.rho_p * g
    gate = jnp.maximum(0.0, gate)
    intensity_multiplicative = base * gate

    intensity = jnp.where(is_additive, intensity_additive, intensity_multiplicative)

    intensity = intensity + environment.cfg.relu_eps

    denom = intensity.sum() + environment.cfg.eps
    p = intensity / denom
    p = p / jnp.maximum(p.sum(), environment.cfg.eps)

    return p


def transition_mf_garnet(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    noise: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """
    Transition function for MFGarnet environment.

    Uses inverse CDF sampling: the noise index determines where in [0,1]
    we sample from the transition distribution.

    Args:
        mean_field: Current mean field distribution
        state: Current state index
        action: Action index
        noise: Noise index (used for inverse CDF sampling)
        environment: MFGarnet environment instance

    Returns:
        Next state index as JAX array
    """
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(environment, MFGarnet), "Environment must be a MFGarnet"

    p = _compute_transition_distribution(mean_field, state, action, environment)

    u = (noise + 0.5) / environment.N_noises

    cdf = jnp.cumsum(p)

    s_next = jnp.argmax(cdf > u)

    s_next = jnp.where(u >= cdf[-1], environment.N_states - 1, s_next)

    return s_next.astype(jnp.int32)


def reward_mf_garnet(
    mean_field: Optional[jnp.ndarray] = None,
    state: int = 0,
    action: int = 0,
    environment: Optional[MFGStationary] = None,
) -> jnp.ndarray:
    """
    Reward function for MFGarnet environment.

    R(s,a,mu) depends on:
    - Base reward R0(s,a)
    - Mean-field interaction via matrix M
    - Additive or multiplicative reward structure

    Args:
        mean_field: Current mean field distribution
        state: Current state index
        action: Action index
        environment: MFGarnet environment instance

    Returns:
        Reward value as JAX array
    """
    assert environment is not None, "Environment must be provided"
    assert mean_field is not None, "Mean field must be provided"
    assert isinstance(environment, MFGarnet), "Environment must be a MFGarnet"

    mu = mean_field / jnp.maximum(mean_field.sum(), environment.cfg.eps)

    M = jnp.array(environment.M)
    R0 = jnp.array(environment.R0)
    interaction = M[state] @ mu

    r0 = R0[state, action]

    is_additive = environment.cfg.reward_structure == "additive"

    reward_additive = environment.cfg.cr * r0 + environment.cfg.rho_r * interaction

    reward_multiplicative = r0 * (
        environment.cfg.cr + environment.cfg.rho_r * interaction
    )

    reward = jnp.where(is_additive, reward_additive, reward_multiplicative)

    return reward
