from dataclasses import dataclass, field
from functools import partial
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp

from envs.mfg_model_class import MFGStationary

jax_jit = partial(jax.jit, static_argnames="spec")


@dataclass(frozen=True)
class EnvSpec:
    environment: MFGStationary
    transition: Callable[
        [Optional[jnp.ndarray], int, int, int, Optional[MFGStationary]], jnp.ndarray
    ] = field(compare=False)
    reward: Callable[
        [Optional[jnp.ndarray], int, int, Optional[MFGStationary]], jnp.ndarray
    ] = field(compare=False)


@jax_jit
def mean_field_by_transition_kernel_one_step_jax(
    policy: jnp.ndarray,
    spec: EnvSpec,
) -> jnp.ndarray:
    """
    Computes the new state distribution using JAX's vmap.

    Parameters:
    policy: Policy of dimension N_states by N_actions.
    spec: Environment specification

    Logic:
    next_states_san represents the next states for all (s, a, n) via nested vmap → shape (S, A, N)
    probs represents the joint probabilities for each (s, a, n)
    new_state_dist is the new state distribution that assigns probability to each state.
    Finally, normalize the new state distribution to ensure it is a probability distribution.
    """
    S, A, N = (
        spec.environment.N_states,
        spec.environment.N_actions,
        spec.environment.N_noises,
    )
    stationary_mf = jnp.asarray(spec.environment.stationary_mean_field)
    noise_prob = jnp.asarray(spec.environment.noise_prob)

    s_idx = jnp.arange(S)
    a_idx = jnp.arange(A)
    n_idx = jnp.arange(N)

    def trans_for_sa(s, a):
        return jax.vmap(
            lambda n: spec.transition(stationary_mf, s, a, n, spec.environment)
        )(n_idx)

    next_states_san = jax.vmap(lambda s: jax.vmap(lambda a: trans_for_sa(s, a))(a_idx))(
        s_idx
    )

    probs = (
        policy[:, :, None] * stationary_mf[:, None, None] * noise_prob[None, None, :]
    )

    new_state_dist = jnp.zeros(S).at[next_states_san].add(probs)

    return new_state_dist / new_state_dist.sum()


@partial(jax.jit, static_argnames=("spec", "num_iterations"))
def mean_field_by_transition_kernel_multi_jax(
    policy: jnp.ndarray,
    spec: EnvSpec,
    num_iterations: int,
    initial_mean_field: jnp.ndarray,
) -> jnp.ndarray:
    """
    Iteratively applies the mean-field transition kernel for `num_iterations` steps
    entirely within JAX using jax.lax.fori_loop.

    Parameters:
    - policy: Array with shape (S, A)
    - spec: Static environment specification
    - num_iterations: Number of transition iterations (static for JIT)
    - initial_mean_field: Initial mean field distribution (required to avoid JIT caching issues)

    Logic: spec.environment is the environment instance
    """
    S, A, N = (
        spec.environment.N_states,
        spec.environment.N_actions,
        spec.environment.N_noises,
    )
    policy = policy.reshape(S, A)
    noise_prob = jnp.asarray(spec.environment.noise_prob)

    s_idx = jnp.arange(S)
    a_idx = jnp.arange(A)
    n_idx = jnp.arange(N)

    def one_step(current_mf: jnp.ndarray) -> jnp.ndarray:
        def trans_for_sa(s, a):
            return jax.vmap(
                lambda n: spec.transition(current_mf, s, a, n, spec.environment)
            )(n_idx)

        next_states_san = jax.vmap(
            lambda s: jax.vmap(lambda a: trans_for_sa(s, a))(a_idx)
        )(s_idx)

        probs = (
            policy[:, :, None] * current_mf[:, None, None] * noise_prob[None, None, :]
        )
        new_state_dist = jnp.zeros(S).at[next_states_san].add(probs)
        return new_state_dist / new_state_dist.sum()

    def body_fun(_, mf):
        return one_step(mf)

    mf_final = jax.lax.fori_loop(0, num_iterations, body_fun, initial_mean_field)
    return mf_final


@jax_jit
def Vpi_opt_jax(
    mean_field: jnp.ndarray,
    spec: EnvSpec,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the optimal value function using jax.lax.scan."""
    S, A, N = (
        spec.environment.N_states,
        spec.environment.N_actions,
        spec.environment.N_noises,
    )
    noise_prob = jnp.asarray(spec.environment.noise_prob)

    s_idx = jnp.arange(S)
    a_idx = jnp.arange(A)
    n_idx = jnp.arange(N)

    rewards_sa = jax.vmap(
        lambda s: jax.vmap(lambda a: spec.reward(mean_field, s, a, spec.environment))(
            a_idx
        )
    )(s_idx)

    def trans_for_sa(s, a):
        return jax.vmap(
            lambda n: spec.transition(mean_field, s, a, n, spec.environment)
        )(n_idx)

    next_states_san = jax.vmap(lambda s: jax.vmap(lambda a: trans_for_sa(s, a))(a_idx))(
        s_idx
    )

    def bellman_step(V_k, _):
        expected_V_sa = jnp.einsum("n,san->sa", noise_prob, V_k[next_states_san])
        Q_sa = rewards_sa + spec.environment.gamma * expected_V_sa

        V_k_plus_1 = jnp.max(Q_sa, axis=1)
        best_action = jnp.argmax(Q_sa, axis=1)

        return V_k_plus_1, best_action

    V_initial = jnp.zeros(S)
    V_final, optimal_actions_hist = jax.lax.scan(
        bellman_step, V_initial, xs=None, length=spec.environment.horizon - 1
    )

    pi_opt = jax.nn.one_hot(optimal_actions_hist[-1], num_classes=A)

    return V_final, pi_opt


@jax_jit
def V_eval_jax(
    policy: jnp.ndarray,
    mean_field: jnp.ndarray,
    spec: EnvSpec,
) -> jnp.ndarray:
    """Evaluates the value function using jax.lax.scan."""
    S, A, N = (
        spec.environment.N_states,
        spec.environment.N_actions,
        spec.environment.N_noises,
    )
    noise_prob = jnp.asarray(spec.environment.noise_prob)

    s_idx = jnp.arange(S)
    a_idx = jnp.arange(A)
    n_idx = jnp.arange(N)

    rewards_sa = jax.vmap(
        lambda s: jax.vmap(lambda a: spec.reward(mean_field, s, a, spec.environment))(
            a_idx
        )
    )(s_idx)

    def trans_for_sa(s, a):
        return jax.vmap(
            lambda n: spec.transition(mean_field, s, a, n, spec.environment)
        )(n_idx)

    next_states_san = jax.vmap(lambda s: jax.vmap(lambda a: trans_for_sa(s, a))(a_idx))(
        s_idx
    )

    def bellman_step_eval(V_k, _):
        expected_V_sa = jnp.einsum("n,san->sa", noise_prob, V_k[next_states_san])
        Q_sa = rewards_sa + spec.environment.gamma * expected_V_sa
        V_k_plus_1 = jnp.einsum("sa,sa->s", policy, Q_sa)
        return V_k_plus_1, None  # Return new V, no stacked output needed

    V_initial = jnp.zeros(S)
    V_final, _ = jax.lax.scan(
        bellman_step_eval, V_initial, xs=None, length=spec.environment.horizon - 1
    )

    return V_final


@jax_jit
def Q_eval_jax(
    policy: jnp.ndarray,
    mean_field: jnp.ndarray,
    spec: EnvSpec,
) -> jnp.ndarray:
    """Evaluates the state-action value function using jax.lax.scan.

    Returns Q-values from the second-to-last iteration (matching the original Q_eval behavior).
    """
    S, A, N = (
        spec.environment.N_states,
        spec.environment.N_actions,
        spec.environment.N_noises,
    )
    noise_prob = jnp.asarray(spec.environment.noise_prob)
    policy = policy.reshape(S, A)

    s_idx = jnp.arange(S)
    a_idx = jnp.arange(A)
    n_idx = jnp.arange(N)

    rewards_sa = jax.vmap(
        lambda s: jax.vmap(lambda a: spec.reward(mean_field, s, a, spec.environment))(
            a_idx
        )
    )(s_idx)

    def trans_for_sa(s, a):
        return jax.vmap(
            lambda n: spec.transition(mean_field, s, a, n, spec.environment)
        )(n_idx)

    next_states_san = jax.vmap(lambda s: jax.vmap(lambda a: trans_for_sa(s, a))(a_idx))(
        s_idx
    )

    def bellman_step_eval(carry, _):
        V_k = carry
        expected_V_sa = jnp.einsum("n,san->sa", noise_prob, V_k[next_states_san])
        Q_sa = rewards_sa + spec.environment.gamma * expected_V_sa
        V_k_plus_1 = jnp.einsum("sa,sa->s", policy, Q_sa)
        return V_k_plus_1, Q_sa

    V_initial = jnp.zeros(S)
    V_final, Q_sa_hist = jax.lax.scan(
        bellman_step_eval, V_initial, xs=None, length=spec.environment.horizon - 1
    )
    if spec.environment.horizon >= 3:
        return Q_sa_hist[-2]
    elif spec.environment.horizon == 2:
        return jnp.zeros((S, A))
    else:
        return jnp.zeros((S, A))


@jax_jit
def exploitability_jax(
    policy: jnp.ndarray,
    spec: EnvSpec,
    initial_mean_field: jnp.ndarray,
) -> jnp.ndarray:
    """Computes exploitability using the new JAX-native functions.

    Parameters:
    - policy: Policy array with shape (S, A)
    - spec: Static environment specification
    - initial_mean_field: Initial mean field distribution (required to avoid JIT caching issues)
    """
    policy = policy.reshape(spec.environment.N_states, spec.environment.N_actions)

    mean_field_pi = mean_field_by_transition_kernel_multi_jax(
        policy, spec, num_iterations=50, initial_mean_field=initial_mean_field
    )
    V_pi = V_eval_jax(policy, mean_field_pi, spec)
    V_opt, _ = Vpi_opt_jax(mean_field_pi, spec)

    return jnp.dot(mean_field_pi, V_opt) - jnp.dot(mean_field_pi, V_pi)


@partial(jax.jit, static_argnames=("spec", "num_particles"))
def exploitability_batch_jax(
    policies: jnp.ndarray,
    spec: EnvSpec,
    initial_mean_field: jnp.ndarray,
    num_particles: int,
) -> jnp.ndarray:
    """Computes exploitability for a batch of policies using vmap.

    Parameters:
    - policies: Batch of policies with shape (num_particles, S, A)
    - spec: Static environment specification
    - initial_mean_field: Initial mean field distribution
    - num_particles: Number of particles (static for JIT)

    Returns:
    - Array of exploitabilities with shape (num_particles,)
    """

    def single_exploitability(policy):
        return exploitability_jax(policy, spec, initial_mean_field)

    return jax.vmap(single_exploitability)(policies)
