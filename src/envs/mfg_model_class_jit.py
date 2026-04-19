from collections.abc import Callable
from dataclasses import dataclass, field
from functools import partial

from envs.mfg_model_class import MFGStationary
import jax
import jax.numpy as jnp

jax_jit = partial(jax.jit, static_argnames="spec")


def get_jax_device(device_str: str = "cpu"):
    """Return the first JAX device matching the requested backend.

    Args:
        device_str: ``"cuda"`` maps to the JAX GPU backend; anything else uses CPU.
                    Falls back to CPU when the requested backend is unavailable.
    """
    backend = "gpu" if device_str == "cuda" else "cpu"
    try:
        return jax.devices(backend)[0]
    except RuntimeError:
        return jax.devices("cpu")[0]


@dataclass(frozen=True)
class EnvSpec:
    environment: MFGStationary
    transition: Callable[
        [jnp.ndarray | None, int, int, int, MFGStationary | None], jnp.ndarray
    ] = field(compare=False)
    reward: Callable[
        [jnp.ndarray | None, int, int, MFGStationary | None], jnp.ndarray
    ] = field(compare=False)


def _reward_matrix(mean_field: jnp.ndarray, spec: EnvSpec) -> jnp.ndarray:
    """Build the dense reward matrix R[s, a] for a fixed mean field."""
    S, A = spec.environment.N_states, spec.environment.N_actions
    s_idx = jnp.arange(S)
    a_idx = jnp.arange(A)
    return jax.vmap(
        lambda s: jax.vmap(lambda a: spec.reward(mean_field, s, a, spec.environment))(
            a_idx
        )
    )(s_idx)


def _transition_tensor(mean_field: jnp.ndarray, spec: EnvSpec) -> jnp.ndarray:
    """Build the dense transition tensor T[s, a, n] -> next_state."""
    S, A, N = (
        spec.environment.N_states,
        spec.environment.N_actions,
        spec.environment.N_noises,
    )
    s_idx = jnp.arange(S)
    a_idx = jnp.arange(A)
    n_idx = jnp.arange(N)

    def trans_for_sa(s, a):
        return jax.vmap(
            lambda n: spec.transition(mean_field, s, a, n, spec.environment)
        )(n_idx)

    return jax.vmap(lambda s: jax.vmap(lambda a: trans_for_sa(s, a))(a_idx))(s_idx)


def _normalized_transition_step(
    policy: jnp.ndarray,
    mean_field: jnp.ndarray,
    next_states_san: jnp.ndarray,
    noise_prob: jnp.ndarray,
    num_states: int,
) -> jnp.ndarray:
    """Apply a transition tensor to a policy/mean-field pair and renormalize."""
    probs = policy[:, :, None] * mean_field[:, None, None] * noise_prob[None, None, :]
    new_state_dist = (
        jnp.zeros(num_states, dtype=probs.dtype).at[next_states_san].add(probs)
    )
    return new_state_dist / new_state_dist.sum()


def _q_from_value(
    value: jnp.ndarray,
    rewards_sa: jnp.ndarray,
    next_states_san: jnp.ndarray,
    noise_prob: jnp.ndarray,
    gamma: float,
) -> jnp.ndarray:
    expected_v_sa = jnp.einsum("n,san->sa", noise_prob, value[next_states_san])
    return rewards_sa + gamma * expected_v_sa


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
    S = spec.environment.N_states
    stationary_mf = jnp.asarray(spec.environment.stationary_mean_field)
    noise_prob = jnp.asarray(spec.environment.noise_prob)
    next_states_san = _transition_tensor(stationary_mf, spec)
    return _normalized_transition_step(
        policy, stationary_mf, next_states_san, noise_prob, S
    )


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
    S, A = spec.environment.N_states, spec.environment.N_actions
    policy = policy.reshape(S, A)
    noise_prob = jnp.asarray(spec.environment.noise_prob)

    def one_step(current_mf: jnp.ndarray) -> jnp.ndarray:
        next_states_san = _transition_tensor(current_mf, spec)
        return _normalized_transition_step(
            policy, current_mf, next_states_san, noise_prob, S
        )

    def body_fun(_, mf):
        return one_step(mf)

    mf_final = jax.lax.fori_loop(0, num_iterations, body_fun, initial_mean_field)
    return mf_final


@jax_jit
def Vpi_opt_jax(
    mean_field: jnp.ndarray,
    spec: EnvSpec,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Computes the optimal value function using jax.lax.scan."""
    S, A = spec.environment.N_states, spec.environment.N_actions
    if spec.environment.horizon < 2:
        zero_actions = jnp.zeros(S, dtype=jnp.int32)
        return jnp.zeros(S), jax.nn.one_hot(zero_actions, num_classes=A)

    noise_prob = jnp.asarray(spec.environment.noise_prob)
    rewards_sa = _reward_matrix(mean_field, spec)
    next_states_san = _transition_tensor(mean_field, spec)

    def bellman_step(V_k, _):
        Q_sa = _q_from_value(
            V_k, rewards_sa, next_states_san, noise_prob, spec.environment.gamma
        )
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
    S = spec.environment.N_states
    noise_prob = jnp.asarray(spec.environment.noise_prob)
    rewards_sa = _reward_matrix(mean_field, spec)
    next_states_san = _transition_tensor(mean_field, spec)

    def bellman_step_eval(V_k, _):
        Q_sa = _q_from_value(
            V_k, rewards_sa, next_states_san, noise_prob, spec.environment.gamma
        )
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
    S, A = spec.environment.N_states, spec.environment.N_actions
    if spec.environment.horizon < 3:
        return jnp.zeros((S, A))

    noise_prob = jnp.asarray(spec.environment.noise_prob)
    policy = policy.reshape(S, A)
    rewards_sa = _reward_matrix(mean_field, spec)
    next_states_san = _transition_tensor(mean_field, spec)

    def bellman_step_eval(carry, _):
        V_k = carry
        Q_sa = _q_from_value(
            V_k, rewards_sa, next_states_san, noise_prob, spec.environment.gamma
        )
        V_k_plus_1 = jnp.einsum("sa,sa->s", policy, Q_sa)
        return V_k_plus_1, Q_sa

    V_initial = jnp.zeros(S)
    _, Q_sa_hist = jax.lax.scan(
        bellman_step_eval, V_initial, xs=None, length=spec.environment.horizon - 1
    )
    return Q_sa_hist[-2]


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


@partial(jax.jit, static_argnames="spec")
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


def exploitability_batch_pmap(
    policies: jnp.ndarray,
    spec: EnvSpec,
    initial_mean_field: jnp.ndarray,
    num_particles: int,
) -> jnp.ndarray:
    """Compute exploitability for a batch of policies, sharding across all JAX devices.

    Uses ``jax.pmap`` to distribute particle evaluation across multiple devices when
    more than one device is available.  Falls back to the ``vmap`` implementation on
    single-device setups (the common CPU case).

    Parameters:
    - policies: Batch of policies with shape (num_particles, S, A)
    - spec: Static environment specification
    - initial_mean_field: Initial mean field distribution
    - num_particles: Number of particles

    Returns:
    - Array of exploitabilities with shape (num_particles,)
    """
    n_devices = len(jax.devices())
    if n_devices == 1:
        return exploitability_batch_jax(
            policies, spec, initial_mean_field, num_particles
        )

    # Pad particle count to the nearest multiple of n_devices for even sharding.
    remainder = num_particles % n_devices
    pad = (n_devices - remainder) % n_devices
    if pad:
        padding = jnp.zeros((pad,) + policies.shape[1:], dtype=policies.dtype)
        policies_padded = jnp.concatenate([policies, padding], axis=0)
    else:
        policies_padded = policies

    per_device = policies_padded.shape[0] // n_devices
    policies_sharded = policies_padded.reshape(
        n_devices, per_device, *policies.shape[1:]
    )

    def _per_device_batch(shard):
        return jax.vmap(lambda p: exploitability_jax(p, spec, initial_mean_field))(
            shard
        )

    results = jax.pmap(_per_device_batch)(policies_sharded)
    return results.reshape(-1)[:num_particles]
