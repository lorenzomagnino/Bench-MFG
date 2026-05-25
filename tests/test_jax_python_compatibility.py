"""Pytest suite validating JAX and Python environment implementations are equivalent.

These tests verify that the JAX-accelerated and pure-Python code paths produce
numerically consistent results for the core MFG primitives.

Requires: numpy, jax, and the project benchmfg.envs package.
Skipped automatically when dependencies are not installed.
"""

import pytest

# Guard: skip the entire module if heavy dependencies are absent.
np = pytest.importorskip("numpy", reason="numpy is required for these tests")
jax = pytest.importorskip("jax", reason="jax is required for these tests")
jnp = jax.numpy

LasryLionsChain = pytest.importorskip(
    "benchmfg.envs.lasry_lions_chain.lasry_lions_chain",
    reason="project benchmfg.envs package required",
).LasryLionsChain

_jit_mod = pytest.importorskip(
    "benchmfg.envs.lasry_lions_chain.lasry_lions_chain_jit",
    reason="project benchmfg.envs package required",
)
transition_lasry_lions_chain = _jit_mod.transition_lasry_lions_chain
reward_lasry_lions_chain = _jit_mod.reward_lasry_lions_chain

MFGStationary = pytest.importorskip(
    "benchmfg.envs.mfg_model_class",
    reason="project benchmfg.envs package required",
).MFGStationary

_jit_core = pytest.importorskip(
    "benchmfg.envs.mfg_model_class_jit",
    reason="project benchmfg.envs package required",
)
EnvSpec = _jit_core.EnvSpec
Q_eval_jax = _jit_core.Q_eval_jax
Vpi_opt_jax = _jit_core.Vpi_opt_jax
exploitability_jax = _jit_core.exploitability_jax
mean_field_by_transition_kernel_multi_jax = (
    _jit_core.mean_field_by_transition_kernel_multi_jax
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_STATES = 10
N_ACTIONS = 3
TOLERANCE = 1e-4

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def env():
    """Construct a small LasryLionsChain environment for testing."""
    mu_0 = np.ones(N_STATES) / N_STATES
    noise_prob = np.array([0.025, 0.95, 0.025])
    return LasryLionsChain(
        N_states=N_STATES,
        N_actions=N_ACTIONS,
        N_noises=3,
        horizon=5,
        mean_field=mu_0,
        noise_prob=noise_prob,
        crowd_penalty_coefficient=1.0,
        movement_penalty=0.1,
        center_attraction=0.5,
        gamma=0.9,
        is_noisy=True,
    )


@pytest.fixture(scope="module")
def env_spec(env):
    """Wrap env in an EnvSpec for JAX kernels."""
    return EnvSpec(
        environment=env,
        transition=transition_lasry_lions_chain,
        reward=reward_lasry_lions_chain,
    )


@pytest.fixture
def uniform_policy():
    return np.ones((N_STATES, N_ACTIONS)) / N_ACTIONS


@pytest.fixture
def uniform_mean_field():
    return np.ones(N_STATES) / N_STATES


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_vpi_opt_value_function(env, env_spec) -> None:
    """Python and JAX value functions agree within tolerance."""
    mean_field = np.ones(N_STATES) / N_STATES
    v_py, _ = env.Vpi_opt(mean_field)
    v_jax = np.asarray(Vpi_opt_jax(jnp.asarray(mean_field), env_spec)[0])
    assert np.max(np.abs(v_py - v_jax)) < TOLERANCE


def test_vpi_opt_policy_actions(env, env_spec) -> None:
    """Python and JAX greedy policies select the same actions."""
    mean_field = np.ones(N_STATES) / N_STATES
    _, pi_py = env.Vpi_opt(mean_field)
    _, pi_jax = Vpi_opt_jax(jnp.asarray(mean_field), env_spec)
    pi_jax = np.asarray(pi_jax)
    assert np.all(np.argmax(pi_py, axis=1) == np.argmax(pi_jax, axis=1))


def test_q_eval_shape(env, env_spec, uniform_policy, uniform_mean_field) -> None:
    """Q-value matrices from both implementations have the expected shape."""
    q_py = env.Q_eval(uniform_policy, uniform_mean_field)
    q_jax = np.asarray(
        Q_eval_jax(
            jnp.asarray(uniform_policy), jnp.asarray(uniform_mean_field), env_spec
        )
    )
    assert q_py.shape == (N_STATES, N_ACTIONS)
    assert q_jax.shape == q_py.shape


def test_q_eval_values(env, env_spec, uniform_policy, uniform_mean_field) -> None:
    """Python and JAX Q-values agree within tolerance."""
    q_py = env.Q_eval(uniform_policy, uniform_mean_field)
    q_jax = np.asarray(
        Q_eval_jax(
            jnp.asarray(uniform_policy), jnp.asarray(uniform_mean_field), env_spec
        )
    )
    rel_error = np.max(np.abs(q_py - q_jax)) / (np.abs(q_py).max() + 1e-10)
    assert rel_error < TOLERANCE


def test_mean_field_update_valid_distribution(env, env_spec, uniform_policy) -> None:
    """Both implementations return a valid probability distribution."""
    initial_mf = np.ones(N_STATES) / N_STATES
    env.stationary_mean_field = initial_mf.copy()
    mf_py = env.mean_field_by_transition_kernel(uniform_policy, num_transition_steps=20)

    mf_jax = np.asarray(
        mean_field_by_transition_kernel_multi_jax(
            jnp.asarray(uniform_policy),
            env_spec,
            num_iterations=20,
            initial_mean_field=jnp.asarray(initial_mf),
        )
    )

    assert np.isclose(mf_py.sum(), 1.0, atol=1e-6)
    assert np.all(mf_py >= 0)
    assert np.isclose(mf_jax.sum(), 1.0, atol=1e-6)
    assert np.all(mf_jax >= 0)


def test_exploitability_close(env, env_spec, uniform_policy) -> None:
    """Python and JAX exploitability values agree within a relaxed tolerance."""
    expl_py = env.exploitability(uniform_policy)
    expl_jax = float(
        exploitability_jax(
            jnp.asarray(uniform_policy),
            env_spec,
            initial_mean_field=jnp.asarray(env.stationary_mean_field),
        )
    )
    rel_error = abs(expl_py - expl_jax) / (abs(expl_py) + 1e-10)
    assert rel_error < 1e-3
