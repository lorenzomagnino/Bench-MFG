"""Smoke tests for JAX compute kernels used in the MFG benchmark.

Covers:
  - Forward pass: each JAX function executes without error.
  - Backward pass: jax.grad / jax.value_and_grad produces finite gradients.
  - JIT compatibility: functions tolerate repeated jit-compiled calls.

Architecture under test (all in envs.mfg_model_class_jit):
  1. mean_field_by_transition_kernel_one_step_jax  – single MF update step
  2. mean_field_by_transition_kernel_multi_jax      – multi-step fori_loop MF update
  3. Vpi_opt_jax                                    – optimal value / policy
  4. V_eval_jax                                     – policy value evaluation
  5. Q_eval_jax                                     – Q-value evaluation
  6. exploitability_jax                             – exploitability scalar
  7. exploitability_batch_jax                       – vmap'd batch exploitability
"""

import pytest

np = pytest.importorskip("numpy", reason="numpy required")
jax = pytest.importorskip("jax", reason="jax required")
jnp = jax.numpy

LasryLionsChain = pytest.importorskip(
    "envs.lasry_lions_chain.lasry_lions_chain",
    reason="project envs package required",
).LasryLionsChain

_jit_env = pytest.importorskip(
    "envs.lasry_lions_chain.lasry_lions_chain_jit",
    reason="project envs package required",
)
transition_fn = _jit_env.transition_lasry_lions_chain
reward_fn = _jit_env.reward_lasry_lions_chain

_core = pytest.importorskip(
    "envs.mfg_model_class_jit",
    reason="project envs package required",
)
EnvSpec = _core.EnvSpec
mean_field_one_step = _core.mean_field_by_transition_kernel_one_step_jax
mean_field_multi = _core.mean_field_by_transition_kernel_multi_jax
Vpi_opt_jax = _core.Vpi_opt_jax
V_eval_jax = _core.V_eval_jax
Q_eval_jax = _core.Q_eval_jax
exploitability_jax = _core.exploitability_jax
exploitability_batch_jax = _core.exploitability_batch_jax

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_STATES = 8
N_ACTIONS = 3
N_PARTICLES = 4


@pytest.fixture(scope="module")
def env():
    mu_0 = np.ones(N_STATES) / N_STATES
    noise_prob = np.array([0.025, 0.95, 0.025])
    return LasryLionsChain(
        N_states=N_STATES,
        N_actions=N_ACTIONS,
        N_noises=3,
        horizon=4,
        mean_field=mu_0,
        noise_prob=noise_prob,
        crowd_penalty_coefficient=1.0,
        movement_penalty=0.1,
        center_attraction=0.5,
        gamma=0.9,
        is_noisy=True,
    )


@pytest.fixture(scope="module")
def spec(env):
    return EnvSpec(environment=env, transition=transition_fn, reward=reward_fn)


@pytest.fixture(scope="module")
def uniform_policy():
    return jnp.ones((N_STATES, N_ACTIONS)) / N_ACTIONS


@pytest.fixture(scope="module")
def uniform_mf():
    return jnp.ones(N_STATES) / N_STATES


# ---------------------------------------------------------------------------
# 1. mean_field_by_transition_kernel_one_step_jax
# ---------------------------------------------------------------------------


def test_mf_one_step_forward(env, spec, uniform_policy):
    """One-step MF forward pass returns a valid probability distribution."""
    # stationary_mean_field must be set on the env for the kernel to read it
    env.stationary_mean_field = np.ones(N_STATES) / N_STATES
    out = mean_field_one_step(uniform_policy, spec)
    assert out.shape == (N_STATES,)
    assert jnp.isclose(out.sum(), 1.0, atol=1e-5)
    assert jnp.all(out >= 0)


def test_mf_one_step_jit_stable(env, spec, uniform_policy):
    """Repeated JIT calls on the same spec produce identical results."""
    env.stationary_mean_field = np.ones(N_STATES) / N_STATES
    out1 = mean_field_one_step(uniform_policy, spec)
    out2 = mean_field_one_step(uniform_policy, spec)
    assert jnp.allclose(out1, out2, atol=1e-6)


# ---------------------------------------------------------------------------
# 2. mean_field_by_transition_kernel_multi_jax
# ---------------------------------------------------------------------------


def test_mf_multi_forward(spec, uniform_policy, uniform_mf):
    """Multi-step MF forward pass returns a valid probability distribution."""
    out = mean_field_multi(
        uniform_policy, spec, num_iterations=10, initial_mean_field=uniform_mf
    )
    assert out.shape == (N_STATES,)
    assert jnp.isclose(out.sum(), 1.0, atol=1e-5)
    assert jnp.all(out >= 0)


def test_mf_multi_backward(spec, uniform_policy, uniform_mf):
    """Gradients of the MF norm w.r.t. the policy are finite."""

    def loss(policy):
        mf = mean_field_multi(
            policy, spec, num_iterations=5, initial_mean_field=uniform_mf
        )
        return jnp.sum(mf**2)

    grad = jax.grad(loss)(uniform_policy)
    assert grad.shape == uniform_policy.shape
    assert jnp.all(jnp.isfinite(grad)), "Non-finite gradient in mf_multi backward pass"


# ---------------------------------------------------------------------------
# 3. Vpi_opt_jax
# ---------------------------------------------------------------------------


def test_vpi_opt_forward(spec, uniform_mf):
    """Vpi_opt_jax returns value array of shape (S,) and valid one-hot policy."""
    V, pi = Vpi_opt_jax(uniform_mf, spec)
    assert V.shape == (N_STATES,)
    assert pi.shape == (N_STATES, N_ACTIONS)
    # Policy rows should sum to 1 (one-hot)
    assert jnp.allclose(pi.sum(axis=1), jnp.ones(N_STATES), atol=1e-5)
    assert jnp.all(jnp.isfinite(V))


def test_vpi_opt_backward(spec, uniform_mf):
    """Gradient of mean(V_opt) w.r.t. mean field is finite."""

    def loss(mf):
        V, _ = Vpi_opt_jax(mf, spec)
        return jnp.mean(V)

    grad = jax.grad(loss)(uniform_mf)
    assert grad.shape == (N_STATES,)
    assert jnp.all(jnp.isfinite(grad)), "Non-finite gradient in Vpi_opt backward pass"


def test_vpi_opt_horizon_one_returns_valid_default_policy(env):
    """Horizon-1 problems should not fail and should match the Python fallback."""
    env_h1 = LasryLionsChain(
        N_states=env.N_states,
        N_actions=env.N_actions,
        N_noises=env.N_noises,
        horizon=1,
        mean_field=np.asarray(env.stationary_mean_field),
        noise_prob=np.asarray(env.noise_prob),
        crowd_penalty_coefficient=env.crowd_penalty_coefficient,
        movement_penalty=env.movement_penalty,
        center_attraction=env.center_attraction,
        gamma=env.gamma,
        is_noisy=env.is_noisy,
    )
    spec_h1 = EnvSpec(environment=env_h1, transition=transition_fn, reward=reward_fn)
    value, policy = Vpi_opt_jax(jnp.ones(N_STATES) / N_STATES, spec_h1)
    assert value.shape == (N_STATES,)
    assert jnp.allclose(value, jnp.zeros(N_STATES))
    assert policy.shape == (N_STATES, N_ACTIONS)
    assert jnp.allclose(policy[:, 0], jnp.ones(N_STATES))


# ---------------------------------------------------------------------------
# 4. V_eval_jax
# ---------------------------------------------------------------------------


def test_v_eval_forward(spec, uniform_policy, uniform_mf):
    """V_eval_jax returns a finite value array of shape (S,)."""
    V = V_eval_jax(uniform_policy, uniform_mf, spec)
    assert V.shape == (N_STATES,)
    assert jnp.all(jnp.isfinite(V))


def test_v_eval_backward(spec, uniform_policy, uniform_mf):
    """Gradient of mean(V_eval) w.r.t. policy is finite."""

    def loss(policy):
        return jnp.mean(V_eval_jax(policy, uniform_mf, spec))

    grad = jax.grad(loss)(uniform_policy)
    assert grad.shape == uniform_policy.shape
    assert jnp.all(jnp.isfinite(grad)), "Non-finite gradient in V_eval backward pass"


# ---------------------------------------------------------------------------
# 5. Q_eval_jax
# ---------------------------------------------------------------------------


def test_q_eval_forward(spec, uniform_policy, uniform_mf):
    """Q_eval_jax returns a finite Q-matrix of shape (S, A)."""
    Q = Q_eval_jax(uniform_policy, uniform_mf, spec)
    assert Q.shape == (N_STATES, N_ACTIONS)
    assert jnp.all(jnp.isfinite(Q))


def test_q_eval_backward(spec, uniform_policy, uniform_mf):
    """Gradient of sum(Q) w.r.t. policy is finite."""

    def loss(policy):
        return jnp.sum(Q_eval_jax(policy, uniform_mf, spec))

    grad = jax.grad(loss)(uniform_policy)
    assert grad.shape == uniform_policy.shape
    assert jnp.all(jnp.isfinite(grad)), "Non-finite gradient in Q_eval backward pass"


def test_q_eval_horizon_two_returns_zeros(env, uniform_policy, uniform_mf):
    """Horizon-2 problems should follow the documented zero-Q fallback."""
    env_h2 = LasryLionsChain(
        N_states=env.N_states,
        N_actions=env.N_actions,
        N_noises=env.N_noises,
        horizon=2,
        mean_field=np.asarray(env.stationary_mean_field),
        noise_prob=np.asarray(env.noise_prob),
        crowd_penalty_coefficient=env.crowd_penalty_coefficient,
        movement_penalty=env.movement_penalty,
        center_attraction=env.center_attraction,
        gamma=env.gamma,
        is_noisy=env.is_noisy,
    )
    spec_h2 = EnvSpec(environment=env_h2, transition=transition_fn, reward=reward_fn)
    q_values = Q_eval_jax(uniform_policy, uniform_mf, spec_h2)
    assert q_values.shape == (N_STATES, N_ACTIONS)
    assert jnp.allclose(q_values, jnp.zeros((N_STATES, N_ACTIONS)))


# ---------------------------------------------------------------------------
# 6. exploitability_jax
# ---------------------------------------------------------------------------


def test_exploitability_forward(spec, uniform_policy, uniform_mf):
    """exploitability_jax returns a finite non-negative scalar."""
    expl = exploitability_jax(uniform_policy, spec, initial_mean_field=uniform_mf)
    assert expl.shape == ()
    assert float(expl) >= 0.0
    assert jnp.isfinite(expl)


def test_exploitability_backward(spec, uniform_policy, uniform_mf):
    """Gradient of exploitability w.r.t. policy is finite."""

    def loss(policy):
        return exploitability_jax(policy, spec, initial_mean_field=uniform_mf)

    grad = jax.grad(loss)(uniform_policy)
    assert grad.shape == uniform_policy.shape
    assert jnp.all(jnp.isfinite(grad)), "Non-finite gradient in exploitability backward"


# ---------------------------------------------------------------------------
# 7. exploitability_batch_jax
# ---------------------------------------------------------------------------


def test_exploitability_batch_forward(spec, uniform_mf):
    """exploitability_batch_jax returns a finite array of shape (num_particles,)."""
    policies = jnp.ones((N_PARTICLES, N_STATES, N_ACTIONS)) / N_ACTIONS
    out = exploitability_batch_jax(
        policies, spec, uniform_mf, num_particles=N_PARTICLES
    )
    assert out.shape == (N_PARTICLES,)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out >= 0)


def test_exploitability_batch_consistent(spec, uniform_mf):
    """Batch and single exploitability agree on identical policies."""
    policy = jnp.ones((N_STATES, N_ACTIONS)) / N_ACTIONS
    policies = jnp.stack([policy] * N_PARTICLES)
    batch_out = exploitability_batch_jax(
        policies, spec, uniform_mf, num_particles=N_PARTICLES
    )
    single_out = exploitability_jax(policy, spec, initial_mean_field=uniform_mf)
    assert jnp.allclose(
        batch_out, single_out, atol=1e-5
    ), f"Batch and single exploitability diverge: {batch_out} vs {float(single_out)}"
