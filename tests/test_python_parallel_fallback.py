"""Tests for Python reference-path fallback when process pools are unavailable."""

import pytest

np = pytest.importorskip("numpy", reason="numpy is required for these tests")

_py_core = pytest.importorskip(
    "envs.mfg_model_class",
    reason="project envs package required",
)

LasryLionsChain = pytest.importorskip(
    "envs.lasry_lions_chain.lasry_lions_chain",
    reason="project envs package required",
).LasryLionsChain


class _FailingPool:
    def __init__(self, *args, **kwargs):
        raise PermissionError("simulated sandbox limitation")


@pytest.fixture
def env():
    n_states = 8
    mu_0 = np.ones(n_states) / n_states
    noise_prob = np.array([0.025, 0.95, 0.025])
    return LasryLionsChain(
        N_states=n_states,
        N_actions=3,
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


def test_transition_matrix_falls_back_to_sequential(monkeypatch, env):
    monkeypatch.setattr(_py_core, "ProcessPoolExecutor", _FailingPool)
    mean_field = np.asarray(env.stationary_mean_field)
    transition_matrix = env._build_transition_matrix(mean_field)
    assert transition_matrix.shape == (env.N_states, env.N_actions, env.N_noises)
    assert transition_matrix.dtype == np.intp


def test_reward_matrix_falls_back_to_sequential(monkeypatch, env):
    monkeypatch.setattr(_py_core, "ProcessPoolExecutor", _FailingPool)
    mean_field = np.asarray(env.stationary_mean_field)
    reward_matrix = env._build_reward_matrix(mean_field)
    assert reward_matrix.shape == (env.N_states, env.N_actions)
    assert np.all(np.isfinite(reward_matrix))


def test_reference_algorithms_work_without_process_pool(monkeypatch, env):
    monkeypatch.setattr(_py_core, "ProcessPoolExecutor", _FailingPool)
    policy = np.ones((env.N_states, env.N_actions)) / env.N_actions
    mean_field = env.mean_field_by_transition_kernel(policy, num_transition_steps=5)
    q_values = env.Q_eval(policy, mean_field)
    value, greedy_policy = env.Vpi_opt(mean_field)

    assert mean_field.shape == (env.N_states,)
    assert np.isclose(mean_field.sum(), 1.0, atol=1e-6)
    assert q_values.shape == (env.N_states, env.N_actions)
    assert value.shape == (env.N_states,)
    assert greedy_policy.shape == (env.N_states, env.N_actions)
    assert np.allclose(greedy_policy.sum(axis=1), 1.0)
