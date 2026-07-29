from benchmfg.api import load_config, make_environment, make_solver
from benchmfg.envs.contraction_game.contraction_game import ContractionGame
from benchmfg.learner.jax.fp_jax import DampedFP_jax
from benchmfg.learner.python.fp_py import DampedFP_python
from benchmfg.rl import (
    ContinuousBeachBarEnv,
    DQNBestResponse,
    FixedMeanFieldEnv,
    PPOBestResponse,
)
from benchmfg.utility.create_solver import get_env_spec
import numpy as np
import pytest


def _contraction_game(horizon: int = 3):
    return ContractionGame(
        N_states=2,
        N_actions=2,
        N_noises=1,
        horizon=horizon,
        mean_field=np.array([0.6, 0.4]),
        noise_prob=np.array([1.0]),
        switching_cost=0.1,
        congestion_coefficient=0.1,
        gamma=0.9,
    )


class _SwitchSolver:
    def solve(self, mean_field):
        return np.array([[0.0, 1.0], [0.0, 1.0]])


def test_fixed_mean_field_env_matches_wrapped_model():
    pytest.importorskip("gymnasium")
    model = _contraction_game(horizon=2)
    env = FixedMeanFieldEnv(model, np.array([0.6, 0.4]))

    obs, info = env.reset(seed=0)
    assert info == {}

    next_obs, reward, terminated, truncated, info = env.step(1)
    assert next_obs == 1 - obs
    assert reward == pytest.approx(model.reward(env.mean_field, obs, 1))
    assert terminated is False
    assert truncated is False
    assert info == {}


def test_continuous_beach_bar_env_has_box_spaces():
    pytest.importorskip("gymnasium")
    env = ContinuousBeachBarEnv(horizon=1)

    obs, _ = env.reset(seed=0)
    next_obs, reward, terminated, truncated, _ = env.step(np.array([0.01]))

    assert env.observation_space.contains(obs)
    assert env.observation_space.contains(next_obs)
    assert env.action_space.contains(np.array([0.01], dtype=np.float32))
    assert np.isfinite(reward)
    assert terminated is False
    assert truncated is True


def test_python_dampedfp_uses_injected_best_response():
    model = _contraction_game()
    solver = DampedFP_python(
        model=model,
        initial_policy=np.full((2, 2), 0.5),
        num_iterations=1,
        lambda_schedule="pure",
        num_transition_steps=1,
        best_response_solver=_SwitchSolver(),
    )

    policy, _, _ = solver.eval()

    assert np.allclose(policy, [[0.0, 1.0], [0.0, 1.0]])


def test_jax_dampedfp_uses_injected_best_response():
    model = _contraction_game()
    solver = DampedFP_jax(
        env_spec=get_env_spec(model, "ContractionGame"),
        initial_policy=np.full((2, 2), 0.5),
        num_iterations=1,
        lambda_schedule="pure",
        num_transition_steps=1,
        best_response_solver=_SwitchSolver(),
    )

    policy, _, _ = solver.eval()

    assert np.allclose(policy, [[0.0, 1.0], [0.0, 1.0]])


def test_create_solver_wires_ppo_best_response():
    cfg = load_config(
        [
            "environment=contraction_game",
            "algorithm=damped_fixed_point",
            "algorithm.dampedfp.best_response=ppo",
        ]
    )
    env, initial_policy = make_environment(cfg)
    solver = make_solver(cfg, environment=env, initial_policy=initial_policy)

    assert isinstance(solver.best_response_solver, PPOBestResponse)
    assert solver.best_response_solver.n_envs == 1
    assert solver.best_response_solver.warm_start is False


def test_create_solver_wires_dqn_best_response():
    cfg = load_config(
        [
            "environment=contraction_game",
            "algorithm=damped_fixed_point",
            "algorithm.dampedfp.best_response=dqn",
        ]
    )
    env, initial_policy = make_environment(cfg)
    solver = make_solver(cfg, environment=env, initial_policy=initial_policy)

    assert isinstance(solver.best_response_solver, DQNBestResponse)
    assert solver.best_response_solver.n_envs == 1
    assert solver.best_response_solver.warm_start is True


def test_ppo_best_response_smoke():
    pytest.importorskip("stable_baselines3")
    model = _contraction_game(horizon=4)

    policy = PPOBestResponse(
        model,
        total_timesteps=16,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
    ).solve(model.stationary_mean_field)

    assert policy.shape == (2, 2)
    assert np.allclose(policy.sum(axis=1), 1.0)


def test_ppo_best_response_uses_model_gamma(monkeypatch):
    model = _contraction_game(horizon=4)
    seen = {}

    class _Agent:
        def __init__(self, *_args, gamma=None, device=None, **_kwargs):
            seen["gamma"] = gamma
            seen["device"] = device

        def learn(self, **_kwargs):
            return self

        def predict(self, _state, deterministic=True):
            return 0, None

        def set_env(self, _env):
            pass

    monkeypatch.setattr("benchmfg.rl._require_sb3", lambda: _Agent)

    PPOBestResponse(
        model, total_timesteps=1, normalize_reward=False, device="cuda"
    ).solve(model.stationary_mean_field)

    assert seen["gamma"] == model.gamma
    assert seen["device"] == "cuda"


def test_ppo_warm_start_reuses_agent(monkeypatch):
    model = _contraction_game(horizon=4)
    seen = {"agents": 0, "set_env": 0, "reset_flags": []}

    class _Agent:
        def __init__(self, *_args, **_kwargs):
            seen["agents"] += 1

        def learn(self, **kwargs):
            seen["reset_flags"].append(kwargs["reset_num_timesteps"])
            return self

        def predict(self, _state, deterministic=True):
            return 0, None

        def set_env(self, _env):
            seen["set_env"] += 1

    monkeypatch.setattr("benchmfg.rl._require_sb3", lambda: _Agent)

    solver = PPOBestResponse(
        model,
        total_timesteps=1,
        normalize_reward=False,
        warm_start=True,
    )
    solver.solve(model.stationary_mean_field)
    solver.solve(model.stationary_mean_field)

    assert seen == {"agents": 1, "set_env": 1, "reset_flags": [True, False]}


def test_dqn_best_response_smoke():
    pytest.importorskip("stable_baselines3")
    model = _contraction_game(horizon=4)

    policy = DQNBestResponse(
        model,
        total_timesteps=16,
        learning_starts=0,
        batch_size=8,
    ).solve(model.stationary_mean_field)

    assert policy.shape == (2, 2)
    assert np.allclose(policy.sum(axis=1), 1.0)


def test_dqn_best_response_uses_model_gamma(monkeypatch):
    model = _contraction_game(horizon=4)
    seen = {}

    class _Policy:
        def load_state_dict(self, _state):
            pass

        def state_dict(self):
            return {}

    class _Agent:
        policy = _Policy()

        def __init__(self, *_args, gamma=None, device=None, **_kwargs):
            seen["gamma"] = gamma
            seen["device"] = device

        def learn(self, **_kwargs):
            return self

        def predict(self, _state, deterministic=True):
            return 0, None

    monkeypatch.setattr("benchmfg.rl._require_dqn", lambda: _Agent)

    DQNBestResponse(
        model, total_timesteps=1, normalize_reward=False, device="cuda"
    ).solve(model.stationary_mean_field)

    assert seen["gamma"] == model.gamma
    assert seen["device"] == "cuda"


def test_dqn_warm_start_copies_previous_policy(monkeypatch):
    model = _contraction_game(horizon=4)
    seen = {"state_dict": 0, "load_state_dict": 0}

    class _Policy:
        def load_state_dict(self, _state):
            seen["load_state_dict"] += 1

        def state_dict(self):
            seen["state_dict"] += 1
            return {"weights": 1}

    class _Agent:
        def __init__(self, *_args, **_kwargs):
            self.policy = _Policy()

        def learn(self, **_kwargs):
            return self

        def predict(self, _state, deterministic=True):
            return 0, None

    monkeypatch.setattr("benchmfg.rl._require_dqn", lambda: _Agent)

    solver = DQNBestResponse(model, total_timesteps=1, normalize_reward=False)
    solver.solve(model.stationary_mean_field)
    solver.solve(model.stationary_mean_field)

    assert seen == {"state_dict": 1, "load_state_dict": 1}


def test_ppo_continuous_beach_bar_smoke():
    PPO = pytest.importorskip("stable_baselines3").PPO
    env = ContinuousBeachBarEnv(horizon=4)

    model = PPO("MlpPolicy", env, n_steps=8, batch_size=8, n_epochs=1, verbose=0)
    model.learn(total_timesteps=16)
    obs, _ = env.reset(seed=0)
    action, _ = model.predict(obs, deterministic=True)

    assert env.action_space.contains(np.asarray(action, dtype=np.float32))
