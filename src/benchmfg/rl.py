from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from benchmfg.envs.mfg_model_class import MFGStationary

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised when optional extra is absent
    gym = None
    spaces = None


def _require_gym():
    if gym is None or spaces is None:
        raise ImportError("Install BenchMFG with the 'rl' extra to use Gym envs.")


def _require_sb3():
    try:
        from stable_baselines3 import PPO
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install BenchMFG with the 'rl' extra to use PPO.") from exc
    return PPO


def _require_dqn():
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError("Install BenchMFG with the 'rl' extra to use DQN.") from exc
    return DQN


def _require_vec_env():
    try:
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise ImportError(
            "Install BenchMFG with the 'rl' extra to normalize rewards."
        ) from exc
    return DummyVecEnv, VecNormalize


class BestResponseSolver(Protocol):
    def solve(self, mean_field: np.ndarray) -> np.ndarray: ...


def normalize_tabular_policy(policy: np.ndarray, n_states: int, n_actions: int):
    policy = np.asarray(policy, dtype=float).reshape(n_states, n_actions)
    policy = np.clip(policy, 0.0, None)
    row_sums = policy.sum(axis=1, keepdims=True)
    if np.any(row_sums <= 0):
        raise ValueError("best-response policy rows must have positive mass")
    return policy / row_sums


def _make_fixed_mean_field_env(
    model: MFGStationary,
    mean_field: np.ndarray,
    *,
    reset_uniform_eps: float,
    episode_length: int | None,
    normalize_reward: bool,
    n_envs: int,
):
    n_envs = int(n_envs)
    if n_envs < 1:
        raise ValueError("n_envs must be >= 1")

    def make_env():
        return FixedMeanFieldEnv(
            model,
            mean_field,
            reset_uniform_eps=reset_uniform_eps,
            episode_length=episode_length,
        )

    if not normalize_reward and n_envs == 1:
        return make_env()
    DummyVecEnv, VecNormalize = _require_vec_env()
    env = DummyVecEnv([make_env for _ in range(n_envs)])
    if not normalize_reward:
        return env
    return VecNormalize(
        env,
        norm_obs=False,
        norm_reward=True,
        gamma=model.gamma,
    )


class FixedMeanFieldEnv(gym.Env if gym is not None else object):
    """Gymnasium wrapper for the fixed-mean-field MDP behind an MFG best response."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        model: MFGStationary,
        mean_field: np.ndarray,
        reset_uniform_eps: float = 0.0,
        episode_length: int | None = None,
    ):
        _require_gym()
        self.model = model
        self.mean_field = np.asarray(mean_field, dtype=float).reshape(model.N_states)
        self.mean_field = self.mean_field / self.mean_field.sum()
        self.reset_distribution = (
            1.0 - reset_uniform_eps
        ) * self.mean_field + reset_uniform_eps * np.full(
            model.N_states, 1.0 / model.N_states
        )
        self.reset_distribution = (
            self.reset_distribution / self.reset_distribution.sum()
        )
        self.episode_length = int(episode_length or model.horizon)
        self.observation_space = spaces.Discrete(model.N_states)
        self.action_space = spaces.Discrete(model.N_actions)
        self._rng = np.random.default_rng()
        self._state = 0
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._state = int(
            self._rng.choice(self.model.N_states, p=self.reset_distribution)
        )
        return self._state, {}

    def step(self, action):
        action = int(action)
        if not self.action_space.contains(action):
            raise ValueError(f"action {action} out of range")
        noise = int(self._rng.choice(self.model.N_noises, p=self.model.noise_prob))
        reward = float(
            self.model.reward(self.mean_field, state=self._state, action=action)
        )
        self._state = int(
            self.model.transition(
                self.mean_field, state=self._state, action=action, noise=noise
            )
        )
        self._steps += 1
        return self._state, reward, False, self._steps >= self.episode_length, {}


class ContinuousBeachBarEnv(gym.Env if gym is not None else object):
    """Small continuous-control beach-bar env for PPO/Gym verification."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        horizon: int = 100,
        max_speed: float = 0.05,
        bar_locations: tuple[float, float] = (0.25, 0.75),
        movement_penalty: float = 0.1,
        attraction: float = 1.0,
    ):
        _require_gym()
        self.horizon = int(horizon)
        self.max_speed = float(max_speed)
        self.bar_locations = np.asarray(bar_locations, dtype=np.float32)
        self.movement_penalty = float(movement_penalty)
        self.attraction = float(attraction)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(1,), dtype=np.float32)
        self.action_space = spaces.Box(
            -self.max_speed, self.max_speed, shape=(1,), dtype=np.float32
        )
        self._rng = np.random.default_rng()
        self._position = np.array([0.5], dtype=np.float32)
        self._steps = 0

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._steps = 0
        self._position = self._rng.uniform(0.0, 1.0, size=1).astype(np.float32)
        return self._position.copy(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(1)
        action = np.clip(action, -self.max_speed, self.max_speed)
        self._position = np.clip(self._position + action, 0.0, 1.0).astype(np.float32)
        dist = float(np.min(np.abs(self._position[0] - self.bar_locations)))
        reward = -self.attraction * dist - self.movement_penalty * float(abs(action[0]))
        self._steps += 1
        return (
            self._position.copy(),
            reward,
            False,
            self._steps >= self.horizon,
            {},
        )


@dataclass
class PPOBestResponse:
    model: MFGStationary
    total_timesteps: int = 10_000
    n_steps: int = 128
    batch_size: int = 64
    learning_rate: float = 3e-4
    n_epochs: int = 10
    seed: int = 0
    reset_uniform_eps: float = 0.1
    episode_length: int | None = None
    normalize_reward: bool = True
    device: str = "cpu"
    n_envs: int = 1
    warm_start: bool = False
    agent: object | None = None
    vec_env: object | None = None

    def solve(self, mean_field: np.ndarray) -> np.ndarray:
        PPO = _require_sb3()
        env = _make_fixed_mean_field_env(
            self.model,
            mean_field,
            reset_uniform_eps=self.reset_uniform_eps,
            episode_length=self.episode_length,
            normalize_reward=self.normalize_reward,
            n_envs=self.n_envs,
        )
        is_new_agent = self.agent is None or not self.warm_start
        if is_new_agent:
            self.agent = PPO(
                "MlpPolicy",
                env,
                gamma=self.model.gamma,
                n_steps=self.n_steps,
                batch_size=self.batch_size,
                learning_rate=self.learning_rate,
                n_epochs=self.n_epochs,
                seed=self.seed,
                device=self.device,
                verbose=0,
            )
        else:
            self.agent.set_env(env)
        self.agent.learn(
            total_timesteps=self.total_timesteps,
            reset_num_timesteps=is_new_agent,
        )
        policy = np.zeros((self.model.N_states, self.model.N_actions), dtype=float)
        for state in range(self.model.N_states):
            action, _ = self.agent.predict(state, deterministic=True)
            policy[state, int(np.asarray(action).item())] = 1.0
        return policy


@dataclass
class DQNBestResponse:
    model: MFGStationary
    total_timesteps: int = 10_000
    learning_rate: float = 1e-3
    buffer_size: int = 10_000
    learning_starts: int = 100
    batch_size: int = 32
    train_freq: int = 1
    gradient_steps: int = 1
    exploration_fraction: float = 0.1
    exploration_final_eps: float = 0.05
    seed: int = 0
    reset_uniform_eps: float = 0.1
    episode_length: int | None = None
    normalize_reward: bool = False
    device: str = "cpu"
    n_envs: int = 1
    warm_start: bool = True
    agent: object | None = None

    def solve(self, mean_field: np.ndarray) -> np.ndarray:
        DQN = _require_dqn()
        env = _make_fixed_mean_field_env(
            self.model,
            mean_field,
            reset_uniform_eps=self.reset_uniform_eps,
            episode_length=self.episode_length,
            normalize_reward=self.normalize_reward,
            n_envs=self.n_envs,
        )
        old_policy = (
            self.agent.policy.state_dict()
            if self.warm_start and self.agent is not None
            else None
        )
        agent = DQN(
            "MlpPolicy",
            env,
            gamma=self.model.gamma,
            learning_rate=self.learning_rate,
            buffer_size=self.buffer_size,
            learning_starts=self.learning_starts,
            batch_size=self.batch_size,
            train_freq=self.train_freq,
            gradient_steps=self.gradient_steps,
            exploration_fraction=self.exploration_fraction,
            exploration_final_eps=self.exploration_final_eps,
            seed=self.seed,
            device=self.device,
            verbose=0,
        )
        if old_policy is not None:
            agent.policy.load_state_dict(old_policy)
        agent.learn(total_timesteps=self.total_timesteps)
        self.agent = agent
        policy = np.zeros((self.model.N_states, self.model.N_actions), dtype=float)
        for state in range(self.model.N_states):
            action, _ = agent.predict(state, deterministic=True)
            policy[state, int(np.asarray(action).item())] = 1.0
        return policy
