# Extending BenchMFG

This guide shows the minimal path for adding a new environment or a new JAX
algorithm. BenchMFG uses Hydra config groups, so code and YAML registration both
matter.

## Add A New Environment

Create a Python environment class under `src/benchmfg/envs/<name>/`.

```python
# src/benchmfg/envs/my_game/my_game.py
import numpy as np

from benchmfg.envs.mfg_model_class import MFGStationary


class MyGame(MFGStationary):
    def __init__(self, N_states, N_actions, N_noises, horizon, mean_field, noise_prob, gamma=0.99):
        super().__init__(
            N_states=N_states,
            N_actions=N_actions,
            N_noises=N_noises,
            horizon=horizon,
            mean_field=mean_field,
            noise_prob=noise_prob,
            gamma=gamma,
        )

    def transition(self, mean_field: np.ndarray, state: int, action: int, noise: int) -> int:
        next_state = state + action - 1
        return int(np.clip(next_state, 0, self.N_states - 1))

    def reward(self, mean_field: np.ndarray, state: int, action: int) -> float:
        return float(state - 0.1 * abs(action - 1))
```

Add JAX transition/reward functions. These are used by the JAX solvers through
`EnvSpec`.

```python
# src/benchmfg/envs/my_game/my_game_jit.py
import jax.numpy as jnp

from benchmfg.envs.mfg_model_class import MFGStationary


def transition_my_game(mean_field, state: int, action: int, noise: int, environment: MFGStationary):
    next_state = state + action - 1
    return jnp.clip(next_state, 0, environment.N_states - 1).astype(jnp.int32)


def reward_my_game(mean_field, state: int, action: int, environment: MFGStationary):
    return state - 0.1 * jnp.abs(action - 1)
```

Register the environment in four places:

1. Add a YAML file under `src/benchmfg/config/environment/my_game.yaml`.
2. Add any config dataclass fields in `src/benchmfg/conf/environment/environment_schema.py`.
3. Add a constructor helper and branch in `src/benchmfg/utility/config_utils.py` and `create_environment.py`.
4. Add the JAX functions to `ENV_JIT_FUNCTIONS` in `src/benchmfg/utility/create_solver.py`.

Minimal YAML:

```yaml
# @package _global_
environment:
  name: MyGame
  num_states: 10
  num_actions: 3
  num_noises: 1
  horizon: 50
  gamma: 0.99
  grid:
    is_grid: false
  dynamics:
    is_noisy: false
    noise_probabilities: [1.0]
  initial_distribution:
    type: uniform
```

Smoke test:

```bash
benchmfg train environment=my_game algorithm=omd device=cpu algorithm.omd.num_iterations=2
```

## Add A New JAX Algorithm

Create a solver under `src/benchmfg/learner/jax/`. A JAX solver should accept an
`EnvSpec`, return NumPy arrays, and expose an `eval(logger=None)` method.

```python
# src/benchmfg/learner/jax/my_algo_jax.py
import numpy as np

from benchmfg.envs.mfg_model_class_jit import (
    EnvSpec,
    exploitability_jax,
    mean_field_by_transition_kernel_multi_jax,
)


class MyAlgo_jax:
    def __init__(self, env_spec: EnvSpec, initial_policy: np.ndarray, num_iterations: int, jax_device=None):
        self.env_spec = env_spec
        self.initial_policy = initial_policy
        self.num_iterations = num_iterations
        self.jax_device = jax_device

    def eval(self, logger=None):
        policy = self.initial_policy
        mean_field = mean_field_by_transition_kernel_multi_jax(
            policy,
            self.env_spec,
            num_iterations=20,
            initial_mean_field=self.env_spec.environment.stationary_mean_field,
        )
        exploitabilities = [
            float(exploitability_jax(policy, self.env_spec, initial_mean_field=mean_field))
        ]
        return np.asarray(policy), np.asarray(mean_field), exploitabilities
```

Register the algorithm:

1. Add a dataclass in `src/benchmfg/conf/algorithm/algorithm_schema.py`.
2. Add a config field in `AlgorithmConfig`.
3. Add `src/benchmfg/config/algorithm/my_algo.yaml`.
4. Add a creation branch in `src/benchmfg/utility/create_solver.py`.
5. Add the algorithm name to plotting metadata only if it produces comparable output directories.

Minimal YAML:

```yaml
# @package _global_
algorithm:
  _target_: MyAlgo
  my_algo:
    num_iterations: 100
```

Smoke test:

```bash
benchmfg train algorithm=my_algo environment=lasry_lions_chain device=cpu
```

## Add An RL Best Response

BenchMFG keeps MFG environments stateless. For RL, wrap a fixed mean field into
the MDP seen by a representative agent:

```python
from benchmfg.rl import DQNBestResponse, FixedMeanFieldEnv, PPOBestResponse

rl_env = FixedMeanFieldEnv(environment, mean_field)
policy = PPOBestResponse(environment, total_timesteps=10_000).solve(mean_field)
policy = DQNBestResponse(environment, total_timesteps=10_000).solve(mean_field)
```

Use `algorithm.dampedfp.best_response=ppo` or
`algorithm.dampedfp.best_response=dqn` to replace exact dynamic-programming best
responses inside DampedFP. Install `bench-mfg-suite[rl]` for Gymnasium and
Stable-Baselines3 support.

## Checklist

- Run `benchmfg env list` or `benchmfg algo list` and confirm the new YAML appears.
- Run a tiny CPU experiment before testing CUDA.
- Add focused tests for transition/reward validity or solver output shape.
- Run `uv run ruff check src/benchmfg tests main.py`.
- Run `uv run pytest tests/`.
