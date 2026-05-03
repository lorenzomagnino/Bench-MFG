from dataclasses import dataclass
from functools import partial
import logging

from envs.mfg_model_class_jit import (
    EnvSpec,
    exploitability_batch_pmap,
    mean_field_by_transition_kernel_multi_jax,
)
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


@dataclass
class PSOCudaState:
    positions: jax.Array
    velocities: jax.Array
    best_positions: jax.Array
    swarm_best_position: jax.Array
    best_values_by_particles: jax.Array
    swarm_best_value: jax.Array


@partial(jax.jit, static_argnames=("num_particles", "dim"))
def _swarm_step(
    positions: jax.Array,
    velocities: jax.Array,
    best_positions: jax.Array,
    swarm_best_position: jax.Array,
    key: jax.Array,
    w: float,
    c1: float,
    c2: float,
    num_particles: int,
    dim: int,
) -> tuple[jax.Array, jax.Array]:
    key_r1, key_r2 = jax.random.split(key)
    r1 = jax.random.uniform(key_r1, (num_particles, dim))
    r2 = jax.random.uniform(key_r2, (num_particles, dim))
    velocities = (
        w * velocities
        + c1 * r1 * (best_positions - positions)
        + c2 * r2 * (swarm_best_position - positions)
    )
    positions = positions + velocities
    return positions, velocities


@jax.jit
def _update_state(
    positions: jax.Array,
    velocities: jax.Array,
    best_positions: jax.Array,
    swarm_best_position: jax.Array,
    swarm_best_value: jax.Array,
    current_values: jax.Array,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    improved_mask = current_values < swarm_best_value
    best_positions = jnp.where(
        improved_mask[:, None],
        positions,
        best_positions,
    )
    iteration_best_value = jnp.min(current_values)
    best_idx = jnp.argmin(current_values)
    better_global = iteration_best_value < swarm_best_value
    swarm_best_position = jnp.where(
        better_global,
        best_positions[best_idx],
        swarm_best_position,
    )
    swarm_best_value = jnp.where(
        better_global,
        iteration_best_value,
        swarm_best_value,
    )
    return (
        positions,
        velocities,
        best_positions,
        swarm_best_position,
        current_values,
        swarm_best_value,
    )


def _policy_to_logits(policy: np.ndarray, temperature: float) -> np.ndarray:
    epsilon = 1e-12
    safe_policy = np.clip(policy, epsilon, 1.0)
    return np.log(safe_policy) * temperature


def _mellowmax_jax(logits: jax.Array, omega: float = 16.55) -> jax.Array:
    n_actions = logits.shape[-1]
    c = jnp.max(logits, axis=-1, keepdims=True)
    exp_logits = jnp.exp(omega * (logits - c))
    log_sum_exp = jnp.log(jnp.sum(exp_logits, axis=-1, keepdims=True) / n_actions)
    mellowmax_vals = c + (log_sum_exp / omega)
    return jnp.exp(omega * (logits - mellowmax_vals))


def _policy_from_logits_batch(
    logits: jax.Array, temperature: float, policy_type: str | None
) -> jax.Array:
    if policy_type == "mellowmax":
        return _mellowmax_jax(logits)
    return jax.nn.softmax(logits / temperature, axis=-1)


def _policy_from_logits_single(
    logits: jax.Array, temperature: float, policy_type: str | None
) -> jax.Array:
    if policy_type == "mellowmax":
        return _mellowmax_jax(logits)
    return jax.nn.softmax(logits / temperature, axis=-1)


class PSO_jax_cuda:
    def __init__(
        self,
        env_spec: EnvSpec,
        num_particles: int,
        num_iterations: int,
        w: float,
        c1: float,
        c2: float,
        temperature: float,
        init_solution=None,
        initialization_type: str | None = None,
        policy_type: str | None = None,
        shuffle: str | None = None,
        init_policy_temp: float | None = None,
        random_seed: int = 0,
        jax_device=None,
    ) -> None:
        self.env_spec = env_spec
        self.dim = env_spec.environment.N_states * env_spec.environment.N_actions
        self.num_particles = num_particles
        self.num_iterations = num_iterations
        self.w_arr = np.zeros(num_iterations) + w
        self.c1 = c1
        self.c2 = c2
        self.init_solution = init_solution
        self.temperature = temperature
        self.init_policy_temp = (
            init_policy_temp if init_policy_temp is not None else temperature
        )
        self.positions_evolution: list[np.ndarray] = []
        self.initialization_type = initialization_type
        self.shuffle_type = shuffle
        self.policy_type = policy_type
        self.random_seed = random_seed
        self.jax_device = (
            jax_device if jax_device is not None else jax.devices("cpu")[0]
        )
        logging.info(
            "Initialized CUDA PSO with %s iterations, %s particles, total_dim: %s",
            self.num_iterations,
            self.num_particles,
            self.dim,
        )

    def _put(self, arr):
        return jax.device_put(arr, self.jax_device)

    def monitor_evolution(self, positions: jax.Array) -> None:
        reshaped_positions = np.asarray(positions).reshape(
            self.num_particles,
            self.env_spec.environment.N_states,
            self.env_spec.environment.N_actions,
        )
        self.positions_evolution.append(reshaped_positions.copy())

    def _batch_exploitability(
        self,
        positions: jax.Array,
        current_stationary_mf: jax.Array,
    ) -> jax.Array:
        logits_batch = positions.reshape(
            self.num_particles,
            self.env_spec.environment.N_states,
            self.env_spec.environment.N_actions,
        )
        policies_batch = _policy_from_logits_batch(
            logits_batch,
            self.temperature,
            self.policy_type,
        )
        return exploitability_batch_pmap(
            policies_batch,
            self.env_spec,
            current_stationary_mf,
            self.num_particles,
        )

    def initialize_pso_components(self, predefined_ratio=0.05) -> PSOCudaState:
        if self.init_solution is not None:
            logging.debug("Using predefined solution as initialization")
            num_particles_to_initialize_with_initial_policy = int(
                self.num_particles * predefined_ratio
            )
            logits = _policy_to_logits(
                self.init_solution, self.init_policy_temp
            ).flatten()
            positions = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))
            for i in range(num_particles_to_initialize_with_initial_policy):
                positions[i] = logits
            velocities = np.zeros((self.num_particles, self.dim))
        else:
            if self.initialization_type == "one_uniform":
                positions = np.random.randn(self.num_particles, self.dim) * 0.5
                positions[0] = np.zeros(self.dim)
            elif self.initialization_type == "dirichlet":
                logging.info("Using dirichlet initialization")
                policies = np.random.dirichlet(
                    np.ones(self.env_spec.environment.N_actions),
                    size=(self.num_particles, self.env_spec.environment.N_states),
                )
                positions = np.log(policies).reshape((self.num_particles, self.dim))
            else:
                logging.info("Using PSO uniform initialization")
                positions = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))

            velocities = np.random.randn(self.num_particles, self.dim) * 0.1

        positions_jax = self._put(positions)
        return PSOCudaState(
            positions=positions_jax,
            velocities=self._put(velocities),
            best_positions=positions_jax,
            swarm_best_position=self._put(np.zeros(self.dim)),
            best_values_by_particles=self._put(np.zeros(self.num_particles)),
            swarm_best_value=self._put(np.array(np.inf, dtype=np.float32)),
        )

    def eval(self, verbose=False, logger=None) -> tuple:
        progress_arr = np.zeros(self.num_iterations)
        key = jax.random.PRNGKey(self.random_seed)
        current_stationary_mf = self._put(
            self.env_spec.environment.stationary_mean_field
        )

        state = self.initialize_pso_components()
        self.monitor_evolution(state.positions)
        current_values = self._batch_exploitability(
            state.positions,
            current_stationary_mf,
        )
        best_idx = int(np.asarray(jnp.argmin(current_values)))
        state = PSOCudaState(
            positions=state.positions,
            velocities=state.velocities,
            best_positions=state.best_positions,
            swarm_best_position=state.best_positions[best_idx],
            best_values_by_particles=current_values,
            swarm_best_value=jnp.min(current_values),
        )

        best_policy = _policy_from_logits_single(
            state.swarm_best_position.reshape(
                self.env_spec.environment.N_states,
                self.env_spec.environment.N_actions,
            ),
            self.temperature,
            self.policy_type,
        )
        best_mean_field = mean_field_by_transition_kernel_multi_jax(
            best_policy,
            self.env_spec,
            50,
            initial_mean_field=current_stationary_mf,
        )
        current_stationary_mf = best_mean_field

        for i in tqdm(range(0, self.num_iterations), desc="Optimization loop"):
            key, subkey = jax.random.split(key)
            positions, velocities = _swarm_step(
                state.positions,
                state.velocities,
                state.best_positions,
                state.swarm_best_position,
                subkey,
                self.w_arr[i],
                self.c1,
                self.c2,
                self.num_particles,
                self.dim,
            )
            state.positions = positions
            state.velocities = velocities
            self.monitor_evolution(state.positions)
            current_values = self._batch_exploitability(
                state.positions,
                current_stationary_mf,
            )
            (
                state.positions,
                state.velocities,
                state.best_positions,
                state.swarm_best_position,
                state.best_values_by_particles,
                state.swarm_best_value,
            ) = _update_state(
                state.positions,
                state.velocities,
                state.best_positions,
                state.swarm_best_position,
                state.swarm_best_value,
                current_values,
            )
            swarm_best_value = float(np.asarray(state.swarm_best_value))
            if verbose:
                log.info(
                    "Iteration %d out of %d, best (min) exploitability: %s.",
                    i + 1,
                    self.num_iterations,
                    swarm_best_value,
                )
            progress_arr[i] = swarm_best_value
            if i % 10 == 0 or i == 1:
                logging.debug(
                    "Exploitability at iteration %d/%d : %s",
                    i + 1,
                    self.num_iterations,
                    swarm_best_value,
                )

            best_policy = _policy_from_logits_single(
                state.swarm_best_position.reshape(
                    self.env_spec.environment.N_states,
                    self.env_spec.environment.N_actions,
                ),
                self.temperature,
                self.policy_type,
            )
            best_mean_field = mean_field_by_transition_kernel_multi_jax(
                best_policy,
                self.env_spec,
                50,
                initial_mean_field=current_stationary_mf,
            )
            current_stationary_mf = best_mean_field

            if logger is not None:
                logger.log_iteration(i, swarm_best_value, np.asarray(best_mean_field))

        log.info("Exploitability with PSO algorithm: %s", state.swarm_best_value)
        self.env_spec.environment.stationary_mean_field = np.asarray(
            current_stationary_mf
        )
        return (
            np.asarray(best_policy),
            np.asarray(current_stationary_mf),
            progress_arr,
        )
