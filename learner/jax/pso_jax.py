from dataclasses import dataclass
import logging
from typing import Optional, Tuple

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm

from envs.mfg_model_class_jit import (
    EnvSpec,
    exploitability_batch_jax,
    exploitability_jax,
    mean_field_by_transition_kernel_multi_jax,
)


@dataclass
class PSOEvalComponents:
    positions: np.ndarray
    velocities: np.ndarray
    best_positions: np.ndarray
    swarm_best_position: np.ndarray
    best_values_by_particles: np.ndarray
    swarm_best_value: np.ndarray


class PSO_jax:
    """
    This class implements the Particle Swarm Optimization algorithm for the MFG model.

    Attributes:
    - env_spec: environment specification
    - dim: total dimension of the search space
    - num_particles: swarm size
    - num_iterations: number of iterations
    - w_arr: np.ndarray
    - c1: individual (cognitive) component weight
    - c2: collective (social) component weight
    - init_solution: initial policy
    - temperature: temperature for the Boltzmann distribution
    - initialization_type: how to initialize the particles
    - policy_type: type of policy
    - shuffle: how to shuffle the particles
    - shuffle_type: type of shuffle
    - shuffle_probability: probability of shuffling the particles
    - shuffle_size: size of the shuffle
    - stall_max_iterations: maximum number of iterations without improvement
    - stall_threshold: threshold for the improvement
    """

    def __init__(
        self,
        env_spec: EnvSpec,
        num_particles: int,
        num_iterations: int,
        w: float,
        c1: float,
        c2: float,
        temperature: float,
        init_solution=None,  # NOTE: if the user provides an initial policy then
        initialization_type: Optional[str] = None,
        policy_type: Optional[str] = None,
        shuffle: Optional[str] = None,
        init_policy_temp: Optional[
            float
        ] = None,  # temperature for initial policy conversion (if init_solution provided)
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
        self.positions_evolution = []  # To store logits evolution over iterations
        self.initialization_type = initialization_type
        self.shuffle_type = shuffle
        self.policy_type = policy_type  # mellowmax or softmax
        logging.info(
            f"Initialized PSO with {self.num_iterations} iterations, \n{self.num_particles} particles, \ntotal_dim: {self.dim}"
        )

    def monitor_evolution(self, positions):
        """
        Monitors the positions (logits) of all particles for all timesteps and states.

        Args:
        - positions (np.ndarray): Current positions of the particles.
        """

        positions = positions.reshape(
            self.num_particles,
            self.env_spec.environment.N_states,
            self.env_spec.environment.N_actions,
        )
        reshaped_positions = np.copy(positions)
        self.positions_evolution.append(reshaped_positions)

    def _batch_exploitability(self, positions: np.ndarray) -> np.ndarray:
        """
        Compute exploitability for all particles in a vectorized manner using JAX vmap.

        Args:
        - positions: Array of shape (num_particles, dim) containing logits for all particles

        Returns:
        - Array of shape (num_particles,) containing exploitability for each particle
        """
        logits_batch = positions.reshape(
            self.num_particles,
            self.env_spec.environment.N_states,
            self.env_spec.environment.N_actions,
        )

        if self.policy_type == "mellowmax":
            policies_batch = np.array(
                [
                    mellowmax(logits, self.env_spec.environment.N_actions)
                    for logits in logits_batch
                ]
            )
        else:
            logits_max = np.max(logits_batch, axis=-1, keepdims=True)
            logits_stable = logits_batch - logits_max
            logits_scaled = logits_stable / self.temperature
            exp_logits = np.exp(logits_scaled)
            policies_batch = exp_logits / np.sum(exp_logits, axis=-1, keepdims=True)

        initial_mf = jnp.asarray(self.env_spec.environment.stationary_mean_field)

        exploitabilities = exploitability_batch_jax(
            jnp.asarray(policies_batch),
            self.env_spec,
            initial_mf,
            self.num_particles,
        )

        return np.array(exploitabilities, copy=True)

    def initialize_pso_components(self, predefined_ratio=0.05) -> PSOEvalComponents:
        """
        Gives the initial position and velocity of each particles.
        """
        if self.init_solution is not None:
            logging.debug("Using predefined solution as initialization")

            num_particles_to_initialize_with_initial_policy = int(
                self.num_particles * predefined_ratio
            )
            logits = policy_to_logits(self.init_solution, self.init_policy_temp)
            logits = logits.flatten()
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
                positions = np.log(policies)
                positions = positions.reshape((self.num_particles, self.dim))
            elif self.initialization_type == "PSO_uniform":
                logging.info("Using PSO uniform initialization")
                positions = np.random.uniform(-1.0, 1.0, (self.num_particles, self.dim))

            velocities = np.random.randn(self.num_particles, self.dim) * 0.1
            self.initial_positions = positions.reshape(
                self.num_particles,
                self.env_spec.environment.N_states,
                self.env_spec.environment.N_actions,
            )

        pso_components = PSOEvalComponents(
            positions=positions,
            velocities=velocities,
            best_positions=np.copy(positions),
            swarm_best_position=np.zeros(self.dim),
            best_values_by_particles=np.zeros(self.num_particles),
            swarm_best_value=np.array([np.inf], dtype=np.float32)[0],
        )
        return pso_components

    def dynamic(
        self,
        t,
        pso_components: PSOEvalComponents,
    ) -> PSOEvalComponents:
        """
        Update the velocity and the position of each particle
        """
        r1 = np.random.uniform(0, 1, (self.num_particles, self.dim))
        r2 = np.random.uniform(0, 1, (self.num_particles, self.dim))

        velocities = (
            self.w_arr[t] * pso_components.velocities
            + self.c1 * r1 * (pso_components.best_positions - pso_components.positions)
            + self.c2
            * r2
            * (pso_components.swarm_best_position - pso_components.positions)
        )
        pso_components.positions += velocities

        return pso_components

    def eval(
        self,
        verbose=False,
        logger=None,
    ) -> Tuple:
        """
        Running PSO.
        Return:  best_policy, MF(best_policy), [all the exploitabiities at each iteration]
        """
        progress_arr = np.zeros(self.num_iterations)

        pso_components = self.initialize_pso_components()
        self.monitor_evolution(pso_components.positions)
        pso_components.best_values_by_particles = self._batch_exploitability(
            pso_components.positions
        )
        pso_components.swarm_best_position = pso_components.best_positions[
            np.argmin(pso_components.best_values_by_particles)
        ]
        pso_components.swarm_best_value = np.min(
            pso_components.best_values_by_particles
        )
        for i in tqdm(range(0, self.num_iterations), desc="Optimization loop"):
            pso_components = self.dynamic(
                i,
                pso_components,
            )
            self.monitor_evolution(pso_components.positions)
            pso_components.best_values_by_particles = self._batch_exploitability(
                pso_components.positions
            )
            pso_components = update(pso_components)
            print_to_console(
                verbose,
                i,
                pso_components.swarm_best_value,
                self.num_iterations,
            )
            progress_arr[i] = pso_components.swarm_best_value
            if i % 10 == 0 or i == 1:
                logging.debug(
                    f"Exploitability at iteration {i + 1}/{self.num_iterations} : {pso_components.swarm_best_value}"
                )

            swarm_best_position = pso_components.swarm_best_position.reshape(
                (
                    self.env_spec.environment.N_states,
                    self.env_spec.environment.N_actions,
                )
            )
            if self.policy_type == "mellowmax":
                best_policy = mellowmax(
                    swarm_best_position, self.env_spec.environment.N_actions
                )
            else:
                best_policy = boltzmann_policy(swarm_best_position, self.temperature)
            current_stationary_mf = jnp.asarray(
                self.env_spec.environment.stationary_mean_field
            )
            mean_field = mean_field_by_transition_kernel_multi_jax(
                best_policy, self.env_spec, 50, initial_mean_field=current_stationary_mf
            )
            self.env_spec.environment.stationary_mean_field = np.asarray(mean_field)

            if logger is not None:
                logger.log_iteration(i, pso_components.swarm_best_value, mean_field)

        print(f"Exploitability with PSO algorithm : {pso_components.swarm_best_value}")
        return (
            best_policy,
            mean_field,
            progress_arr,
        )


def exploitability_function(
    logits: np.ndarray,
    env_spec: EnvSpec,
    policy_type: Optional[str],
    temperature: float,
) -> float:
    """Compute the exploitability of a policy represented by logits."""
    logits = logits.reshape(
        (env_spec.environment.N_states, env_spec.environment.N_actions)
    )
    if policy_type == "mellowmax":
        policy = mellowmax(logits, env_spec.environment.N_actions)
    elif policy_type == "boltzmann":
        policy = boltzmann_policy(logits, temperature)
    else:
        raise ValueError(f"Invalid policy type: {policy_type}")
    return float(
        exploitability_jax(
            policy,
            env_spec,
            initial_mean_field=jnp.asarray(env_spec.environment.stationary_mean_field),
        )
    )


def shuffle_particles(
    pso_components: PSOEvalComponents,
    shuffle_size: float,
    shuffle_type: Optional[str],
    shuffle_probability: float,
) -> PSOEvalComponents:
    num_particles, dim = pso_components.positions.shape
    if np.random.rand() < shuffle_probability:
        print("Performing rebirth")
        distances = np.abs(
            pso_components.best_values_by_particles - pso_components.swarm_best_value
        )
        if shuffle_type == "worst":
            sorted_indices = np.argsort(distances)[::-1]
            num_to_replace = int(num_particles * shuffle_size)
            worst_indices = sorted_indices[:num_to_replace]
            pso_components.positions[worst_indices] = np.random.uniform(
                -1.0, 1.0, (num_to_replace, dim)
            )
            pso_components.best_positions[worst_indices] = pso_components.positions[
                worst_indices
            ]
            pso_components.velocities[worst_indices] = np.zeros((num_to_replace, dim))
        if shuffle_type == "best":
            best_particle_index = np.argmin(distances)
            remaining_indices = np.arange(num_particles) != best_particle_index
            sorted_indices = np.argsort(distances[remaining_indices])

            num_to_replace = int(num_particles * shuffle_size)
            top_indices = np.where(remaining_indices)[0][
                sorted_indices[:num_to_replace]
            ]
            pso_components.positions[top_indices] = np.random.uniform(
                -1.0, 1.0, (num_to_replace, dim)
            )
            pso_components.best_positions[top_indices] = pso_components.positions[
                top_indices
            ]
            pso_components.velocities[top_indices] = np.zeros((num_to_replace, dim))
    return pso_components


def update(
    pso_components: PSOEvalComponents,
) -> PSOEvalComponents:
    improved_indices = np.where(
        pso_components.best_values_by_particles < pso_components.swarm_best_value
    )
    pso_components.best_positions[improved_indices] = pso_components.positions[
        improved_indices
    ]
    pso_components.best_values_by_particles[
        improved_indices
    ] = pso_components.best_values_by_particles[improved_indices]
    iteration_best_fitness = np.min(pso_components.best_values_by_particles)
    if iteration_best_fitness < pso_components.swarm_best_value:
        pso_components.swarm_best_position = pso_components.best_positions[
            np.argmin(pso_components.best_values_by_particles)
        ]
        pso_components.swarm_best_value = np.min(
            pso_components.best_values_by_particles
        )
    return pso_components


def print_to_console(verbose_bool, i, swarm_best_fitness, num_iterations):
    if verbose_bool:
        output = "Iteration  {ci}  out of  {mi} , best (min) exploitability:  {sbf}."
        print(output.format(ci=i + 1, mi=num_iterations, sbf=swarm_best_fitness))


def policy_to_logits(policy: np.ndarray, temperature: float) -> np.ndarray:
    """
    Convert a policy (probabilities) to logits using the inverse Boltzmann transformation.

    Parameters:
    - policy: 3D numpy array of shape (N_states, N_actions).

    Returns:
    - logits: 3D numpy array of shape (N_states, N_actions).
    """

    epsilon = 1e-12
    safe_policy = np.clip(policy, epsilon, 1.0)
    logits = np.log(safe_policy) * temperature
    return logits


def mellowmax(logits: np.ndarray, N_actions: int, omega: float = 16.55) -> np.ndarray:
    """
    Compute the mellowmax transformation with numerical stabilization.

    Parameters:
    - logits (np.ndarray): Array of logits to be transformed.
    - omega (float): Mellowmax parameter controlling the smoothing effect.

    Returns:
    - np.ndarray: Transformed mellowmax values.
    """
    n = N_actions
    c = np.max(logits, axis=-1, keepdims=True)  # Shape: (N_states, 1)
    exp_logits = np.exp(omega * (logits - c))  # Shape: (N_states, N_actions)
    log_sum_exp = np.log(
        np.sum(exp_logits, axis=-1, keepdims=True) / n
    )  # Shape: (N_states, 1)
    mellowmax_vals = c + (log_sum_exp / omega)  # Shape: (N_states, 1)

    mellowmax_probs = np.exp(
        omega * (logits - mellowmax_vals)
    )  # Broadcast mellowmax_vals

    return mellowmax_probs


def boltzmann_policy(logits: np.ndarray, temperature: float) -> np.ndarray:
    """
    Given an array of logits it gives a probability distribution using Boltzmann normalization (softmax probabiltiy).
    """
    logits_max = np.max(logits, axis=-1, keepdims=True)
    logits_stable = logits - logits_max
    logits_scaled = logits_stable / temperature
    exp_logits = np.exp(logits_scaled)
    softmax_probs = exp_logits / np.sum(
        exp_logits, axis=-1, keepdims=True
    )  # (N_states, N_actions)
    return softmax_probs


def plot_logits_evolution(state, positions_evolution, N_actions):
    """
    Plots the evolution of logits over iterations for a specific and state.

    Args:
    - timestep (int): The timestep to monitor.
    - state (int): The state to monitor.
    """
    positions_evolution = np.array(positions_evolution)
    num_actions = N_actions
    num_iterations = len(positions_evolution)

    num_cols = 6
    num_rows = -(-num_iterations // num_cols)

    fig, axes = plt.subplots(
        num_rows, num_cols, figsize=(20, num_rows * 5), squeeze=False
    )

    for idx, iteration in enumerate(
        range(0, num_iterations, max(1, num_iterations // (num_rows * num_cols)))
    ):
        data = positions_evolution[iteration, :, state, :]
        row, col = divmod(idx, num_cols)

        ax = axes[row, col]
        for action in range(num_actions):
            values = data[:, action]
            ax.scatter(
                [action] * len(values),
                values,
                crowd_penalty_coefficient=0.5,
                label=f"Action {action - 1}" if iteration == 0 else None,
            )

        ax.set_title(f"Iteration {iteration}, state={state})")
        ax.set_xlabel("Actions")
        ax.set_ylabel("Logits")
        ax.set_xticks(range(num_actions))
        ax.grid(axis="x", linestyle="--")

    for idx in range(idx + 1, num_rows * num_cols):  # noqa: B020
        row, col = divmod(idx, num_cols)
        fig.delaxes(axes[row, col])

    plt.tight_layout()
    plt.show()
