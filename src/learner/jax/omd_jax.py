from dataclasses import dataclass
import logging

from envs.mfg_model_class_jit import (
    EnvSpec,
    Q_eval_jax,
    exploitability_jax,
    mean_field_by_transition_kernel_multi_jax,
)
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm
from utility.policy_average import softmax_policy_jax

log = logging.getLogger(__name__)


@dataclass
class OMDComponents:
    q_values: jax.Array
    regularized_q_values: jax.Array
    policy: jax.Array
    mean_field: jax.Array


class OMD_jax:
    def __init__(
        self,
        env_spec: EnvSpec,
        initial_policy: np.ndarray,
        learning_rate: float,
        num_iterations: int,
        early_stopping_enabled: bool = False,
        temperature: float = 0.1,
        jax_device=None,
    ) -> None:
        self.horizon, self.N_states, self.N_actions = (
            env_spec.environment.horizon,
            env_spec.environment.N_states,
            env_spec.environment.N_actions,
        )
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.env_spec = env_spec
        self.initial_policy = initial_policy
        self.early_stopping_enabled = early_stopping_enabled
        self.temperature = temperature
        self.jax_device = (
            jax_device if jax_device is not None else jax.devices("cpu")[0]
        )

    def _put(self, arr):
        """Place a numpy/JAX array on the configured JAX device."""
        return jax.device_put(arr, self.jax_device)

    def initialize(self) -> tuple[OMDComponents, list]:
        """
        Initialize Q-table components and other variables for OMD algorithm.

        Returns:
            OMDComponents: Initial Q-values, regularized Q-values, policy, and mean field
            list: Initial exploitability values
        """
        initial_policy = self._put(self.initial_policy)
        current_stationary_mf = self._put(
            self.env_spec.environment.stationary_mean_field
        )
        initial_mean_field = mean_field_by_transition_kernel_multi_jax(
            initial_policy,
            self.env_spec,
            num_iterations=20,
            initial_mean_field=current_stationary_mf,
        )
        initial_q_values = jnp.zeros(
            (self.N_states, self.N_actions),
            dtype=jnp.float32,
        )
        omd_components = OMDComponents(
            q_values=initial_q_values,
            regularized_q_values=initial_q_values,
            policy=initial_policy,
            mean_field=initial_mean_field,
        )
        exploitability = float(
            exploitability_jax(
                initial_policy,
                self.env_spec,
                initial_mean_field=initial_mean_field,
            )
        )
        exploitabilities = [exploitability]

        return omd_components, exploitabilities

    def eval(self, logger=None):
        """
        Main OMD algorithm evaluation loop.

        Args:
            logger: Optional wandb logger for tracking metrics

        Returns:
            Tuple containing final policy, mean field, and exploitability history
        """
        omd_components, exploitabilities = self.initialize()

        log.info("Initial Exploitability: %s", exploitabilities[0])
        if logger is not None:
            logger.log_iteration(
                0, exploitabilities[0], np.asarray(omd_components.mean_field)
            )

        for iteration in tqdm(
            range(1, self.num_iterations + 1),
            desc=f"Running OMD (temperature: {self.temperature})",
        ):
            omd_components.q_values = Q_eval_jax(
                omd_components.policy,
                omd_components.mean_field,
                self.env_spec,
            )
            omd_components.regularized_q_values = (
                omd_components.regularized_q_values
                + self.learning_rate * omd_components.q_values
            )
            omd_components.policy = softmax_policy_jax(
                omd_components.regularized_q_values, self.temperature
            )
            omd_components.mean_field = mean_field_by_transition_kernel_multi_jax(
                omd_components.policy,
                self.env_spec,
                num_iterations=20,
                initial_mean_field=omd_components.mean_field,
            )

            exploitability = float(
                exploitability_jax(
                    omd_components.policy,
                    self.env_spec,
                    initial_mean_field=omd_components.mean_field,
                )
            )
            exploitabilities.append(exploitability)

            if logger is not None:
                logger.log_iteration(
                    iteration, exploitability, np.asarray(omd_components.mean_field)
                )

        log.info("Final OMD exploitability: %s", exploitabilities[-1])
        self.env_spec.environment.stationary_mean_field = np.asarray(
            omd_components.mean_field
        )

        return (
            np.asarray(omd_components.policy),
            np.asarray(omd_components.mean_field),
            exploitabilities,
        )
