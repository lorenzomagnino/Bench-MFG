from dataclasses import dataclass
from typing import Tuple

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from envs.mfg_model_class_jit import (
    EnvSpec,
    Q_eval_jax,
    exploitability_jax,
    mean_field_by_transition_kernel_multi_jax,
)
from utility.policy_average import softmax_policy


@dataclass
class OMDComponents:
    q_values: np.ndarray
    regularized_q_values: np.ndarray
    policy: np.ndarray
    mean_field: np.ndarray


class OMD_jax:
    def __init__(
        self,
        env_spec: EnvSpec,
        initial_policy: np.ndarray,
        learning_rate: float,
        num_iterations: int,
        early_stopping_enabled: bool = False,
        temperature: float = 0.1,
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

    def initialize(self) -> Tuple[OMDComponents, list]:
        """
        Initialize Q-table components and other variables for OMD algorithm.

        Returns:
            OMDComponents: Initial Q-values, regularized Q-values, policy, and mean field
            list: Initial exploitability values
        """
        initial_policy = self.initial_policy
        current_stationary_mf = jnp.asarray(
            self.env_spec.environment.stationary_mean_field
        )
        initial_mean_field = mean_field_by_transition_kernel_multi_jax(
            jnp.asarray(initial_policy),
            self.env_spec,
            num_iterations=20,
            initial_mean_field=current_stationary_mf,
        )
        initial_mean_field = np.asarray(initial_mean_field)
        self.env_spec.environment.stationary_mean_field = initial_mean_field.copy()
        initial_q_values = np.zeros((self.N_states, self.N_actions))
        omd_components = OMDComponents(
            q_values=initial_q_values,
            regularized_q_values=initial_q_values,
            policy=initial_policy,
            mean_field=initial_mean_field,
        )
        exploitability = float(
            exploitability_jax(
                jnp.asarray(initial_policy),
                self.env_spec,
                initial_mean_field=jnp.asarray(initial_mean_field),
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

        print(f"Initial Exploitability: {exploitabilities[0]}")
        if logger is not None:
            logger.log_iteration(0, exploitabilities[0], omd_components.mean_field)

        for iteration in tqdm(
            range(1, self.num_iterations + 1),
            desc=f"Running OMD (temperature: {self.temperature})",
        ):
            omd_components.q_values = np.asarray(
                Q_eval_jax(
                    jnp.asarray(omd_components.policy),
                    jnp.asarray(omd_components.mean_field),
                    self.env_spec,
                )
            )
            omd_components.regularized_q_values = (
                omd_components.regularized_q_values
                + self.learning_rate * omd_components.q_values
            )
            omd_components.policy = softmax_policy(
                omd_components.regularized_q_values, self.temperature
            )
            current_stationary_mf = jnp.asarray(
                self.env_spec.environment.stationary_mean_field
            )
            omd_components.mean_field = mean_field_by_transition_kernel_multi_jax(
                jnp.asarray(omd_components.policy),
                self.env_spec,
                num_iterations=20,
                initial_mean_field=current_stationary_mf,
            )
            omd_components.mean_field = np.asarray(omd_components.mean_field)
            self.env_spec.environment.stationary_mean_field = (
                omd_components.mean_field.copy()
            )

            exploitability = float(
                exploitability_jax(
                    jnp.asarray(omd_components.policy),
                    self.env_spec,
                    initial_mean_field=jnp.asarray(omd_components.mean_field),
                )
            )
            exploitabilities.append(exploitability)

            if logger is not None:
                logger.log_iteration(
                    iteration, exploitability, omd_components.mean_field
                )

        print(f"FINAL OMD EXPLOITABILITY: {exploitabilities[-1]}")

        return (
            omd_components.policy,
            omd_components.mean_field,
            exploitabilities,
        )
