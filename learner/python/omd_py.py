from dataclasses import dataclass
from typing import Tuple

import numpy as np
from tqdm import tqdm

from envs.mfg_model_class import MFGStationary
from utility.policy_average import softmax_policy


@dataclass
class OMDComponents:
    q_values: np.ndarray
    regularized_q_values: np.ndarray
    policy: np.ndarray
    mean_field: np.ndarray


class OMD_python:
    def __init__(
        self,
        model: MFGStationary,
        initial_policy: np.ndarray,
        learning_rate: float,
        num_iterations: int,
        early_stopping_enabled: bool = False,
        temperature: float = 0.1,
    ) -> None:
        self.horizon, self.N_states, self.N_actions = (
            model.horizon,
            model.N_states,
            model.N_actions,
        )
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.model = model
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
        initial_mean_field = self.model.mean_field_by_transition_kernel(
            initial_policy, num_transition_steps=20
        )
        self.model.update_stationary_mean_field(initial_mean_field)
        initial_q_values = np.zeros((self.N_states, self.N_actions))
        omd_components = OMDComponents(
            q_values=initial_q_values,
            regularized_q_values=initial_q_values,
            policy=initial_policy,
            mean_field=initial_mean_field,
        )
        exploitability = self.model.exploitability(initial_policy)
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
            omd_components.q_values = self.model.Q_eval(
                omd_components.policy, omd_components.mean_field
            )

            omd_components.regularized_q_values = (
                omd_components.regularized_q_values
                + self.learning_rate * omd_components.q_values
            )
            omd_components.policy = softmax_policy(
                omd_components.regularized_q_values, self.temperature
            )

            omd_components.mean_field = self.model.mean_field_by_transition_kernel(
                omd_components.policy, num_transition_steps=20
            )
            self.model.update_stationary_mean_field(omd_components.mean_field)

            exploitability = self.model.exploitability(omd_components.policy)
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
