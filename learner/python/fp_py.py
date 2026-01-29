from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
from tqdm import tqdm

from envs.mfg_model_class import MFGStationary

LambdaSchedule = Literal["damped", "pure", "fictitious_play"]


@dataclass
class FPState:
    policy: np.ndarray
    mean_field: np.ndarray


class DampedFP_python:
    def __init__(
        self,
        model: MFGStationary,
        initial_policy: np.ndarray,
        num_iterations: int,
        early_stopping_enabled: bool = False,
        lambda_schedule: LambdaSchedule = "damped",
        damped_constant: float = 0.2,
        num_transition_steps: int = 20,
    ) -> None:
        self.horizon, self.N_states, self.N_actions = (
            model.horizon,
            model.N_states,
            model.N_actions,
        )
        self.initial_policy = initial_policy
        self.num_iterations = num_iterations
        self.model = model
        self.early_stopping_enabled = early_stopping_enabled
        self.damped_constant = damped_constant
        self.lambda_schedule = lambda_schedule
        self.num_transition_steps = int(num_transition_steps)

    def _lambda_k(self, k: int) -> float:
        if self.lambda_schedule == "pure":
            return 1.0
        if self.lambda_schedule == "fictitious_play":
            return 1.0 / (k + 1.0)
        return float(self.damped_constant)

    def _average_policies_uniform(self, policies: list[np.ndarray]) -> np.ndarray:
        avg = np.mean(np.stack(policies, axis=0), axis=0)
        avg = np.clip(avg, 1e-12, 1.0)
        avg = avg / avg.sum(axis=1, keepdims=True)
        return avg

    def _average_policies_weighted(
        self, policies: list[np.ndarray], mean_fields: list[np.ndarray]
    ) -> np.ndarray:
        """
        Compute weighted average policy according to the formula:
        π̄ⁿ_j(a|x) = (Σ_{i=0}^j μⁿ_{π^i}(x) * πⁿ_i(a|x)) / (Σ_{i=0}^j μⁿ_{π^i}(x))

        Parameters:
        policies: List of policies, each with shape (N_states, N_actions)
        mean_fields: List of mean fields induced by each policy, each with shape (N_states,)

        Returns:
        Weighted average policy with shape (N_states, N_actions)
        """
        numerator = np.zeros((self.N_states, self.N_actions))
        denominator = np.zeros(self.N_states)
        for policy, mean_field in zip(policies, mean_fields):
            numerator += mean_field[:, None] * policy
            denominator += mean_field
        denominator = np.clip(denominator, 1e-12, None)
        avg_policy = numerator / denominator[:, None]
        avg_policy = np.clip(avg_policy, 1e-12, 1.0)
        avg_policy = avg_policy / avg_policy.sum(axis=1, keepdims=True)
        return avg_policy

    def initialize(self) -> Tuple[FPState, list]:
        mu = self.model.mean_field_by_transition_kernel(
            self.initial_policy, num_transition_steps=self.num_transition_steps
        )
        mu = mu / mu.sum()
        self.model.update_stationary_mean_field(mu)

        exploitability = self.model.exploitability(self.initial_policy)
        return FPState(policy=self.initial_policy.copy(), mean_field=mu), [
            exploitability
        ]

    def eval(self, logger=None):
        state, exploitabilities = self.initialize()
        policy_history: list[np.ndarray] = [state.policy.copy()]
        mean_field_history: list[np.ndarray] = [state.mean_field.copy()]
        print(f"Initial Exploitability: {exploitabilities[0]}")
        print(f"Lambda Schedule: {self.lambda_schedule}")
        if logger is not None:
            logger.log_iteration(0, exploitabilities[0], state.mean_field)
        for k in tqdm(range(1, self.num_iterations + 1), desc="Running"):
            _, policy_best_response = self.model.Vpi_opt(state.mean_field)
            mean_field_br = self.model.mean_field_by_transition_kernel(
                policy_best_response, num_transition_steps=self.num_transition_steps
            )
            mean_field_br = mean_field_br / mean_field_br.sum()

            if self.lambda_schedule == "fictitious_play":
                policy_history.append(policy_best_response.copy())
                mean_field_history.append(mean_field_br.copy())

            alpha_k = self._lambda_k(k)
            average_mean_field = (
                1.0 - alpha_k
            ) * state.mean_field + alpha_k * mean_field_br
            state.mean_field = average_mean_field / average_mean_field.sum()
            self.model.update_stationary_mean_field(state.mean_field.copy())

            state.policy = policy_best_response  # NOTE: lack of consistency? state = (policy_best_response, average_mean_field)
            if self.lambda_schedule == "fictitious_play" and len(policy_history) > 0:
                # average_policy = self._average_policies_uniform(policy_history)
                average_policy = self._average_policies_weighted(
                    policy_history, mean_field_history
                )
                exploitability = self.model.exploitability(average_policy)
            else:
                exploitability = self.model.exploitability(state.policy)
            exploitabilities.append(exploitability)
            if logger is not None:
                logger.log_iteration(k, exploitability, state.mean_field)

        final_policy = (
            average_policy
            if self.lambda_schedule == "fictitious_play"
            else state.policy
        )
        final_exploitability = self.model.exploitability(final_policy)
        print(f"Exploitability (returned policy): {final_exploitability}")
        return final_policy, state.mean_field, exploitabilities
