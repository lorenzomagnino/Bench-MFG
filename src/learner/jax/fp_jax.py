from dataclasses import dataclass
import logging
from typing import Literal

from envs.mfg_model_class_jit import (
    EnvSpec,
    Vpi_opt_jax,
    exploitability_jax,
    mean_field_by_transition_kernel_multi_jax,
)
import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)

LambdaSchedule = Literal["damped", "pure", "fictitious_play"]


@dataclass
class FPState:
    policy: jax.Array
    mean_field: jax.Array


class DampedFP_jax:
    def __init__(
        self,
        env_spec: EnvSpec,
        initial_policy: np.ndarray,
        num_iterations: int,
        early_stopping_enabled: bool = False,
        lambda_schedule: LambdaSchedule = "damped",
        damped_constant: float = 0.2,
        num_transition_steps: int = 20,
        jax_device=None,
    ) -> None:
        self.horizon, self.N_states, self.N_actions = (
            env_spec.environment.horizon,
            env_spec.environment.N_states,
            env_spec.environment.N_actions,
        )
        self.initial_policy = initial_policy
        self.num_iterations = num_iterations
        self.env_spec = env_spec
        self.early_stopping_enabled = early_stopping_enabled
        self.damped_constant = damped_constant
        self.lambda_schedule = lambda_schedule
        self.num_transition_steps = int(num_transition_steps)
        self.jax_device = (
            jax_device if jax_device is not None else jax.devices("cpu")[0]
        )

    def _put(self, arr):
        """Place a numpy/JAX array on the configured JAX device."""
        return jax.device_put(arr, self.jax_device)

    def _lambda_k(self, k: int) -> float:
        if self.lambda_schedule == "pure":
            return 1.0
        if self.lambda_schedule == "fictitious_play":
            return 1.0 / (k + 1.0)
        return float(self.damped_constant)

    def _average_policies_uniform(self, policies: list[jax.Array]) -> jax.Array:
        avg = jnp.mean(jnp.stack(policies, axis=0), axis=0)
        avg = jnp.clip(avg, 1e-12, 1.0)
        avg = avg / avg.sum(axis=1, keepdims=True)
        return avg

    def _average_policies_weighted(
        self, policies: list[jax.Array], mean_fields: list[jax.Array]
    ) -> jax.Array:
        """
        Compute weighted average policy according to the formula:
        π̄ⁿ_j(a|x) = (Σ_{i=0}^j μⁿ_{π^i}(x) * πⁿ_i(a|x)) / (Σ_{i=0}^j μⁿ_{π^i}(x))

        Parameters:
        policies: List of policies, each with shape (N_states, N_actions)
        mean_fields: List of mean fields induced by each policy, each with shape (N_states,)

        Returns:
        Weighted average policy with shape (N_states, N_actions)
        """
        policy_stack = jnp.stack(policies, axis=0)
        mean_field_stack = jnp.stack(mean_fields, axis=0)
        numerator = jnp.sum(mean_field_stack[:, :, None] * policy_stack, axis=0)
        denominator = jnp.clip(jnp.sum(mean_field_stack, axis=0), 1e-12)
        avg_policy = numerator / denominator[:, None]
        avg_policy = jnp.clip(avg_policy, 1e-12, 1.0)
        avg_policy = avg_policy / avg_policy.sum(axis=1, keepdims=True)

        return avg_policy

    def initialize(self) -> tuple[FPState, list]:
        initial_policy = self._put(self.initial_policy)
        current_stationary_mf = self._put(
            self.env_spec.environment.stationary_mean_field
        )
        mu = mean_field_by_transition_kernel_multi_jax(
            initial_policy,
            self.env_spec,
            num_iterations=self.num_transition_steps,
            initial_mean_field=current_stationary_mf,
        )
        mu = mu / mu.sum()

        exploitability = float(
            exploitability_jax(
                initial_policy,
                self.env_spec,
                initial_mean_field=mu,
            )
        )
        return FPState(policy=initial_policy, mean_field=mu), [exploitability]

    def eval(self, logger=None):
        state, exploitabilities = self.initialize()
        policy_history: list[jax.Array] = []
        mean_field_history: list[jax.Array] = []
        log.info("Initial Exploitability: %s", exploitabilities[0])
        log.info("Lambda Schedule: %s", self.lambda_schedule)
        if logger is not None:
            logger.log_iteration(0, exploitabilities[0], np.asarray(state.mean_field))
        for k in tqdm(range(1, self.num_iterations + 1), desc="Running"):
            _, policy_best_response = Vpi_opt_jax(state.mean_field, self.env_spec)
            mean_field_br = mean_field_by_transition_kernel_multi_jax(
                policy_best_response,
                self.env_spec,
                num_iterations=self.num_transition_steps,
                initial_mean_field=state.mean_field,
            )
            mean_field_br = mean_field_br / mean_field_br.sum()

            if self.lambda_schedule == "fictitious_play":
                policy_history.append(policy_best_response.copy())
                mean_field_history.append(mean_field_br.copy())

            alpha_k = self._lambda_k(k)
            state.mean_field = (
                1.0 - alpha_k
            ) * state.mean_field + alpha_k * mean_field_br
            state.mean_field = state.mean_field / state.mean_field.sum()

            state.policy = policy_best_response
            state_mf_jax = state.mean_field
            if self.lambda_schedule == "fictitious_play" and len(policy_history) > 0:
                average_policy = self._average_policies_weighted(
                    policy_history, mean_field_history
                )
                exploitability = float(
                    exploitability_jax(
                        average_policy,
                        self.env_spec,
                        initial_mean_field=state_mf_jax,
                    )
                )
            else:
                exploitability = float(
                    exploitability_jax(
                        state.policy,
                        self.env_spec,
                        initial_mean_field=state_mf_jax,
                    )
                )
            exploitabilities.append(exploitability)
            if logger is not None:
                logger.log_iteration(k, exploitability, np.asarray(state.mean_field))

        final_policy = (
            average_policy
            if self.lambda_schedule == "fictitious_play"
            else state.policy
        )
        final_exploitability = float(
            exploitability_jax(
                final_policy,
                self.env_spec,
                initial_mean_field=state.mean_field,
            )
        )
        log.info("Exploitability (returned policy): %s", final_exploitability)
        self.env_spec.environment.stationary_mean_field = np.asarray(state.mean_field)
        return np.asarray(final_policy), np.asarray(state.mean_field), exploitabilities
