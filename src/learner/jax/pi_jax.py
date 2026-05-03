"""
Policy Iteration algorithms for Mean Field Games.

Implements:
- Standard Policy Iteration (greedy policy, direct MF update)
- Smooth Policy Iteration (greedy policy, averaged MF update)
- Boltzmann Policy Iteration (softmax policy, direct MF update)
"""

from dataclasses import dataclass
import logging
from typing import Literal

from envs.mfg_model_class_jit import (
    EnvSpec,
    Q_eval_jax,
    exploitability_jax,
    mean_field_by_transition_kernel_multi_jax,
)
import jax
import numpy as np
from tqdm import tqdm
from utility.policy_average import greedy_policy_jax, softmax_policy_jax

log = logging.getLogger(__name__)

PIVariant = Literal[
    "policy_iteration", "smooth_policy_iteration", "boltzmann_policy_iteration"
]


@dataclass
class PIComponents:
    policy: jax.Array
    mean_field: jax.Array
    q_values: jax.Array


class PI_jax:
    def __init__(
        self,
        env_spec: EnvSpec,
        initial_policy: np.ndarray,
        num_iterations: int,
        early_stopping_enabled: bool = False,
        variant: PIVariant = "policy_iteration",
        temperature: float = 0.5,
        damped_constant: float | None = (
            None  # for smooth_policy_iteration: None = 1/(k+1), else constant
        ),
        jax_device=None,
    ) -> None:
        """
        Policy Iteration solver for MFG.

        Args:
            variant:
              - "policy_iteration": Greedy improvement, direct MF update
                  π_{k+1} = argmax_a Q(s,a), μ_{k+1} = T(π_{k+1})

              - "smooth_policy_iteration": Greedy improvement, averaged MF update
                  π_{k+1} = argmax_a Q(s,a),
                  μ_{k+1} = λ * T(π_{k+1}) + (1-λ) * μ_k
                  (λ = 1/(k+1) if averaging_weight=None, else constant)

              - "boltzmann_policy_iteration": Softmax improvement, direct MF update
                  π_{k+1} = softmax(Q(s,·) / τ), μ_{k+1} = T(π_{k+1})

            temperature: float = 0.5  # temperature for boltzmann_policy_iteration (used during algorithm iterations)
            damped_constant: Optional[float] = (
                None  # for smooth_policy_iteration: None = 1/(k+1), else constant
            )
        """
        self.horizon, self.N_states, self.N_actions = (
            env_spec.environment.horizon,
            env_spec.environment.N_states,
            env_spec.environment.N_actions,
        )
        self.num_iterations = num_iterations
        self.env_spec = env_spec
        self.initial_policy = initial_policy
        self.early_stopping_enabled = early_stopping_enabled
        self.variant = variant
        self.temperature = temperature
        self.damped_constant = damped_constant
        self.jax_device = (
            jax_device if jax_device is not None else jax.devices("cpu")[0]
        )

        if self.variant not in (
            "policy_iteration",
            "smooth_policy_iteration",
            "boltzmann_policy_iteration",
        ):
            raise ValueError(f"Unknown PI variant: {self.variant}")

        if (
            self.variant == "smooth_policy_iteration"
            and self.damped_constant is not None
            and not (0 < self.damped_constant <= 1.0)
        ):
            raise ValueError(
                f"damped_constant must be in (0,1], got {self.damped_constant}"
            )

    def _put(self, arr):
        """Place a numpy/JAX array on the configured JAX device."""
        return jax.device_put(arr, self.jax_device)

    def initialize(self) -> tuple[PIComponents, list]:
        """Initialize policy iteration components."""
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

        initial_q_values = Q_eval_jax(
            initial_policy,
            initial_mean_field,
            self.env_spec,
        )

        pi_components = PIComponents(
            policy=initial_policy,
            mean_field=initial_mean_field,
            q_values=initial_q_values,
        )

        exploitability = float(
            exploitability_jax(
                initial_policy,
                self.env_spec,
                initial_mean_field=initial_mean_field,
            )
        )
        exploitabilities = [exploitability]

        return pi_components, exploitabilities

    def eval(self, logger=None):
        """Main Policy Iteration evaluation loop."""
        pi_components, exploitabilities = self.initialize()

        log.info("Initial Exploitability: %s", exploitabilities[0])
        log.info("PI Variant: %s", self.variant)

        if self.variant == "smooth_policy_iteration":
            if self.damped_constant is None:
                log.info("Mean field averaging: 1/(k+1) (Fictitious Play style)")
            else:
                log.info("Mean field averaging: constant λ=%s", self.damped_constant)
        elif self.variant == "boltzmann_policy_iteration":
            log.info("Temperature: %s", self.temperature)

        if logger is not None:
            logger.log_iteration(
                0, exploitabilities[0], np.asarray(pi_components.mean_field)
            )

        for iteration in tqdm(
            range(1, self.num_iterations + 1),
            desc=f"Running PI ({self.variant})",
        ):
            pi_components.q_values = Q_eval_jax(
                pi_components.policy,
                pi_components.mean_field,
                self.env_spec,
            )
            if self.variant in ("policy_iteration", "smooth_policy_iteration"):
                pi_components.policy = greedy_policy_jax(pi_components.q_values)
            elif self.variant == "boltzmann_policy_iteration":
                pi_components.policy = softmax_policy_jax(
                    pi_components.q_values, self.temperature
                )

            new_mean_field = mean_field_by_transition_kernel_multi_jax(
                pi_components.policy,
                self.env_spec,
                num_iterations=20,
                initial_mean_field=pi_components.mean_field,
            )

            if self.variant == "smooth_policy_iteration":
                if self.damped_constant is None:
                    lambda_k = 1.0 / (iteration + 1)
                else:
                    lambda_k = self.damped_constant

                pi_components.mean_field = (
                    lambda_k * new_mean_field
                    + (1 - lambda_k) * pi_components.mean_field
                )
                pi_components.mean_field /= pi_components.mean_field.sum()
            else:
                pi_components.mean_field = new_mean_field

            exploitability = float(
                exploitability_jax(
                    pi_components.policy,
                    self.env_spec,
                    initial_mean_field=pi_components.mean_field,
                )
            )
            exploitabilities.append(exploitability)

            if logger is not None:
                logger.log_iteration(
                    iteration, exploitability, np.asarray(pi_components.mean_field)
                )

            if exploitability < 1e-6 and self.early_stopping_enabled:
                log.info("Reached Exploitability close to zero: %s", exploitability)
                break

        log.info("Final Exploitability: %s", exploitabilities[-1])
        self.env_spec.environment.stationary_mean_field = np.asarray(
            pi_components.mean_field
        )

        return (
            np.asarray(pi_components.policy),
            np.asarray(pi_components.mean_field),
            exploitabilities,
        )
