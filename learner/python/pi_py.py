"""
Policy Iteration algorithms for Mean Field Games.

Implements:
- Standard Policy Iteration (greedy policy, direct MF update)
- Smooth Policy Iteration (greedy policy, averaged MF update)
- Boltzmann Policy Iteration (softmax policy, direct MF update)
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import numpy as np
from tqdm import tqdm

from envs.mfg_model_class import MFGStationary
from utility.policy_average import greedy_policy, softmax_policy

PIVariant = Literal[
    "policy_iteration", "smooth_policy_iteration", "boltzmann_policy_iteration"
]


@dataclass
class PIComponents:
    policy: np.ndarray
    mean_field: np.ndarray
    q_values: np.ndarray


class PI_python:
    def __init__(
        self,
        model: MFGStationary,
        initial_policy: np.ndarray,
        num_iterations: int,
        early_stopping_enabled: bool = False,
        variant: PIVariant = "policy_iteration",
        temperature: float = 0.5,
        damped_constant: Optional[float] = (
            None  # for smooth_policy_iteration: None = 1/(k+1), else constant
        ),
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
            model.horizon,
            model.N_states,
            model.N_actions,
        )
        self.num_iterations = num_iterations
        self.model = model
        self.initial_policy = initial_policy
        self.early_stopping_enabled = early_stopping_enabled
        self.variant = variant
        self.temperature = temperature
        self.damped_constant = damped_constant

        if self.variant not in (
            "policy_iteration",
            "smooth_policy_iteration",
            "boltzmann_policy_iteration",
        ):
            raise ValueError(f"Unknown PI variant: {self.variant}")

        if (  # noqa: SIM102
            self.variant == "smooth_policy_iteration"
            and self.damped_constant is not None
        ):
            if not (0 < self.damped_constant <= 1.0):
                raise ValueError(
                    f"damped_constant must be in (0,1], got {self.damped_constant}"
                )

    def initialize(self) -> Tuple[PIComponents, list]:
        """Initialize policy iteration components."""
        initial_policy = self.initial_policy
        initial_mean_field = self.model.mean_field_by_transition_kernel(
            initial_policy, num_transition_steps=20
        )
        self.model.update_stationary_mean_field(initial_mean_field)

        initial_q_values = self.model.Q_eval(initial_policy, initial_mean_field)

        pi_components = PIComponents(
            policy=initial_policy,
            mean_field=initial_mean_field,
            q_values=initial_q_values,
        )

        exploitability = self.model.exploitability(initial_policy)
        exploitabilities = [exploitability]

        return pi_components, exploitabilities

    def eval(self, logger=None):
        """Main Policy Iteration evaluation loop."""
        pi_components, exploitabilities = self.initialize()

        print(f"Initial Exploitability: {exploitabilities[0]}")
        print(f"PI Variant: {self.variant}")

        if self.variant == "smooth_policy_iteration":
            if self.damped_constant is None:
                print("Mean field averaging: 1/(k+1) (Fictitious Play style)")
            else:
                print(f"Mean field averaging: constant λ={self.damped_constant}")
        elif self.variant == "boltzmann_policy_iteration":
            print(f"Temperature: {self.temperature}")

        if logger is not None:
            logger.log_iteration(0, exploitabilities[0], pi_components.mean_field)

        for iteration in tqdm(
            range(1, self.num_iterations + 1),
            desc=f"Running PI ({self.variant})",
        ):
            pi_components.q_values = self.model.Q_eval(
                pi_components.policy, pi_components.mean_field
            )
            if self.variant in ("policy_iteration", "smooth_policy_iteration"):
                pi_components.policy = greedy_policy(pi_components.q_values)
            elif self.variant == "boltzmann_policy_iteration":
                pi_components.policy = softmax_policy(
                    pi_components.q_values, self.temperature
                )

            new_mean_field = self.model.mean_field_by_transition_kernel(
                pi_components.policy, num_transition_steps=20
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

            self.model.update_stationary_mean_field(pi_components.mean_field)

            exploitability = self.model.exploitability(pi_components.policy)
            exploitabilities.append(exploitability)

            if logger is not None:
                logger.log_iteration(
                    iteration, exploitability, pi_components.mean_field
                )

            if exploitability < 1e-6 and self.early_stopping_enabled:
                print(f"Reached Exploitability close to zero: {exploitability}")
                break

        print(f"Final Exploitability: {exploitabilities[-1]}")

        return (
            pi_components.policy,
            pi_components.mean_field,
            exploitabilities,
        )
