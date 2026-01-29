"""
SIS Epidemic Model - Dynamics-Coupled Mean Field Game

This module implements a Dynamics-Coupled MFG where the transition probabilities
depend explicitly on the population distribution μ, while the reward function
is decoupled from μ.

The SIS (Susceptible-Infected-Susceptible) model represents a population where
agents balance social activity with health risks.

State Space: X = {0: Susceptible (S), 1: Infected (I)}
Action Space: Discretized intensity levels of social activity [0, 1]
Dynamics (Coupled): p(I|S, a, μ) = β · a · μ(I), recovery I→S with probability ν
Reward (Decoupled): r(x, a) = a - C · I(x = I)
"""

from typing import Optional

import numpy as np

from envs.mfg_model_class import MFGStationary


class SISEpidemic(MFGStationary):
    """
    SIS Epidemic model where infection probability depends on population prevalence.

    State space: X = {0 (S), 1 (I)}
    Action space: Discretized intensity levels of social activity [0, 1]
    """

    def __init__(
        self,
        N_actions: int,  # Number of intensity levels (e.g., 5 for 0, 0.25, 0.5, 0.75, 1.0)
        horizon: int,
        mean_field: np.ndarray,
        beta: float = 0.5,  # Transmission rate
        nu: float = 0.1,  # Recovery rate
        cost_infection: float = 5.0,
        gamma: float = 0.99,
    ):
        # TRICK: We define Noise as a Uniform Distribution over [0, 1]
        # We use 100 discrete points to approximate continuous probability
        self.precision_N = 100
        noise_prob = np.full(self.precision_N, 1.0 / self.precision_N)

        self.beta = beta
        self.nu = nu
        self.cost_infection = cost_infection
        self.action_levels = np.linspace(0, 1, N_actions)

        super().__init__(
            N_states=2,
            N_actions=N_actions,
            N_noises=self.precision_N,
            horizon=horizon,
            mean_field=mean_field,
            noise_prob=noise_prob,
            gamma=gamma,
            grid_dim=None,
        )

    def transition(
        self,
        mean_field: Optional[np.ndarray] = None,
        state: Optional[int] = None,
        action: Optional[int] = None,
        noise: Optional[int] = None,
    ) -> int:
        """
        Transition logic using 'noise' as a probability threshold.

        Parameters:
            mean_field (np.ndarray): Current mean field distribution (required)
            state (int): Current state (0: Susceptible, 1: Infected)
            action (int): Action index
            noise (int): Noise index (used as probability threshold)

        Returns:
            int: Next state
        """
        if mean_field is None or state is None or action is None or noise is None:
            raise ValueError("mean_field, state, action, and noise must be provided")

        random_roll = (noise + 0.5) / self.precision_N

        social_intensity = self.action_levels[action]
        if state == 0:
            prevalence = mean_field[1]
            prob_infection = self.beta * social_intensity * prevalence
            prob_infection = min(max(prob_infection, 0), 1)

            if random_roll < prob_infection:
                return 1
            else:
                return 0

        elif state == 1:
            if random_roll < self.nu:
                return 0
            else:
                return 1

        return state

    def reward(
        self,
        mean_field: Optional[np.ndarray] = None,
        state: Optional[int] = None,
        action: Optional[int] = None,
    ) -> float:
        """
        Reward function: r(x, a) = a - C * I(x=I)

        The reward is decoupled from the mean field - it depends only on
        state and action.

        Parameters:
            mean_field (np.ndarray): Current mean field (not used in reward)
            state (int): Current state (0: Susceptible, 1: Infected)
            action (int): Action index

        Returns:
            float: Reward value
        """
        if state is None or action is None:
            raise ValueError("state and action must be provided")

        social_intensity = self.action_levels[action]

        utility = social_intensity
        infection_cost = self.cost_infection if state == 1 else 0.0

        return utility - infection_cost
