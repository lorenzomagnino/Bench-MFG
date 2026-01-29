"""
No Interaction Game on a Discrete Chain
"""

from typing import Optional

import numpy as np

from envs.mfg_model_class import MFGStationary


class NoInteractionChain(MFGStationary):
    """
    Discrete one-dimensional chain environment with no interaction dynamics.
    """

    def __init__(
        self,
        N_states: int,
        N_actions: int,
        N_noises: int,
        horizon: int,
        mean_field: np.ndarray,
        noise_prob: np.ndarray,
        gamma: float = 0.99,
        is_noisy: bool = True,
        movement_penalty: float = 0.1,
        grid_dim: Optional[np.ndarray] = None,
    ):
        """
        Initialize the No Interaction Chain environment.

        Parameters:
            N_states (int): Number of states in the chain
            horizon (int): Time horizon
            mean_field (np.ndarray): Initial population distribution
            noise_prob (np.ndarray): Probability distribution over noise values
            gamma (float): Discount factor (default: 0.99)
            noise_values (np.ndarray): Possible noise values (default: [-1, 0, 1])
            grid_dim (np.ndarray): Grid dimensions (optional)
        """

        self.is_noisy = is_noisy
        self.movement_penalty = movement_penalty
        self.action_to_direction = {0: -1, 1: 0, 2: 1}
        self.noise_to_movement = {0: -1, 1: 0, 2: 1}

        super().__init__(
            N_states=N_states,
            N_actions=N_actions,
            N_noises=N_noises,
            horizon=horizon,
            mean_field=mean_field,
            noise_prob=noise_prob,
            gamma=gamma,
            grid_dim=grid_dim,
        )

    def transition(
        self,
        mean_field: Optional[np.ndarray] = None,
        state: Optional[int] = None,
        action: Optional[int] = None,
        noise: Optional[int] = None,
    ) -> int:
        """
        Compute the next state according to: x' = x + a + ε

        Parameters:
            mean_field (np.ndarray): Current mean field (not used in transition dynamics)
            state (int): Current state (0-indexed)
            action (int): Action index (0: left, 1: stay, 2: right)
            noise (int): Noise index

        Returns:
            int: Next state with boundary conditions applied
        """
        if state is None or action is None or noise is None:
            raise ValueError("State, action, and noise must all be provided")

        direction = self.action_to_direction[action]
        noise_direction = self.noise_to_movement[noise]
        if not self.is_noisy:
            noise_direction = 0
        next_state = state + direction + noise_direction
        next_state = max(0, min(next_state, self.N_states - 1))
        return next_state

    def reward(self, mean_field: np.ndarray, state: int, action: int) -> float:
        """
        Compute reward: R(x, a, μ) = r(x, a) + center_reward(x) - g(x, μ)

        Parameters:
            mean_field (np.ndarray): Current mean field distribution
            state (int): Current state
            action (int): Action taken

        Returns:
            float: Reward value
        """
        direction = self.action_to_direction[action]
        action_penalty = -self.movement_penalty * abs(direction)

        return action_penalty + state
