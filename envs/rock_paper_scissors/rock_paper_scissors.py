"""
Rock-Paper-Scissors Mean Field Game

This module implements a non-potential Mean Field Game based on the Rock-Paper-Scissors
game. The environment uses a skew-symmetric interaction matrix, making it a cyclic game
that does not admit a potential function.

The reward function is: g(x, μ) = [Aμ]x = Σ_{y∈𝒳} Axy μ(y)
where A is the skew-symmetric interaction matrix:
    A = [ 0  -1   1 ]  (Rock)
        [ 1   0  -1 ]  (Paper)
        [ -1   1   0 ]  (Scissors)

This can be interpreted as: g(x, μ) = μ(prey of x) – μ(predator of x)

The Nash equilibrium is: μ* = [1/3, 1/3, 1/3]ᵀ

Since A is skew-symmetric (Aᵀ = -A), this MFG is not a potential game, and
gradient-based methods may exhibit oscillatory behavior around the equilibrium.
"""

from typing import Optional

import numpy as np

from envs.mfg_model_class import MFGStationary


class RockPaperScissors(MFGStationary):
    """
    Rock-Paper-Scissors Mean Field Game environment.

    State space: X = {0, 1, 2} representing Rock, Paper, Scissors (0-indexed)
    Action space: A = {0, 1, 2} representing choosing Rock, Paper, or Scissors

    Attributes:
        interaction_matrix (np.ndarray): Skew-symmetric interaction matrix A
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
        grid_dim: Optional[np.ndarray] = None,
    ):
        """
        Initialize the Rock-Paper-Scissors MFG environment.

        Parameters:
            N_states (int): Number of states (should be 3 for RPS)
            N_actions (int): Number of actions (should be 3 for RPS)
            N_noises (int): Number of noise values
            horizon (int): Time horizon
            mean_field (np.ndarray): Initial population distribution
            noise_prob (np.ndarray): Probability distribution over noise values
            gamma (float): Discount factor (default: 0.99)
            grid_dim (np.ndarray): Grid dimensions (optional, not used for RPS)
        """
        if N_states != 3:
            raise ValueError("Rock-Paper-Scissors requires exactly 3 states")
        if N_actions != 3:
            raise ValueError("Rock-Paper-Scissors requires exactly 3 actions")

        # Skew-symmetric interaction matrix
        # Axy = 1 means x beats y, Axy = -1 means x loses to y
        # Row 0 (Rock): beats Scissors (col 2), loses to Paper (col 1)
        # Row 1 (Paper): beats Rock (col 0), loses to Scissors (col 2)
        # Row 2 (Scissors): beats Paper (col 1), loses to Rock (col 0)
        self.interaction_matrix = np.array(
            [
                [0, -1, 1],  # Rock
                [1, 0, -1],  # Paper
                [-1, 1, 0],  # Scissors
            ],
            dtype=np.float64,
        )

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
        Transition function: action directly determines next state.

        In RPS, the action represents choosing which state (Rock, Paper, or Scissors)
        to be in. The transition is deterministic - the next state is the action.

        Parameters:
            mean_field (np.ndarray): Current mean field (not used in transition)
            state (int): Current state (0: Rock, 1: Paper, 2: Scissors)
            action (int): Action index (0: Rock, 1: Paper, 2: Scissors)
            noise (int): Noise index (not used in deterministic transition)

        Returns:
            int: Next state (same as action)
        """
        if action is None:
            raise ValueError("Action must be provided")

        # Action directly determines the next state
        # Ensure action is valid (0, 1, or 2)
        next_state = int(action) % self.N_states
        return next_state

    def reward(
        self, mean_field: np.ndarray, state: int, action: Optional[int] = None
    ) -> float:
        """
        Compute reward: g(x, μ) = [Aμ]x = Σ_{y∈𝒳} Axy μ(y)

        This can be interpreted as: g(x, μ) = μ(prey of x) – μ(predator of x)
        where prey is the state that x beats, and predator is the state that beats x.

        Parameters:
            mean_field (np.ndarray): Current mean field distribution
            state (int): Current state (0: Rock, 1: Paper, 2: Scissors)
            action (int): Action taken (not used in reward calculation)

        Returns:
            float: Reward value
        """
        # Compute g(x, μ) = [Aμ]x = Σ_{y∈𝒳} Axy μ(y)
        reward = np.dot(self.interaction_matrix[state], mean_field)
        return float(reward)
