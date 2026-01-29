"""
Kinetic Congestion / Crowd Dynamics - Dynamics-Coupled Mean Field Game

This module implements a Dynamics-Coupled MFG where movement success depends on
the population density at the target state, while the reward function is
decoupled from μ.

In standard congestion games, high density is penalized via costs. In Kinetic
Congestion, high density physically prevents movement, mimicking fluid dynamics
or pedestrian crowding.

State Space: Grid positions (grid_height × grid_width)
Action Space: {0: Up, 1: Right, 2: Down, 3: Left, 4: Stay}
Dynamics (Coupled): p(y|x, a=y, μ) = 1 - min(1, φ(μ(y)))
                    where φ is a crowding function
Reward (Decoupled): r(x, a) = -I(x ≠ x_target) - c_move · I(a ≠ stay)
"""

from typing import Optional

import numpy as np

from envs.mfg_model_class import MFGStationary


class KineticCongestion(MFGStationary):
    """
    Grid environment where movement success depends on target cell density.

    This is a Dynamics-Coupled MFG where the transition probabilities depend
    on the population distribution μ at the target state, while rewards are
    decoupled from μ.
    """

    def __init__(
        self,
        grid_height: int,
        grid_width: int,
        horizon: int,
        mean_field: np.ndarray,
        target_state: int,
        movement_cost: float = 0.1,
        capacity_threshold: float = 0.4,
        gamma: float = 0.99,
    ):
        """
        Initialize the Kinetic Congestion environment.

        Parameters:
            grid_height (int): Height of the grid
            grid_width (int): Width of the grid
            horizon (int): Time horizon
            mean_field (np.ndarray): Initial population distribution
            target_state (int): Target state index (0-indexed)
            movement_cost (float): Cost for moving (default: 0.1)
            capacity_threshold (float): Maximum allowed density per cell (default: 0.4)
            gamma (float): Discount factor (default: 0.99)
        """
        self.height = grid_height
        self.width = grid_width
        self.target_state = target_state
        self.movement_cost = movement_cost
        self.capacity_threshold = capacity_threshold

        N_states = grid_height * grid_width

        # Actions: 0:Up, 1:Right, 2:Down, 3:Left, 4:Stay
        N_actions = 5

        # TRICK: Noise as Uniform Distribution [0, 1]
        self.precision_N = 100
        noise_prob = np.full(self.precision_N, 1.0 / self.precision_N)

        super().__init__(
            N_states=N_states,
            N_actions=N_actions,
            N_noises=self.precision_N,
            horizon=horizon,
            mean_field=mean_field,
            noise_prob=noise_prob,
            gamma=gamma,
            grid_dim=np.array([grid_height, grid_width]),
        )

    def _get_neighbor(self, state: int, action: int) -> int:
        """
        Helper to get target coordinate based on grid logic.

        Grid coordinate system: row 0 is at the bottom, row increases upward.
        This matches visualization where row 0 appears at the bottom.

        Parameters:
            state (int): Current state index
            action (int): Action index (0:Up, 1:Right, 2:Down, 3:Left, 4:Stay)

        Returns:
            int: Target state index
        """
        row = state // self.width
        col = state % self.width

        if action == 0:  # Up (increase row, move toward top)
            row = min(self.height - 1, row + 1)
        elif action == 1:  # Right
            col = min(self.width - 1, col + 1)
        elif action == 2:  # Down (decrease row, move toward bottom)
            row = max(0, row - 1)
        elif action == 3:  # Left
            col = max(0, col - 1)
        elif action == 4:  # Stay
            return state

        return row * self.width + col

    def phi_congestion(self, density: float) -> float:
        """
        Crowding function with capacity constraint.

        Returns probability of REJECTION based on density and capacity threshold.
        If density >= capacity_threshold, movement is always rejected (100% rejection).
        Otherwise, rejection probability increases linearly with density.

        Parameters:
            density (float): Population density at target state

        Returns:
            float: Probability of rejection (movement failure)
        """
        # If density exceeds or equals capacity threshold, always reject
        if density >= self.capacity_threshold:
            return 1.0
        # Otherwise, linear rejection based on normalized density
        # Normalize by capacity_threshold so rejection reaches 100% at threshold
        return min(1.0, density / self.capacity_threshold)

    def transition(
        self,
        mean_field: Optional[np.ndarray] = None,
        state: Optional[int] = None,
        action: Optional[int] = None,
        noise: Optional[int] = None,
    ) -> int:
        """
        Transition function: movement success depends on target density (dynamics-coupled).

        Uses noise to sample from the transition probability distribution.

        Parameters:
            mean_field (np.ndarray): Current mean field distribution (required)
            state (int): Current state index
            action (int): Action index
            noise (int): Noise index (used as probability threshold)

        Returns:
            int: Next state
        """
        if mean_field is None or state is None or action is None or noise is None:
            raise ValueError("mean_field, state, action, and noise must be provided")

        random_roll = (noise + 0.5) / self.precision_N
        target_state = self._get_neighbor(state, action)
        if target_state == state:
            return state
        target_density = mean_field[target_state]
        prob_rejection = self.phi_congestion(target_density)
        prob_success = 1.0 - prob_rejection

        if random_roll < prob_success:
            return target_state
        else:
            return state  # Move failed, bounce back

    def reward(
        self,
        mean_field: Optional[np.ndarray] = None,
        state: Optional[int] = None,
        action: Optional[int] = None,
    ) -> float:
        """
        Reward function: r(x, a) = -I(x ≠ x_target) - c_move · I(a ≠ stay)

        Pure target seeking reward + movement cost.
        Independent of mean_field (Decoupled reward).

        Parameters:
            mean_field (np.ndarray): Current mean field (not used in reward)
            state (int): Current state index
            action (int): Action index

        Returns:
            float: Reward value
        """
        if state is None or action is None:
            raise ValueError("state and action must be provided")

        is_at_target = 1.0 if state == self.target_state else 0.0

        move_cost = 0.0 if action == 4 else self.movement_cost
        dist_penalty = -1.0 * (1.0 - is_at_target)

        return dist_penalty - move_cost
