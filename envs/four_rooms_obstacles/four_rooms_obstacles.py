"""
2D 4-Rooms Aversion with Static Obstacles.

Grid:
  X = {0,...,10}^2 (11x11)

Actions/Noises (must be 5):
  0 -> up    (+1, 0)  # row 0 at bottom, row increases upward
  1 -> right (0, +1)
  2 -> down  (-1, 0)
  3 -> left  (0, -1)
  4 -> stay  (0, 0)

Dynamics:
  x_{t+1} = x_t + a_t + eps_{t+1}
  eps ~ Uniform(A) (or user-provided noise_prob)
  If proposed next cell is obstacle or out of bounds => stay.

Reward:
  r(x,a,mu) = -alpha * log(mu(x))
"""

from __future__ import annotations

import numpy as np

from envs.mfg_model_class import MFGStationary


class FourRoomsAversion2D(MFGStationary):
    def __init__(
        self,
        horizon: int,
        mean_field: np.ndarray,
        noise_prob: np.ndarray | None = None,
        gamma: float = 0.99,
        alpha: float = 1.0,
        epsilon: float = 1e-12,
        grid_dim: np.ndarray | None = None,
        doors: tuple[tuple[int, int], ...] | None = None,
    ):
        if grid_dim is None:
            grid_dim = np.array([11, 11], dtype=int)
        if len(grid_dim) != 2:
            raise ValueError("grid_dim must be length-2 [rows, cols]")

        self.rows = int(grid_dim[0])
        self.cols = int(grid_dim[1])
        if self.rows != 11 or self.cols != 11:
            raise ValueError("This 4-rooms spec expects an 11x11 grid (0..10)^2")

        self.alpha = float(alpha)
        self.epsilon = float(epsilon)

        self.N_states = self.rows * self.cols
        self.N_actions = 5
        self.N_noises = 5
        self.index_to_move: dict[int, tuple[int, int]] = {
            0: (1, 0),  # up (row 0 at bottom, row increases upward)
            1: (0, 1),  # right
            2: (-1, 0),  # down
            3: (0, -1),  # left
            4: (0, 0),  # stay
        }

        if doors is None:
            doors = ((2, 5), (8, 5), (5, 8), (5, 2))
        self.doors = tuple(doors)

        self.obstacle_mask = self._build_obstacle_mask()

        if noise_prob is None:
            noise_prob = np.ones(self.N_noises, dtype=float) / self.N_noises
        noise_prob = np.asarray(noise_prob, dtype=float)
        if noise_prob.shape != (self.N_noises,):
            raise ValueError(f"noise_prob must have shape ({self.N_noises},)")
        if not np.isclose(noise_prob.sum(), 1.0):
            raise ValueError("noise_prob must sum to 1")

        super().__init__(
            N_states=self.N_states,
            N_actions=self.N_actions,
            N_noises=self.N_noises,
            horizon=horizon,
            mean_field=mean_field,
            noise_prob=noise_prob,
            gamma=gamma,
            grid_dim=np.array([self.rows, self.cols], dtype=int),
        )

    # ---- indexing utils ----
    def state_to_coord(self, state: int) -> tuple[int, int]:
        row = state // self.cols
        col = state % self.cols
        return row, col

    def coord_to_state(self, row: int, col: int) -> int:
        return row * self.cols + col

    def in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_obstacle(self, row: int, col: int) -> bool:
        return bool(self.obstacle_mask[row, col])

    def _build_obstacle_mask(self) -> np.ndarray:
        """
        Cross walls at row=5 and col=5, except door coordinates.
        """
        mask = np.zeros((self.rows, self.cols), dtype=bool)
        mid_row, mid_col = self.rows // 2, self.cols // 2  # 5,5 for 11x11

        door_set = set(self.doors)

        # Vertical wall (all rows at col=5)
        for row in range(self.rows):
            if (row, mid_col) not in door_set:
                mask[row, mid_col] = True

        # Horizontal wall (all cols at row=5)
        for col in range(self.cols):
            if (mid_row, col) not in door_set:
                mask[mid_row, col] = True

        # Ensure doors are free
        for row, col in door_set:
            if self.in_bounds(row, col):
                mask[row, col] = False

        return mask

    # ---- MFGStationary interface ----
    def transition(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
        noise: int | None = None,
    ) -> int:
        if state is None or action is None or noise is None:
            raise ValueError("state, action, and noise must all be provided")

        row, col = self.state_to_coord(int(state))
        action_row, action_col = self.index_to_move[int(action)]
        noise_row, noise_col = self.index_to_move[int(noise)]

        proposed_row = row + action_row + noise_row
        proposed_col = col + action_col + noise_col

        if not self.in_bounds(proposed_row, proposed_col):
            return int(state)

        if self.is_obstacle(proposed_row, proposed_col):
            return int(state)

        return self.coord_to_state(proposed_row, proposed_col)

    def reward(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
    ) -> float:
        if mean_field is None or state is None or action is None:
            raise ValueError("mean_field, state, and action must all be provided")

        mean_field = np.asarray(mean_field, dtype=float).reshape(-1)
        if mean_field.shape[0] != self.N_states:
            raise ValueError(f"mean_field must have shape ({self.N_states},)")

        density = max(float(mean_field[int(state)]), self.epsilon)
        return float(-self.alpha * np.log(density))
