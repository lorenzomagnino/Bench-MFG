from __future__ import annotations

from typing import Sequence

import numpy as np

from envs.mfg_model_class import MFGStationary


class MultipleEquilibria1DGame(MFGStationary):
    def __init__(
        self,
        N_states: int,
        N_actions: int,
        N_noises: int,
        horizon: int,
        mean_field: np.ndarray,
        noise_prob: np.ndarray,
        alpha: float = 1.0,
        beta: float = 1.0,
        targets: Sequence[int] | None = None,
        movement_penalty: float = 1.0,
        gamma: float = 0.99,
        is_noisy: bool = True,
        grid_dim: np.ndarray | None = None,
    ):
        if N_actions != 3:
            raise ValueError("MultipleEquilibria1DGame requires exactly 3 actions")
        if N_noises != 3:
            raise ValueError("MultipleEquilibria1DGame requires exactly 3 noise values")

        if alpha < 0:
            raise ValueError("alpha must be non-negative")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if movement_penalty < 0:
            raise ValueError("movement_penalty must be non-negative")

        self.alpha = float(alpha)
        self.beta = float(beta)
        self.movement_penalty = float(movement_penalty)
        self.is_noisy = bool(is_noisy)

        if targets is None:
            targets = (N_states // 4, (3 * N_states) // 4)
        if len(targets) < 2:
            raise ValueError("targets must contain at least 2 target states")

        self.targets = tuple(int(t) for t in targets)
        for t in self.targets:
            if not (0 <= t < N_states):
                raise ValueError(f"target {t} out of range [0, {N_states - 1}]")

        self.action_values = np.array([-1, 0, 1], dtype=int)
        self.noise_values = np.array([-1, 0, 1], dtype=int)

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
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
        noise: int | None = None,
    ) -> int:
        if state is None or action is None or noise is None:
            raise ValueError("state, action, noise must be provided")

        s = int(state)
        a_move = int(self.action_values[int(action)])
        eps = int(self.noise_values[int(noise)]) if self.is_noisy else 0

        s_next = s + a_move + eps
        s_next = int(np.clip(s_next, 0, self.N_states - 1))
        return s_next

    def reward(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
    ) -> float:
        if mean_field is None or state is None or action is None:
            raise ValueError("mean_field, state, action must be provided")

        mf = np.asarray(mean_field, dtype=float).reshape(-1)
        if mf.shape[0] != self.N_states:
            raise ValueError(f"mean_field must have shape ({self.N_states},)")

        s = int(state)
        a_move = int(self.action_values[int(action)])

        dist_to_targets = min(abs(s - t) for t in self.targets)

        # Reward: r(x,a,μ) = -c₁|a| - c₂ min{|x-x_L|, |x-x_R|} + α μ(x)
        # where c₁ = movement_penalty, c₂ = beta, α = alpha
        r = (
            -self.movement_penalty * float(abs(a_move))  # -c₁|a|
            - self.beta * float(dist_to_targets)  # -c₂ min{|x-x_L|, |x-x_R|}
            + self.alpha * float(mf[s])  # +α μ(x) - positive mean-field interaction
        )
        return float(r)
