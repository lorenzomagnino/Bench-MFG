from __future__ import annotations

import numpy as np

from envs.mfg_model_class import MFGStationary


class ContractionGame(MFGStationary):
    """
    Contraction Game environment.

    A 2-state, 2-action deterministic MFG where:
    - State space: {0, 1}
    - Action space: {Stay, Switch}
    - Dynamics: x_{n+1} = x_n if Stay, x_{n+1} = 1-x_n if Switch
    - Reward: r(x, a, μ) = -C · I(a=Switch) - α μ(x)

    If C > α/(1-γ), the best response is unique and constant (always Stay),
    making the fixed-point operator a contraction.
    """

    def __init__(
        self,
        N_states: int,
        N_actions: int,
        N_noises: int,
        horizon: int,
        mean_field: np.ndarray,
        noise_prob: np.ndarray,
        switching_cost: float = 1.0,
        congestion_coefficient: float = 1.0,
        gamma: float = 0.99,
        is_noisy: bool = False,
        grid_dim: np.ndarray | None = None,
    ):
        if N_states != 2:
            raise ValueError("ContractionGame requires exactly 2 states")
        if N_actions != 2:
            raise ValueError("ContractionGame requires exactly 2 actions")
        if N_noises != 1:
            raise ValueError(
                "ContractionGame requires exactly 1 noise value (deterministic)"
            )

        if switching_cost < 0:
            raise ValueError("switching_cost must be non-negative")
        if congestion_coefficient < 0:
            raise ValueError("congestion_coefficient must be non-negative")

        self.switching_cost = float(switching_cost)
        self.congestion_coefficient = float(congestion_coefficient)
        self.is_noisy = bool(is_noisy)

        # Action mapping: 0 = Stay, 1 = Switch
        self.action_stay = 0
        self.action_switch = 1

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
        """
        Deterministic transition: x_{n+1} = x_n if Stay, x_{n+1} = 1-x_n if Switch.

        Parameters:
            mean_field: Not used (deterministic dynamics)
            state: Current state (0 or 1)
            action: Action taken (0=Stay, 1=Switch)
            noise: Not used (deterministic)

        Returns:
            Next state (0 or 1)
        """
        if state is None or action is None:
            raise ValueError("state and action must be provided")

        s = int(state)
        a = int(action)

        if not (0 <= s < self.N_states):
            raise ValueError(f"state {s} out of range [0, {self.N_states - 1}]")
        if not (0 <= a < self.N_actions):
            raise ValueError(f"action {a} out of range [0, {self.N_actions - 1}]")

        if a == self.action_stay:
            # Stay: x_{n+1} = x_n
            s_next = s
        elif a == self.action_switch:
            # Switch: x_{n+1} = 1 - x_n
            s_next = 1 - s
        else:
            raise ValueError(f"Invalid action {a}")

        return int(s_next)

    def reward(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
    ) -> float:
        """
        Reward function: r(x, a, μ) = -C · I(a=Switch) - α μ(x)

        Parameters:
            mean_field: Current mean field distribution
            state: Current state (0 or 1)
            action: Action taken (0=Stay, 1=Switch)

        Returns:
            Reward value
        """
        if mean_field is None or state is None or action is None:
            raise ValueError("mean_field, state, action must be provided")

        mf = np.asarray(mean_field, dtype=float).reshape(-1)
        if mf.shape[0] != self.N_states:
            raise ValueError(f"mean_field must have shape ({self.N_states},)")

        s = int(state)
        a = int(action)

        if not (0 <= s < self.N_states):
            raise ValueError(f"state {s} out of range [0, {self.N_states - 1}]")
        if not (0 <= a < self.N_actions):
            raise ValueError(f"action {a} out of range [0, {self.N_actions - 1}]")

        # Reward: r(x, a, μ) = -C · I(a=Switch) - α μ(x)
        switching_penalty = -self.switching_cost * float(a == self.action_switch)
        congestion_penalty = -self.congestion_coefficient * float(mf[s])

        r = switching_penalty + congestion_penalty
        return float(r)
