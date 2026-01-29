from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from envs.mfg_model_class import MFGStationary

DynamicsStructure = Literal["additive", "multiplicative"]
RewardStructure = Literal["additive", "multiplicative"]
GameType = Literal["potential", "cyclic"]


@dataclass(frozen=True)
class MFGarnetSampling:
    # Sparsity / structure
    branching_factor: int = 5  # number of nonzero next-states in P0 for each (s,a)

    # Dynamics coupling
    dynamics_structure: DynamicsStructure = "additive"
    cp: float = 0.5
    rho_p: float = 0.5

    # Reward coupling
    reward_structure: RewardStructure = "additive"
    cr: float = 0.5
    rho_r: float = 0.5
    game_type: GameType = "potential"

    # Distributions / numerics
    eps: float = 1e-12  # avoid 0 division error
    relu_eps: float = 0.0  # small positive to add after ReLU
    seed: int = 0

    # Base reward
    reward_scale: float = 1.0


class MFGarnet(MFGStationary):
    """
    MF-GARNET environment.
    """

    def __init__(
        self,
        N_states: int,
        N_actions: int,
        N_noises: int,
        horizon: int,
        mean_field: np.ndarray,
        gamma: float = 0.99,
        sampling: MFGarnetSampling | None = None,
    ):
        if sampling is None:
            sampling = MFGarnetSampling()

        if N_states <= 1:
            raise ValueError("N_states must be >= 2")
        if N_actions <= 0:
            raise ValueError("N_actions must be >= 1")
        if N_noises <= 1:
            raise ValueError("N_noises must be >= 2 (used for inverse-CDF sampling)")
        if not (1 <= sampling.branching_factor <= N_states):
            raise ValueError("branching_factor must be in [1, N_states]")

        self.rng = np.random.default_rng(int(sampling.seed))

        # Randomize coupling coefficients from Uniform[0,1] as per paper
        cp = float(self.rng.uniform(0.0, 1.0))
        rho_p = float(self.rng.uniform(0.0, 1.0))
        cr = float(self.rng.uniform(0.0, 1.0))
        rho_r = float(self.rng.uniform(0.0, 1.0))

        # Create new config with randomized coefficients
        self.cfg = replace(
            sampling,
            cp=cp,
            rho_p=rho_p,
            cr=cr,
            rho_r=rho_r,
        )

        # Seeded random objects
        self.P0_support = np.zeros(
            (N_states, N_actions, self.cfg.branching_factor), dtype=int
        )
        self.P0_prob = np.zeros(
            (N_states, N_actions, self.cfg.branching_factor), dtype=float
        )

        # Sample sparse base transitions P0
        for s in range(N_states):
            for a in range(N_actions):
                ns = self.rng.integers(0, N_states, size=(self.cfg.branching_factor,))
                p = self.rng.dirichlet(np.ones(self.cfg.branching_factor))
                self.P0_support[s, a] = ns
                self.P0_prob[s, a] = p

        # Pre-compute dense P0 matrix for JIT functions
        # Note: NumPy's fancy indexing with duplicates keeps last value, so we replicate that
        self.P0_dense = np.zeros((N_states, N_actions, N_states), dtype=float)
        for s in range(N_states):
            for a in range(N_actions):
                base = np.zeros(N_states, dtype=float)
                base[self.P0_support[s, a]] += self.P0_prob[s, a]
                self.P0_dense[s, a] = base

        self.C = self.rng.normal(size=(N_states, N_actions, N_states, N_states)).astype(
            float
        )

        # Base reward R0(s,a)
        self.R0 = self.rng.normal(
            loc=0.0, scale=self.cfg.reward_scale, size=(N_states, N_actions)
        ).astype(float)

        # Interaction matrix M(s,y)
        M = self.rng.normal(size=(N_states, N_states)).astype(float)
        if self.cfg.game_type == "potential":
            self.M = 0.5 * (M + M.T)
        elif self.cfg.game_type == "cyclic":
            self.M = 0.5 * (M - M.T)
        else:
            raise ValueError(f"Unknown game_type: {self.cfg.game_type}")

        # Uniform noise for inverse-CDF sampling
        noise_prob = np.ones(N_noises, dtype=float) / float(N_noises)

        super().__init__(
            N_states=N_states,
            N_actions=N_actions,
            N_noises=N_noises,
            horizon=horizon,
            mean_field=mean_field,
            noise_prob=noise_prob,
            gamma=gamma,
            grid_dim=None,
        )

    def _mu_normalize(self, mu: np.ndarray) -> np.ndarray:
        mu = np.asarray(mu, dtype=float).reshape(-1)
        if mu.shape[0] != self.N_states:
            raise ValueError(f"mean_field must have shape ({self.N_states},)")
        s = float(mu.sum())
        if s <= 0:
            raise ValueError("mean_field must have positive mass")
        return mu / s

    def _p0_dense(self, s: int, a: int) -> np.ndarray:
        base = np.zeros(self.N_states, dtype=float)
        ns = self.P0_support[s, a]
        base[ns] += self.P0_prob[s, a]
        return base

    # to speed up the caculation of g
    # def _ensure_g_cache(self, mu: np.ndarray) -> None:
    #     mu = self._mu_normalize(mu)
    #     if getattr(self, "_cached_mu", None) is not None and np.allclose(mu, self._cached_mu):
    #         return
    #     self._cached_mu = mu
    #     self._G = np.einsum("sapy,y->sap", self.C, mu, optimize=True)

    def _transition_distribution(self, mu: np.ndarray, s: int, a: int) -> np.ndarray:
        """
        Compute p(.|s,a,mu) over all next-states as a dense vector of length N_states.
        Implements your Option A/B + normalization.
        """
        mu = self._mu_normalize(mu)
        base = self._p0_dense(s, a)
        g = self.C[s, a] @ mu

        if self.cfg.dynamics_structure == "additive":
            intensity = self.cfg.cp * base + self.cfg.rho_p * g
            intensity = np.maximum(0.0, intensity)
        elif self.cfg.dynamics_structure == "multiplicative":
            gate = self.cfg.cp + self.cfg.rho_p * g
            gate = np.maximum(0.0, gate)
            intensity = base * gate
        else:
            raise ValueError(
                f"Unknown dynamics_structure: {self.cfg.dynamics_structure}"
            )

        if self.cfg.relu_eps > 0:
            intensity = intensity + self.cfg.relu_eps

        denom = float(intensity.sum()) + float(self.cfg.eps)
        p = intensity / denom
        p = p / max(float(p.sum()), float(self.cfg.eps))
        return p

    def transition(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
        noise: int | None = None,
    ) -> int:
        if mean_field is None or state is None or action is None or noise is None:
            raise ValueError("mean_field, state, action, noise must be provided")

        s = int(state)
        a = int(action)
        n = int(noise)

        p = self._transition_distribution(mean_field, s, a)

        # Inverse CDF sampling using discretized uniform(0,1)
        u = (n + 0.5) / float(self.N_noises)  # in (0,1)
        cdf = np.cumsum(p)
        s_next = int(np.searchsorted(cdf, u, side="right"))
        if s_next >= self.N_states:
            s_next = self.N_states - 1
        return s_next

    def reward(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
    ) -> float:
        if mean_field is None or state is None or action is None:
            raise ValueError("mean_field, state, action must be provided")

        mu = self._mu_normalize(mean_field)
        s = int(state)
        a = int(action)
        interaction = float(self.M[s] @ mu)

        if self.cfg.reward_structure == "additive":
            return float(self.cfg.cr * self.R0[s, a] + self.cfg.rho_r * interaction)
        elif self.cfg.reward_structure == "multiplicative":
            return float(self.R0[s, a] * (self.cfg.cr + self.cfg.rho_r * interaction))
        else:
            raise ValueError(f"Unknown reward_structure: {self.cfg.reward_structure}")
