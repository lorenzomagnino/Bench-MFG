from abc import abstractmethod
from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

# ---------------------------------------------------------------------------
# Module-level helpers – defined at module scope so ProcessPoolExecutor can
# pickle them when spawning worker processes.
# ---------------------------------------------------------------------------


def _compute_transition_row(args):
    """Compute T[s, :, :] for a single state s (worker helper)."""
    env, s, mean_field, A, N = args
    return [
        [
            env.transition(mean_field=mean_field, state=s, action=a, noise=n)
            for n in range(N)
        ]
        for a in range(A)
    ]


def _compute_reward_row(args):
    """Compute R[s, :] for a single state s (worker helper)."""
    env, s, mean_field, A = args
    return [env.reward(mean_field=mean_field, state=s, action=a) for a in range(A)]


def _map_rows_with_fallback(worker_fn, args_list):
    """Try process-based row parallelism, but fall back to sequential execution.

    Some constrained runtimes do not permit the semaphore primitives required by
    ``ProcessPoolExecutor``. The Python reference path is correctness-oriented, so
    it should still work in those environments even if it cannot parallelize.
    """
    if len(args_list) <= 1:
        return [worker_fn(args) for args in args_list]

    n_workers = min(os.cpu_count() or 1, len(args_list))
    try:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            return list(pool.map(worker_fn, args_list))
    except (NotImplementedError, PermissionError, OSError):
        return [worker_fn(args) for args in args_list]


class MFGStationary:
    """
    Generic Mean Field Game (MFG) Model class.

    Attributes:
    N_states: Number of states.
    N_actions: Number of actions.
    N_noises: Number of noise values.
    horizon: Number of time steps.
    stationary_mean_field: Initial distribution of dimension N_states.
    noise_prob: Distribution of noise, of dimension N_noises.
    gamma: Discount factor (0 <= gamma <= 1).
    grid_dim: dimension of the grid [row, column] ("None" if there is no grid)

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
        grid_dim: np.ndarray | None = None,
    ):
        self.N_states: int = N_states
        self.N_actions: int = N_actions
        self.N_noises: int = N_noises
        self.horizon: int = horizon
        self.stationary_mean_field: np.ndarray = np.array(mean_field)
        self.gamma: float = gamma
        self.grid_dim = grid_dim
        self.noise_prob: np.ndarray = np.array(noise_prob)

    @abstractmethod
    def transition(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
        noise: int | None = None,
    ) -> int:
        """
        Transition function to be implemented by the user.

        Parameters:
        mean_field: Current mean field distribution.
        state: Current state.
        action: Action taken.
        noise: Noise value.

        Returns:
        int: New state.
        """
        pass

    @abstractmethod
    def reward(
        self,
        mean_field: np.ndarray | None = None,
        state: int | None = None,
        action: int | None = None,
    ) -> float:
        """
        Reward function to be implemented by the user.

        Parameters:
        mean_field: Current mean field distribution.
        state: Current state.
        action: Action taken.

        Returns:
        float: Reward.
        """
        pass

    def _build_transition_matrix(self, mean_field: np.ndarray) -> np.ndarray:
        """Build T[s, a, n] -> next_state array for a given mean field.

        Rows (per-state computations) are dispatched to a ``ProcessPoolExecutor``
        so that independent states run in parallel on multiple CPU cores.
        """
        S, A, N = self.N_states, self.N_actions, self.N_noises
        rows = _map_rows_with_fallback(
            _compute_transition_row,
            [(self, s, mean_field, A, N) for s in range(S)],
        )
        return np.array(rows, dtype=np.intp)

    def _build_reward_matrix(self, mean_field: np.ndarray) -> np.ndarray:
        """Build R[s, a] array for a given mean field.

        Rows (per-state computations) are dispatched to a ``ProcessPoolExecutor``
        so that independent states run in parallel on multiple CPU cores.
        """
        S, A = self.N_states, self.N_actions
        rows = _map_rows_with_fallback(
            _compute_reward_row,
            [(self, s, mean_field, A) for s in range(S)],
        )
        return np.array(rows)

    def mean_field_by_transition_kernel(
        self, policy: np.ndarray, num_transition_steps: int = 50
    ) -> np.ndarray:
        """
        Transition kernel to compute the new state distribution given a policy.

        Parameters:
        policy: Policy used by the representative agent.
        num_transition_steps: Number of transition steps to take. Theoretically if num_transition_steps is large enough, the mean field will converge to the stationary mean field.

        Returns:
        array: New state distribution.
        """
        S = self.N_states
        mean_field_copy = self.stationary_mean_field.copy()
        new_state_dist = mean_field_copy
        for _k in range(num_transition_steps):
            # Precompute T[s,a,n] for current mean field, then scatter-add
            T = self._build_transition_matrix(mean_field_copy)
            # probs[s,a,n] = policy[s,a] * mf[s] * noise_prob[n]
            probs = (
                policy[:, :, None]
                * mean_field_copy[:, None, None]
                * self.noise_prob[None, None, :]
            )
            new_state_dist = np.zeros(S)
            np.add.at(new_state_dist, T, probs)
            new_state_dist = new_state_dist / new_state_dist.sum()
            mean_field_copy = new_state_dist

        return new_state_dist

    def V_eval(self, policy: np.ndarray, mean_field: np.ndarray) -> np.ndarray:
        """
        Evaluates the value function for a given policy and mean field using backward induction with fixed horizon .
        Parameters:
        policy: Policy of dimension N_states by N_actions.
        mean_field: Mean field distribution.

        Returns:
        array: Value function of dimension horizon by N_states.
        """
        S = self.N_states
        # Precompute R[s,a] and T[s,a,n] once (mean_field is fixed throughout)
        R = self._build_reward_matrix(mean_field)
        T = self._build_transition_matrix(mean_field)

        V = np.zeros(S)
        for _ in range(self.horizon - 1):
            # expected_V_sa[s,a] = sum_n noise_prob[n] * V[T[s,a,n]]
            expected_V_sa = np.einsum("n,san->sa", self.noise_prob, V[T])
            Q_sa = R + self.gamma * expected_V_sa
            V = np.einsum("sa,sa->s", policy, Q_sa)

        return V

    def _derive_optimal_policy(
        self, action_values: np.ndarray, mixed_policy: bool = False
    ) -> np.ndarray:
        """
        Helper function to derive optimal policy from action values.

        Parameters:
        action_values: Action values for each state, shape (N_states, N_actions).
        mixed_policy: If True, returns mixed policy when there are ties.
                     If False, returns deterministic policy.

        Returns:
        array: Optimal policy of shape (N_states, N_actions).
        """
        pi_opt: np.ndarray = np.zeros((self.N_states, self.N_actions))

        if not mixed_policy:
            # Vectorized deterministic tie-breaking: argmax picks the first maximum
            best_actions = np.argmax(action_values, axis=1)
            pi_opt[np.arange(self.N_states), best_actions] = 1.0
            return pi_opt

        for s in range(self.N_states):
            max_value = np.max(action_values[s])
            optimal_actions = np.where(np.abs(action_values[s] - max_value) < 1e-10)[0]
            pi_opt[s, optimal_actions] = 1.0 / len(optimal_actions)

        return pi_opt

    def _compute_action_values(
        self, mean_field: np.ndarray, state: int, value_function: np.ndarray
    ) -> np.ndarray:
        """
        Helper function to compute action values for a given state.

        Parameters:
        mean_field: Mean field distribution.
        state: Current state.
        value_function: Value function for next iteration.

        Returns:
        array: Action values for the given state.
        """
        action_values: np.ndarray = np.zeros(self.N_actions)
        for a in range(self.N_actions):
            action_values[a] = self.reward(
                mean_field=mean_field, state=state, action=a
            ) + self.gamma * sum(
                self.noise_prob[n]
                * value_function[
                    self.transition(
                        mean_field=mean_field, state=state, action=a, noise=n
                    )
                ]
                for n in range(self.N_noises)
            )
        return action_values

    def Vpi_opt(
        self, mean_field: np.ndarray, mixed_policy: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Computes the optimal value function and policy for a given mean field using dynamic programming.

        Parameters:
        mean_field: Mean field distribution.
        mixed_policy: If True, returns mixed policy when there are ties. If False, returns deterministic policy.
        Returns:
        tuple: Optimal value function and optimal policy.
        """
        S = self.N_states
        # Precompute R[s,a] and T[s,a,n] once (mean_field is fixed throughout)
        R = self._build_reward_matrix(mean_field)
        T = self._build_transition_matrix(mean_field)

        V = np.zeros(S)
        final_Q = np.zeros((S, self.N_actions))
        for _ in range(self.horizon - 1):
            expected_V_sa = np.einsum("n,san->sa", self.noise_prob, V[T])
            Q_sa = R + self.gamma * expected_V_sa
            V = np.max(Q_sa, axis=1)
            final_Q = Q_sa

        pi_opt = self._derive_optimal_policy(final_Q, mixed_policy)
        return V, pi_opt

    def Q_eval(self, policy: np.ndarray, mean_field: np.ndarray) -> np.ndarray:
        """
        Evaluates the state-action value function for a given policy and mean field using backward induction.

        Parameters:
        policy (array): Policy of dimension N_steps by N_states by N_actions.
        mean_field (array): Mean field of dimension N_steps by N_states.

        Returns:
        array: State-action value function of dimension N_steps by N_states by N_actions.
        """
        S, A = self.N_states, self.N_actions
        # Precompute R[s,a] and T[s,a,n] once (mean_field is fixed throughout)
        R = self._build_reward_matrix(mean_field)
        T = self._build_transition_matrix(mean_field)

        V = np.zeros(S)
        Q_prev = np.zeros((S, A))
        Q = np.zeros((S, A))
        for _ in range(self.horizon - 1):
            Q_prev = Q
            expected_V_sa = np.einsum("n,san->sa", self.noise_prob, V[T])
            Q = R + self.gamma * expected_V_sa
            V = np.einsum("sa,sa->s", policy, Q)

        # Return second-to-last Q (matching original q_value_by_iteration[-2] behaviour)
        if self.horizon >= 3:
            return Q_prev
        return np.zeros((S, A))

    def exploitability(self, policy: np.ndarray) -> float:
        """
        Computes the exploitability of a given policy.

        Parameters:
        policy (array): Policy of dimension N_steps by N_states by N_actions.

        Returns:
        float: Exploitability value.
        """
        policy = policy.reshape(self.N_states, self.N_actions)
        mean_field_pi: np.ndarray = self.mean_field_by_transition_kernel(
            policy, num_transition_steps=50
        )
        V_pi: np.ndarray = self.V_eval(policy, mean_field_pi)
        V_opt, _ = self.Vpi_opt(mean_field_pi, mixed_policy=False)
        return np.dot(mean_field_pi, V_opt) - np.dot(mean_field_pi, V_pi)

    def update_stationary_mean_field(self, mean_field: np.ndarray) -> None:
        """
        Updates the stationary mean field.

        Parameters:
        mean_field: Mean field distribution.
        """
        self.stationary_mean_field = mean_field
