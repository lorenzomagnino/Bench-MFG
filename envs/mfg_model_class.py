from abc import abstractmethod
from typing import Optional, Tuple

import numpy as np


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
        grid_dim: Optional[np.ndarray] = None,
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
        mean_field: Optional[np.ndarray] = None,
        state: Optional[int] = None,
        action: Optional[int] = None,
        noise: Optional[int] = None,
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
        mean_field: Optional[np.ndarray] = None,
        state: Optional[int] = None,
        action: Optional[int] = None,
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
        mean_field_copy = self.stationary_mean_field.copy()
        for _k in range(num_transition_steps):
            new_state_dist: np.ndarray = np.zeros(self.N_states)
            for s in range(self.N_states):
                for a in range(self.N_actions):
                    for n in range(self.N_noises):
                        new_state = self.transition(
                            mean_field=mean_field_copy,
                            state=s,
                            action=a,
                            noise=n,
                        )
                        new_state_dist[new_state] += (
                            policy[s, a] * mean_field_copy[s] * self.noise_prob[n]
                        )
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
        value_by_iterations: np.ndarray = np.zeros((self.horizon, self.N_states))
        for k in range(0, self.horizon - 1):
            for s in range(self.N_states):
                value_by_iterations[k + 1, s] = sum(
                    policy[s, a]
                    * (
                        self.reward(mean_field=mean_field, state=s, action=a)
                        + self.gamma
                        * sum(
                            self.noise_prob[n]
                            * value_by_iterations[
                                k,
                                self.transition(
                                    mean_field=mean_field,
                                    state=s,
                                    action=a,
                                    noise=n,
                                ),
                            ]
                            for n in range(self.N_noises)
                        )
                    )
                    for a in range(self.N_actions)
                )
        value_function = value_by_iterations[-1]
        return value_function

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

        for s in range(self.N_states):
            max_value = np.max(action_values[s])
            optimal_actions = np.where(np.abs(action_values[s] - max_value) < 1e-10)[0]

            if mixed_policy:
                pi_opt[s, optimal_actions] = 1.0 / len(optimal_actions)
            else:
                # Deterministic tie-breaking: always pick the first action with maximum value
                # This matches the JAX implementation's argmax behavior
                best_action = optimal_actions[0]
                pi_opt[s, best_action] = 1.0

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
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Computes the optimal value function and policy for a given mean field using dynamic programming.

        Parameters:
        mean_field: Mean field distribution.
        mixed_policy: If True, returns mixed policy when there are ties. If False, returns deterministic policy.
        Returns:
        tuple: Optimal value function and optimal policy.
        """
        V_opt_by_iterations: np.ndarray = np.zeros((self.horizon, self.N_states))
        final_action_values: np.ndarray = np.zeros((self.N_states, self.N_actions))
        for k in range(self.horizon - 1):
            for s in range(self.N_states):
                action_values = self._compute_action_values(
                    mean_field, s, V_opt_by_iterations[k]
                )
                V_opt_by_iterations[k + 1, s] = np.max(action_values)
                if k == self.horizon - 2:
                    final_action_values[s] = action_values.copy()
            pi_opt = self._derive_optimal_policy(final_action_values, mixed_policy)
        value_function_optimal = V_opt_by_iterations[-1]
        return value_function_optimal, pi_opt

    def Q_eval(self, policy: np.ndarray, mean_field: np.ndarray) -> np.ndarray:
        """
        Evaluates the state-action value function for a given policy and mean field using backward induction.

        Parameters:
        policy (array): Policy of dimension N_steps by N_states by N_actions.
        mean_field (array): Mean field of dimension N_steps by N_states.

        Returns:
        array: State-action value function of dimension N_steps by N_states by N_actions.
        """
        q_value_by_iteration: np.ndarray = np.zeros(
            (self.horizon, self.N_states, self.N_actions)
        )
        value_by_iteration: np.ndarray = np.zeros((self.horizon, self.N_states))
        for k in range(0, self.horizon - 1):
            for s in range(self.N_states):
                for a in range(self.N_actions):
                    q_value_by_iteration[k + 1, s, a] = self.reward(
                        mean_field=mean_field, state=s, action=a
                    ) + self.gamma * sum(
                        self.noise_prob[n]
                        * value_by_iteration[
                            k,
                            self.transition(
                                mean_field=mean_field, state=s, action=a, noise=n
                            ),
                        ]
                        for n in range(self.N_noises)
                    )
                value_by_iteration[k + 1, s] = sum(
                    policy[s, a] * q_value_by_iteration[k + 1, s, a]
                    for a in range(self.N_actions)
                )

        q_value_function = q_value_by_iteration[-2]
        return q_value_function

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
