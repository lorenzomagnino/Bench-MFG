import numpy as np


def greedy_policy(q_values: np.ndarray) -> np.ndarray:
    """
    Compute greedy (deterministic) policy from Q-values.

    Args:
        q_values: Q-values of shape (N_states, N_actions)

    Returns:
        np.ndarray: One-hot policy of shape (N_states, N_actions)
                   π(a|s) = 1 if a = argmax_a' Q(s,a'), else 0
    """
    policy = np.zeros_like(q_values)
    N_states = q_values.shape[0]
    for s in range(N_states):
        best_action = np.argmax(q_values[s])
        policy[s, best_action] = 1.0
    return policy


def softmax_policy(q_values: np.ndarray, temperature: float) -> np.ndarray:
    """
    Compute softmax (Boltzmann) policy from Q-values.

    Args:
        q_values: Q-values of shape (N_states, N_actions)
        temperature: softmax temperature

    Returns:
        np.ndarray: Softmax policy of shape (N_states, N_actions)
    """
    policy = np.zeros_like(q_values)
    N_states = q_values.shape[0]
    for s in range(N_states):
        q_shifted = q_values[s] - np.max(q_values[s])
        exp_q = np.exp(q_shifted / temperature)
        policy[s] = exp_q / np.sum(exp_q)
    return policy
