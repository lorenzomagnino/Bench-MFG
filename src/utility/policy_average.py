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
    best_actions = np.argmax(q_values, axis=1)
    policy = np.zeros_like(q_values)
    policy[np.arange(q_values.shape[0]), best_actions] = 1.0
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
    q_shifted = q_values - np.max(q_values, axis=1, keepdims=True)
    exp_q = np.exp(q_shifted / temperature)
    return exp_q / np.sum(exp_q, axis=1, keepdims=True)
