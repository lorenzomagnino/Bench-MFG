# Vectorization & JIT Report — 2026-03-23

## What Was Done

- **`src/utility/policy_average.py` — `greedy_policy`**: Replaced per-state `for` loop with a single `np.argmax(q_values, axis=1)` call followed by a vectorized scatter (`policy[np.arange(N), best_actions] = 1.0`). Eliminated O(N_states) Python iterations.

- **`src/utility/policy_average.py` — `softmax_policy`**: Replaced per-state `for` loop with NumPy broadcasting: `q_shifted = q_values - np.max(q_values, axis=1, keepdims=True)`, then `exp / sum` across the action axis. Eliminated O(N_states) Python iterations.

- **`src/envs/mfg_model_class.py` — `_build_transition_matrix` / `_build_reward_matrix`**: Added two helper methods that materialise `T[s,a,n]` (dtype `intp`) and `R[s,a]` as NumPy arrays from the abstract `transition` / `reward` callbacks. Callers that fix `mean_field` for a whole backward-induction pass can now call these once and reuse the arrays throughout the loop.

- **`src/envs/mfg_model_class.py` — `mean_field_by_transition_kernel`**: Replaced the triple Python `for` loop `(s, a, n)` inside each transition step with a precomputed `T = _build_transition_matrix(mean_field_copy)` (same total callback count) followed by a broadcasting multiply and `np.add.at(new_state_dist, T, probs)` scatter-add. The inner body is now two NumPy array operations instead of O(S·A·N) Python iterations per step.

- **`src/envs/mfg_model_class.py` — `V_eval`**: Replaced the double Python `for` loop `(k, s)` with precomputed `R`, `T` (called once outside the horizon loop) and two `np.einsum` expressions per Bellman step: `expected_V_sa = np.einsum("n,san->sa", noise_prob, V[T])` and `V = np.einsum("sa,sa->s", policy, Q_sa)`.

- **`src/envs/mfg_model_class.py` — `Vpi_opt`**: Same approach as `V_eval` — precomputed `R`, `T`; einsum Bellman steps; replaced the per-state `_compute_action_values` call loop with a single batched `np.max(Q_sa, axis=1)`.

- **`src/envs/mfg_model_class.py` — `Q_eval`**: Precomputed `R`, `T`; einsum Bellman steps tracking both `V` and `Q` per iteration; preserves the original `q_value_by_iteration[-2]` return semantics (second-to-last Q, matching `Q_eval_jax`).

- **`src/envs/mfg_model_class.py` — `_derive_optimal_policy`**: Vectorized the deterministic (`mixed_policy=False`) branch with `np.argmax(action_values, axis=1)` + vectorized scatter. The mixed (tie-sharing) branch retains a state loop because per-row tie counts differ.

- **`src/learner/jax/pso_jax.py` — `_batch_exploitability` mellowmax branch**: Replaced the Python list comprehension `[mellowmax(logits, N_actions) for logits in logits_batch]` with fully vectorized NumPy broadcasting over the particle batch dimension (`logits_batch` shape `(P, S, A)`), eliminating O(num_particles) Python calls.

## Problems Encountered

- **`Q_eval` return semantics**: The original code returns `q_value_by_iteration[-2]` (second-to-last, not final Q), matching `Q_eval_jax`. Carefully tracked `Q_prev` across iterations to reproduce this, with a special-case `np.zeros` return for `horizon < 3`.

- **`mean_field_by_transition_kernel` with mutable mean field**: Because the transition callback receives the current `mean_field_copy` (which changes each step), `T` must be rebuilt each outer iteration. The total number of `transition` calls is unchanged; the gain is replacing the O(S·A·N) scatter loop with `np.add.at`.

- **`mellowmax` omega hardcode**: The standalone `mellowmax` function uses `omega=16.55` as a default. The vectorized batch path in `_batch_exploitability` inlines the same default; both paths remain consistent.

## Declarations

| File | Change | Estimated Speedup |
|------|--------|-------------------|
| `src/utility/policy_average.py` | `greedy_policy`: loop → vectorized argmax + scatter | ~10–50× for large N_states |
| `src/utility/policy_average.py` | `softmax_policy`: loop → broadcasting exp/sum | ~10–50× for large N_states |
| `src/envs/mfg_model_class.py` | `mean_field_by_transition_kernel`: triple loop body → `np.add.at` scatter | ~5–20× inner body per step |
| `src/envs/mfg_model_class.py` | `V_eval`: double loop → einsum Bellman steps | ~20–100× for realistic (S,A,N,H) |
| `src/envs/mfg_model_class.py` | `Vpi_opt`: double loop → einsum + `np.max` | ~20–100× for realistic (S,A,N,H) |
| `src/envs/mfg_model_class.py` | `Q_eval`: triple loop → einsum Bellman steps | ~20–100× for realistic (S,A,N,H) |
| `src/envs/mfg_model_class.py` | `_derive_optimal_policy` (deterministic): per-state loop → argmax + scatter | ~10–50× for large N_states |
| `src/learner/jax/pso_jax.py` | `_batch_exploitability` mellowmax: per-particle Python loop → NumPy batch ops | ~num_particles× |
