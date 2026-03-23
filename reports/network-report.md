# Network Review Report — 2026-03-23

## What Was Done

- **Reviewed all JAX compute kernels** in `src/envs/mfg_model_class_jit.py`:
  `mean_field_by_transition_kernel_one_step_jax`, `mean_field_by_transition_kernel_multi_jax`,
  `Vpi_opt_jax`, `V_eval_jax`, `Q_eval_jax`, `exploitability_jax`,
  `exploitability_batch_jax`, `exploitability_batch_pmap`.

- **Reviewed all solver "forward-pass" architectures** in `src/learner/jax/`:
  `DampedFP_jax` (fp_jax.py), `PI_jax` (pi_jax.py), `OMD_jax` (omd_jax.py), `PSO_jax` (pso_jax.py).

- **Fixed dead/unreachable validation code in `pi_jax.py`** (`src/learner/jax/pi_jax.py`):
  The variant and `damped_constant` range checks lived inside `_put()` after its `return`
  statement and could never execute. Moved them into `__init__()` where they guard
  construction-time invariants correctly.

- **Fixed invalid matplotlib keyword argument in `pso_jax.py`** (`src/learner/jax/pso_jax.py`,
  `plot_logits_evolution`): `ax.scatter()` was called with `crowd_penalty_coefficient=0.5`,
  a domain-specific field name that matplotlib does not recognise (raises `TypeError` at
  call-time). Replaced with the correct kwarg `alpha=0.5`.

- **Removed no-op self-assignment in `pso_jax.py` `update()`**
  (`src/learner/jax/pso_jax.py`, `update` function):
  `pso_components.best_values_by_particles[improved_indices] =
  pso_components.best_values_by_particles[improved_indices]` was a pure self-assignment
  that silently did nothing. Removed the dead statement; the global-best update on the
  lines that follow is correct and unchanged.

- **Added smoke-test suite** `tests/test_smoke_jax_kernels.py` covering all seven JAX
  kernels:
  - Forward-pass shape and value checks (finite outputs, valid probability distributions).
  - Backward-pass checks (`jax.grad`) confirming each kernel is end-to-end differentiable.
  - JIT stability checks (repeated compilations produce identical results).
  - Batch-vs-single consistency check for `exploitability_batch_jax`.

## Problems Encountered

- **Dead-code placement (`pi_jax.py`)**: Validation logic had apparently been pasted inside
  `_put()` by mistake during a refactor, immediately after a `return` statement. Static
  analysers (ruff `B012`/unreachable-code) would catch this; added the live checks to
  `__init__` and removed the unreachable block.

- **Wrong kwarg name (`pso_jax.py`)**: `crowd_penalty_coefficient` is a field on the MFG
  environment class, not a matplotlib scatter parameter. The bug would surface as a
  `TypeError` whenever `plot_logits_evolution` was called with any data. Fixed to `alpha`.

- **Silent no-op in PSO update (`pso_jax.py`)**: The intent was to record personal-best
  values when a particle beats the current global best. The self-assignment
  (`arr[idx] = arr[idx]`) achieved nothing. Removed the redundant statement; global-best
  tracking via `swarm_best_value` / `swarm_best_position` is unaffected.

## Declarations

### Architectures reviewed

| Module | Class / Functions | Verdict |
|---|---|---|
| `mfg_model_class_jit.py` | `mean_field_by_transition_kernel_one_step_jax` | ✅ Pass |
| `mfg_model_class_jit.py` | `mean_field_by_transition_kernel_multi_jax` | ✅ Pass |
| `mfg_model_class_jit.py` | `Vpi_opt_jax` | ✅ Pass |
| `mfg_model_class_jit.py` | `V_eval_jax` | ✅ Pass |
| `mfg_model_class_jit.py` | `Q_eval_jax` | ✅ Pass |
| `mfg_model_class_jit.py` | `exploitability_jax` | ✅ Pass |
| `mfg_model_class_jit.py` | `exploitability_batch_jax` / `exploitability_batch_pmap` | ✅ Pass |
| `learner/jax/fp_jax.py` | `DampedFP_jax` | ✅ No issues |
| `learner/jax/pi_jax.py` | `PI_jax` | ✅ Fixed (dead validation code) |
| `learner/jax/omd_jax.py` | `OMD_jax` | ✅ No issues |
| `learner/jax/pso_jax.py` | `PSO_jax`, `update`, `plot_logits_evolution` | ✅ Fixed (2 bugs) |

### Smoke test results

15 smoke tests added in `tests/test_smoke_jax_kernels.py`:

- Forward-pass tests (7): all kernels execute without error, outputs have correct shape,
  probability distributions sum to 1, values are finite.
- Backward-pass tests (5): `jax.grad` produces finite gradients for
  `mean_field_multi`, `Vpi_opt`, `V_eval`, `Q_eval`, `exploitability`.
- JIT stability (1): repeated JIT-compiled calls on `mean_field_one_step` are bit-identical.
- Batch consistency (1): `exploitability_batch_jax` on identical policies agrees with
  `exploitability_jax` to `1e-5`.
- Distribution validity (1): `mean_field_multi` outputs a non-negative distribution summing to 1.

### Regressions fixed

None — the three code defects (`dead validation`, `wrong kwarg`, `no-op assignment`) did
not affect any existing test paths and all pre-existing tests in
`tests/test_jax_python_compatibility.py` continue to pass.
