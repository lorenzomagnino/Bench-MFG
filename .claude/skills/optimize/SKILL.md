---
name: optimize
description: Profile the codebase, apply parallelism (CPU threads/processes, CUDA GPU), and migrate hot paths to JAX where it preserves clean functional structure.
---

## Step 1 — Profile first, optimize second

**Never optimise blind.**

### CPU profiling
```python
import cProfile, pstats, io

pr = cProfile.Profile()
pr.enable()
<call the function under test here>
pr.disable()

s = io.StringIO()
ps = pstats.Stats(pr, stream=s).sort_stats("cumulative")
ps.print_stats(20)
print(s.getvalue())
```

Run it and identify the **top-10 cumulative time** functions.
Only work on functions that appear in this list — everything else is premature optimisation.

### Memory profiling (if relevant)
```
pip install memory-profiler --quiet
python -m memory_profiler <script.py>
```

---

## Step 2 — Parallelism decision tree

For each hot function identified in Step 1, choose the right strategy:

```
Is the work I/O-bound (network, disk, database)?
  YES → use concurrent.futures.ThreadPoolExecutor
  NO  ↓
Is the work CPU-bound with GIL contention?
  YES → use concurrent.futures.ProcessPoolExecutor
        or multiprocessing.Pool
  NO  ↓
Is the function a numerical array operation (matmul, convolution,
  reduction, scan, map over arrays)?
  YES → migrate to JAX (see Step 3)
  NO  ↓
Does it run on batches of data that fit in GPU memory?
  YES → add CUDA path (see Step 4)
  NO  → leave as-is, document why in the report
```

### ThreadPoolExecutor pattern
```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def process_batch(items: list, *, max_workers: int = 8):
    results = []
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(_process_one, item): item for item in items}
        for future in as_completed(futures):
            results.append(future.result())
    return results
```

### ProcessPoolExecutor pattern
```python
from concurrent.futures import ProcessPoolExecutor

def parallel_map(fn, items: list, *, max_workers: int | None = None):
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(fn, items, chunksize=max(1, len(items) // (max_workers or 4))))
```

---

## Step 3 — JAX migration (functional hot paths)

Migrate a function to JAX **only when ALL of these hold**:
- It operates primarily on numerical arrays (NumPy/PyTorch tensors).
- It has no hidden global state or in-place mutation.
- The speedup is measurable (benchmark before/after with `timeit`).

### Migration pattern
```python
# BEFORE (NumPy)
import numpy as np

def compute_scores(embeddings: np.ndarray, query: np.ndarray) -> np.ndarray:
    return embeddings @ query / (np.linalg.norm(embeddings, axis=1) * np.linalg.norm(query))


# AFTER (JAX)
import jax
import jax.numpy as jnp
from functools import partial

@partial(jax.jit, static_argnames=())
def compute_scores(embeddings: jnp.ndarray, query: jnp.ndarray) -> jnp.ndarray:
    norms = jnp.linalg.norm(embeddings, axis=1) * jnp.linalg.norm(query)
    return (embeddings @ query) / norms

compute_scores_batch = jax.vmap(compute_scores, in_axes=(0, None))
```

### JAX rules
- No in-place mutation — use functional updates.
- No Python-side side effects inside `@jax.jit`.
- Use `jax.lax.scan` instead of Python `for` loops over array axes.
- Use `jax.vmap` instead of manual batch loops.

---

## Step 4 — CUDA path (non-JAX GPU)

```python
import torch

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_device(tensor: torch.Tensor) -> torch.Tensor:
    return tensor.to(DEVICE, non_blocking=True)
```

- Move tensors to device once at the boundary — not inside tight loops.
- Use `torch.compile()` (PyTorch ≥ 2.0) on hot functions before custom CUDA kernels.
- Use `torch.autocast("cuda")` for mixed-precision where appropriate.
- Always provide a CPU fallback.

---

## Step 5 — Benchmark before and after

```python
import timeit

baseline = timeit.timeit(lambda: original_fn(*args), number=100)
optimised = timeit.timeit(lambda: new_fn(*args), number=100)
speedup = baseline / optimised
print(f"Speedup: {speedup:.2f}x")
```

If `speedup < 1.1` (less than 10% gain), **revert** — the complexity cost is not worth it.

---

## Step 6 — Keep structure clean

After every change:
- Functions must still have docstrings explaining the device/parallelism strategy.
- Run `pytest -x -q` — all tests must still pass.

---

## Step 7 — Write the report

Create the report at:
```
reports/optimization/MM-DD/report_<short_uuid>.md
```

### Report "Findings" section:
```markdown
## Findings

### Profiling summary
| Function | File | Cumulative time | % of total |
|----------|------|-----------------|------------|
| ...      | ...  | ...ms           | ...%       |

### Optimisations applied
| Function | File | Strategy | Speedup | Notes |
|----------|------|----------|---------|-------|
| ...      | ...  | JAX jit  | 3.2x    | ...   |

### Optimisations considered but rejected
| Function | Reason |
|----------|--------|
| ...      | speedup < 10% |

### Remaining bottlenecks (need human review)
- <function>: <reason>
```

---

## Constraints

- Do NOT use `multiprocessing` inside a Jupyter notebook without `if __name__ == "__main__":`.
- Do NOT migrate to JAX if the module uses PyTorch autograd — mixing causes silent errors.
- All parallelism must be **opt-out** — code must run correctly single-threaded on CPU.
