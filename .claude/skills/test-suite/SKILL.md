---
name: test-suite
description: Identify untested code, write missing tests, run the full suite, report failures, and save a structured report.
---

## Step 1 — Coverage baseline

Install coverage if needed:
```
pip install pytest pytest-cov --quiet
```

Run coverage to see what is already tested:
```
pytest --cov=. --cov-report=term-missing -q 2>&1 | tee /tmp/coverage_baseline.txt
```

Parse the output and build a list of:
- **Uncovered functions/methods** (lines marked as missed in term-missing output).
- **Completely untested modules** (0 % coverage).

Prioritise writing tests for:
1. Public API functions (no leading `_`).
2. Functions with complex branching logic (if/else chains, loops with conditions).
3. Functions that read/write files, call external services, or mutate state.

---

## Step 2 — Write missing tests

For each untested or under-tested function, create (or extend) a test file following these rules:

### File placement
- Mirror the source structure: tests for `src/foo/bar.py` go in `tests/foo/test_bar.py`.
- If a `tests/` directory does not exist, create it with an empty `__init__.py`.

### Test style
```python
# Use pytest (not unittest) exclusively.
# One test function = one behaviour.
# Name tests: test_<function>_<scenario>_<expected_outcome>

import pytest
from <module_path> import <function_or_class>


def test_<function>_happy_path():
    result = <function>(valid_input)
    assert result == expected_value


def test_<function>_edge_case_empty_input():
    with pytest.raises(ValueError):
        <function>([])


def test_<function>_returns_correct_type():
    result = <function>(valid_input)
    assert isinstance(result, expected_type)
```

### Fixtures & mocks
- Use `pytest.fixture` for shared setup.
- Use `unittest.mock.patch` or `pytest-mock` (install if absent) for external I/O.
- Do NOT hit real databases, networks, or GPUs in unit tests — mock them.

### Parametrize where repetitive
```python
@pytest.mark.parametrize("inp,expected", [
    (0, 0),
    (1, 1),
    (-1, 1),
])
def test_abs_value(inp, expected):
    assert abs(inp) == expected
```

---

## Step 3 — Run the full test suite

```
pytest -x -q --tb=short 2>&1 | tee /tmp/test_run.txt
```

- `-x` stops at first failure so failures don't cascade.
- Read `/tmp/test_run.txt` to identify every failure.

For each failure:
1. Read the full traceback.
2. Determine the root cause:
   - **Bug in source code** → fix the source, re-run, confirm green.
   - **Bug in test** (wrong expectation) → fix the test.
   - **Environmental issue** (missing dependency, wrong path) → fix environment setup.
3. If a fix touches source code and is **significant** (changes logic, not just a typo), flag it in the report under "Source fixes required".
4. Re-run after every fix. Repeat until all tests pass or remaining failures are explicitly marked as `xfail`.

---

## Step 4 — Final coverage check

```
pytest --cov=. --cov-report=term-missing -q 2>&1 | tee /tmp/coverage_final.txt
```

Compare baseline vs final coverage numbers.

---

## Step 5 — Write the report

Create the report at:
```
reports/tests/MM-DD/report_<short_uuid>.md
```

### Report "Findings" section:
```markdown
## Findings

### Coverage delta
| Module | Before | After |
|--------|--------|-------|
| ...    | ...%   | ...%  |

### Tests written
| Test file | Function(s) covered | Test count |
|-----------|---------------------|------------|
| ...       | ...                 | ...        |

### Test failures
| Test | Failure type | Root cause | Fix applied |
|------|-------------|------------|-------------|
| ...  | ...          | ...        | ...         |

### Source fixes required
| File | Function | Change description | Classification |
|------|----------|-------------------|----------------|
| ...  | ...      | ...               | bug fix / clarification |

### Remaining xfail / skipped tests
- <test_id>: <reason>
```

---

## Constraints

- Do NOT delete existing passing tests.
- Do NOT mock so aggressively that tests become meaningless.
- Source-code fixes made to pass tests must be **minimal and targeted**.
- GPU/CUDA-dependent code should be tested with a CPU fallback or skipped via:
  ```python
  pytest.mark.skipif(not torch.cuda.is_available(), reason="no GPU")
  ```
