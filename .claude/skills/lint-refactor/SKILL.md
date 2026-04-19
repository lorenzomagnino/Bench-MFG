---
name: lint-refactor
description: Enforce PEP 8, apply KISS, remove dead code, and improve structure across all Python files.
---

## Step 1 — Discover scope

1. List every `.py` file in the repository (excluding `venv/`, `.venv/`, `__pycache__/`, `migrations/`).
2. For each file, run:
   ```
   python -m pyflakes <file>
   python -m pycodestyle --max-line-length=99 <file>
   ```
   Collect the raw output. If `pyflakes` or `pycodestyle` is not installed, install them first:
   ```
   pip install pyflakes pycodestyle --quiet
   ```

---

## Step 2 — Fix in priority order

Work through every file that has violations. For each file:

### PEP 8 mechanics (auto-fixable)
- Run `python -m autopep8 --in-place --max-line-length=99 --aggressive --aggressive <file>`.
  Install if missing: `pip install autopep8 --quiet`
- After autopep8, re-run pycodestyle and fix any remaining issues by hand.

### KISS & structural improvements (manual)
Apply **only when the improvement is unambiguous** — do not over-engineer:

| Smell | Fix |
|---|---|
| Function > 40 lines | Split into smaller, single-responsibility helpers |
| Nesting depth > 3 | Extract inner blocks to named functions |
| Repeated logic (≥3 occurrences) | Extract to a shared helper or utility module |
| Mutable default arguments | Replace with `None` sentinel + body assignment |
| Bare `except:` clauses | Replace with `except Exception as e:` (or specific exception) |
| Unused imports / variables | Remove (pyflakes will flag these) |
| Magic numbers / strings | Extract to named constants at module top |
| God-class (> 300 lines, > 10 public methods) | Propose a split; only apply if safe without API breakage |
| `import *` | Expand to explicit imports |

### Naming conventions
- Variables/functions: `snake_case`
- Classes: `PascalCase`
- Constants: `UPPER_SNAKE_CASE`
- Private members: `_leading_underscore`

### Docstrings
- Every public function and class must have a Google-style docstring.
  If one is missing, add a minimal one (one-line summary + `Args:` / `Returns:` if non-trivial).

---

## Step 3 — Verify nothing broke

Run the project test suite:
```
pytest -x -q
```
If tests fail after your changes, **revert the specific edit** that caused the failure and note it in the report.

---

## Step 4 — Write the report

Create the report at:
```
reports/lint/MM-DD/report_<short_uuid>.md
```
Use today's actual date for `MM-DD` (e.g. `04-11`).
Generate `<short_uuid>` as 6 random alphanumeric chars (e.g. `a3f9bc`).

### Report "Findings" section:
```markdown
## Findings

### Violations fixed
| File | Rule | Count |
|------|------|-------|
| ...  | ...  | ...   |

### Items skipped (unsafe to auto-fix)
- <file>: <reason>

### Remaining violations (need human review)
- <file>:<line>: <message>
```

---

## Constraints

- Do NOT reformat files that are explicitly excluded by a `.flake8`, `setup.cfg`, or `pyproject.toml` config.
- Do NOT change string literals that are user-facing messages or API keys.
- Do NOT alter `migrations/` directories.
