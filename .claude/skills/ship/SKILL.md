---
name: ship
description: Stage relevant changes, generate a commit message, commit (pre-commit runs automatically), and push to origin.
---

## Step 1 — Understand what changed

Run the following in parallel:
```
git status
git diff
git log --oneline -5
```

From `git status`, identify all modified/added/deleted files.
From `git diff`, understand the nature of every change.
From `git log`, learn the commit message style used in this repo.

---

## Step 2 — Select files to stage

Stage **only** files that are part of the logical change. Never stage:
- `.env`, `*.pem`, `*secret*`, `*credential*`, or any file that could contain secrets.
- Binary files or large generated files unless they are part of the intended change.
- `__pycache__/`, `.pyc`, `.DS_Store`.

Use specific file names — never `git add .` or `git add -A`.

---

## Step 3 — Draft the commit message

Write a concise commit message that follows the style of recent commits in this repo:
- First line: imperative mood, ≤ 72 characters, summarises *what and why*.
- If multiple logical changes are present, add a short bullet list after a blank line.
- End with:
  ```
  Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
  ```

---

## Step 4 — Commit

Pass the message via HEREDOC to avoid shell quoting issues:
```bash
git commit -m "$(cat <<'EOF'
<your message here>

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>
EOF
)"
```

Pre-commit hooks run automatically. If a hook **fails**:
1. Read the error output carefully.
2. Fix the flagged issue (formatting, lint, etc.).
3. Re-stage the affected files.
4. Create a **new** commit — never amend.

---

## Step 5 — Push

```bash
git push
```

If the push is rejected because the remote has new commits, run:
```bash
git pull --rebase && git push
```

Report the final remote URL and branch that was pushed to.
