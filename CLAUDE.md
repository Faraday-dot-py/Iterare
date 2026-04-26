# Iterare — Claude Code Guidelines

## Notifications (Pushbullet)

Worker scripts that run on TIDE should include push notifications so experiment
progress is visible on the user's phone without polling.

### Rule
Every worker script that runs a long experiment MUST include the inline `notify()`
helper and call it at: script start, each sub-unit completion (seed/lambda/round),
and script end. Wrap the main body in `try/except` to also notify on crash/restart.

### Inline helper (copy verbatim into every worker script)

```python
import os as _os, json as _json, urllib.request as _urlreq

def notify(title, body=""):
    """Pushbullet push. No-ops silently if PUSHBULLET_API_KEY not set."""
    key = _os.environ.get("PUSHBULLET_API_KEY", "")
    if not key:
        return
    try:
        data = _json.dumps({"type": "note", "title": title, "body": body}).encode()
        req = _urlreq.Request(
            "https://api.pushbullet.com/v2/pushes", data=data, method="POST",
            headers={"Access-Token": key, "Content-Type": "application/json"},
        )
        _urlreq.urlopen(req, timeout=5)
    except Exception:
        pass
```

Uses stdlib only (`urllib.request`) — no pip install required on TIDE.

### Key injection via `run_script`

Never embed the API key in script files (they're committed to git). Instead, pass
it via `env_inject` when submitting to TIDE:

```python
from iterare.notify import notify  # local notifications
from code.tools.tide.jobs import run_script

result = run_script(
    client, "path/to/worker.py",
    env_inject={"PUSHBULLET_API_KEY": os.getenv("PUSHBULLET_API_KEY", "")},
)
```

The `run_script` function prepends `os.environ.update({...})` to the uploaded
script before execution, so the key is never written to disk on TIDE.

### Notification call sites

| Event | Call |
|-------|------|
| Script start (after model loads) | `notify("ExpN started", "brief description")` |
| Sub-unit done (seed / λ / round) | `notify("ExpN seed=K done", f"proj={...:.4f} → hf={...:.4f}")` |
| New overall best | `notify("ExpN NEW BEST", f"CE={...:.4f}")` |
| Script complete | `notify("ExpN complete", f"best CE={...:.4f}")` |
| Unhandled exception | `notify("ExpN FAILED", str(e))` in `except` block |

### Local use

```python
from iterare.notify import notify
notify("steer-001 watcher", "Exp16 results downloaded")
```

---

## Session Hydration (automatic)

The `UserPromptSubmit` hook (`src/iterare/hooks/prompt_hydrate.py`) runs on every
prompt and injects active run context as `additionalContext` when any task has a
`run.yaml` with `status: active`.

**When injected context is present:**

1. Treat `resume.md` as the authoritative session state. Start from it — do not
   reconstruct intent from conversation history.
2. Respect stop rules listed in `run.yaml`. They are hard constraints. Check them
   before spawning new agents or continuing long-running work.
3. If pending approvals are listed, check them before actions that require
   write authority outside the active task's scope. Resolve with
   `iterare approvals approve <task_id> <approval_id>`.
4. The last checkpoint in `run.yaml` marks where state was last persisted.
   Resume from there if work was interrupted.

**When no context is injected:** no active runs exist; proceed normally.

---

## Commit policy

Commit experiment scripts + results after each experiment completes. Include the
key metric in the commit message (e.g. `HotFlip CE=0.686`).

## Bash commands

Write multi-line Python to a temp file and run it, rather than using
`python3 -c "..."` inline blocks.
