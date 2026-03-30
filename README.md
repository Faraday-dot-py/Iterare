# Iterare

Agent-native applied AI research platform. Runs through Claude Code — no Anthropic API key required.

## Architecture

```
Lead Developer
└── Interface Agent  ← this Claude Code session
    └── Master Agent  (spawned via Agent tool, on approval)
        └── Manager Agent  (spawned by Master via Agent tool)
            └── Worker Agent  (spawned by Manager via Agent tool)
```

Orchestration happens entirely through Claude Code's Agent tool. No separate Python process, no direct API calls.

## Setup

### 1. Install utilities

```bash
cd /home/awebb/Research/iterare
pip install -e .    # or: uv sync
```

### 2. Configure environment (optional)

```bash
cp .env.example .env
# ITERARE_ROOT defaults to the repo directory
```

### 3. Start working

Just talk to Claude Code. Pitch your research idea. The Interface Agent (Claude Code) will handle it directly or propose spawning a Master Agent — you approve before anything runs.

---

## CLI (inspection only)

The CLI is for inspecting state, not for running agents.

```bash
iterare tasks                    # list all tasks
iterare tasks task-20260330-...  # show task state + log
iterare requests                 # list pending tool requests
iterare requests approve req-abc # approve a tool request
iterare requests reject req-abc "reason"
```

---

## Directory Structure

```
iterare/
├── code/           — Finished, working code (organized by project)
│   └── tools/      — Finished tool implementations
├── tasks/          — Task trees: state, READMEs, worker outputs
│   └── task-<id>/
│       ├── state.yaml
│       ├── README.md          ← Master summary (written on completion)
│       ├── manager-000/
│       │   ├── README.md      ← Manager summary
│       │   └── worker-000/
│       │       └── README.md  ← Worker reasoning + results
│       └── shared/            ← Lateral worker communication
├── archive/        — Cold-stored sealed task logs
├── templates/      — Agent prompts (git-versioned, evolve over time)
├── tools/
│   ├── built/      — Finished tools (registered here when promoted from code/)
│   └── requests/   — Tool request queue (YAML, reviewed by Master/Interface)
├── logs/           — Structured JSONL logs (per-task, sealed on completion)
├── docs/           — System documentation
└── src/iterare/    — Python utilities (file tools, logging, task state)
    ├── tools/      — file_tools.py, tool_request.py
    └── utils/      — log.py, task.py
```

**Code promotion**: experimental work lives in `tasks/`. Finished code moves to `code/`. Tools also register in `tools/built/`.

---

## Evidence Quality

All README claims carry a quality marker:

| Level | Meaning |
|-------|---------|
| `[H]` | Peer-reviewed or multiple independent credible sources |
| `[M]` | Single credible source, or strong indirect evidence |
| `[L]` | Reasoning/inference; no direct citation |

---

## Tool Requests

Any agent can request a new tool by writing to `tools/requests/`:

```python
from iterare.tools.tool_request import submit_tool_request
submit_tool_request(
    name="my_tool",
    purpose="one sentence",
    why_existing_insufficient="specific gap",
    inputs="what it takes",
    outputs="what it returns",
    scope="narrow",
    requester="worker-id",
    task_context="task-id",
)
```

Requests are reviewed by the Master (or Interface Agent). Approved tools get built in `code/tools/` and registered in `tools/built/`.

---

## Full Structure Doc

`docs/structure.md`
