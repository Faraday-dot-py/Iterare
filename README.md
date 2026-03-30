# Iterare

Agent-native applied AI research platform.

## Setup

### 1. Prerequisites

- Python 3.11+
- PostgreSQL (for durable task checkpointing)
- Anthropic API key

**Install uv** (recommended):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Quick PostgreSQL setup (local)**:
```bash
sudo apt install postgresql
sudo -u postgres psql -c "CREATE USER iterare WITH PASSWORD 'iterare';"
sudo -u postgres psql -c "CREATE DATABASE iterare OWNER iterare;"
```

### 2. Install dependencies

```bash
cd /path/to/iterare
uv sync        # or: pip install -e .
```

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env — set ANTHROPIC_API_KEY and ITERARE_DB_URL
```

### 4. Run

```bash
iterare
# or
python -m iterare.cli
```

---

## Usage

```
> Your research idea here
```

The Interface Agent will either handle your request directly or propose spawning
a Master Agent to plan and execute it. You approve before anything runs.

**Session management**:
```bash
iterare                   # start or resume saved session
iterare --thread <id>     # resume a specific thread
iterare --clear           # start a fresh session
```

---

## Directory Structure

```
iterare/
├── code/           — Finished, working code (organized by project)
│   └── tools/      — Finished tool implementations
├── tasks/          — Active and completed task trees
│   └── task-<id>/  — Per-task: Master README + Manager/Worker READMEs
├── archive/        — Cold-stored sealed task logs
├── templates/      — Agent guideline templates (git-versioned)
├── tools/
│   ├── built/      — Tool registry
│   └── requests/   — Tool request queue (YAML; polled by Masters)
├── logs/           — Structured hot logs (JSON/YAML, per-task)
├── docs/           — System documentation
└── src/iterare/    — Platform source code
```

**Code promotion**: in-progress and experimental work lives in `tasks/`.
When code is finished, it moves to `code/`. Tools also register in `tools/built/`.

---

## Agent Hierarchy

```
Lead Developer
└── Interface Agent  (you talk here)
    └── Master Agent  (plans, owns system of record)
        └── Manager Agent  (coordinates workers)
            └── Worker Agent  (executes single steps)
```

Full structure: `docs/structure.md`

---

## Evidence Quality

All README claims carry a quality marker:

| Level | Meaning |
|-------|---------|
| `[H]` | Peer-reviewed or multiple independent sources |
| `[M]` | Single credible source |
| `[L]` | Reasoning/inference, no direct citation |

---

## Adding Tools

Any agent can submit a tool request:
```python
submit_tool_request(
    name="my_tool",
    purpose="...",
    why_existing_insufficient="...",
    inputs="...",
    outputs="...",
    scope="narrow",
)
```

Requests land in `tools/requests/` as YAML files. The Master reviews and
assigns builds. Finished tools go in `code/tools/` and register in `tools/built/`.
