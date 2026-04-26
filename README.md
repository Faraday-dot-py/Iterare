# Iterare

Agent-native applied AI research platform. Runs through Claude Code — no Anthropic API key required.

## Architecture

```
Lead Developer
└── Interface Agent  ← this Claude Code session
    └── Orchestrator  (spawned via Agent tool for multi-manager tasks)
        └── Manager Agent  (spawned by Orchestrator via Agent tool)
            └── Worker Agent  (spawned by Manager via Agent tool)
```

The Interface Agent often acts as Orchestrator directly. A dedicated Orchestrator
(Master Agent) is spawned only for large or overnight tasks.

Every significant session runs against a **file-native control plane**:
`run.yaml` (objective + budgets + stop rules), `events.jsonl` (append-only event log),
`resume.md` (hydration packet for the next session), and `checkpoints/`.

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

Talk to Claude Code. Pitch your research idea. The Interface Agent handles it
directly or proposes spawning an Orchestrator — you approve before anything runs.

---

## CLI

The CLI inspects state and manages approvals. It does not run agents.

```bash
iterare tasks                    # list all tasks
iterare tasks <task_id>          # show task state + log
iterare runs <task_id>           # show run control doc + event log
iterare approvals                # list all pending approvals (all tasks)
iterare approvals <task_id>      # list approvals for a task
iterare approvals approve <task_id> <apr_id>
iterare approvals reject  <task_id> <apr_id> "reason"
iterare requests                 # list pending tool requests
iterare requests approve <req_id>
iterare requests reject  <req_id> "reason"
```

---

## Directory Structure

```
iterare/
├── code/           — Finished, working code (organized by project)
│   └── tools/      — Finished tool implementations
├── tasks/          — Task trees + run control plane
│   └── <task-id>/
│       ├── run.yaml        ← objective, budgets, stop rules, checkpoint pointer
│       ├── events.jsonl    ← append-only event log
│       ├── resume.md       ← session hydration packet
│       ├── provenance.jsonl ← claim/artifact lineage
│       ├── state.yaml      ← task metadata
│       ├── README.md       ← Orchestrator summary (written on completion)
│       ├── checkpoints/
│       ├── approvals/
│       ├── leases/
│       └── manager-N/
│           ├── README.md
│           └── worker-N/
│               └── README.md
├── archive/        — Completed tasks: manifest + frozen control plane
│   └── <task-id>/
│       ├── manifest.yaml   ← final status, metrics, artifacts
│       ├── run.yaml, events.jsonl, provenance.jsonl, checkpoints/
├── templates/      — Agent prompts (git-versioned)
├── tools/
│   ├── built/      — Registered tools (YAML metadata)
│   └── requests/   — Tool request queue
├── docs/           — System documentation (structure.md is the full spec)
└── src/iterare/    — Python utilities
```

**Code promotion**: experimental work lives in `tasks/`. Finished code moves to `code/`.
Tools register in `tools/built/`.

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

Tool lifecycle: `request → prototype → evaluate → approve → promote → deprecate`.
Approved tools get built in `code/tools/` and registered in `tools/built/`.

Review the queue: `iterare requests`

---

## Full Structure Doc

`docs/structure.md`

---

## Active Tasks

### [golf-001](tasks/golf-001/README.md) — OpenAI Parameter Golf

**Status: Complete**

Best result: **1.11316 BPB** (sliding-window, stride=64) — beats competition SOTA by −0.0015.

Artifact: `tasks/golf-001/manager-5/worker-1/results/artifact_seed314.ptz` (15.76 MB, ready to submit).

Architecture: 11-layer GQA transformer, GPTQ int6 quant, AR self-gen calibration, XSA all layers, BigramHash 3072.

---

### [steer-001](tasks/steer-001/README.md) — Steering Prefix Research (MATS-10.0)

**Status: In progress** | Compute: TIDE (2× NVIDIA A100 80GB)

Goal: engineer discrete token prefixes that reliably steer LLM behavior across many suffix prompts without stating intent.

Current SOTA: **CE = 0.59892** (Exp52/53 — tierra swap confirmed)

Key findings: ST estimator dominant improvement. tierra token uniquely effective as swap target. HotFlip is a local optimizer; warm restart from SOTA is the only escape.
