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

Goal: engineer discrete token prefixes that reliably steer LLM behavior across many suffix prompts without stating intent. Core challenge: closing the soft→discrete projection gap.

Current SOTA: **CE = 0.686** (Exp 16 λ=0, float32 sims — preliminary)

| Exp | Description | Status |
|-----|-------------|--------|
| 1–14 | Baseline through alternating ST+HotFlip | ✅ Complete |
| 15 | ST + cosine LR annealing + best-prefix tracking | ✅ Complete (HF CE=0.738, worse than Exp11) |
| 16 | ST + Voronoi margin regularization (3 λ values) | 🔄 Running — GPU 1 |
| 17 | Multi-seed (5×) ST+anneal+best-prefix, TOPK=50 | 🔄 Running — GPU 0 |
| 18 | TOPK escalation (50→100→200) from Exp11 best | 📋 Queued — GPU 1 after Exp16 |
| 19 | ST + cosine annealing + best-prefix, PREFIX_LEN=16 | 📋 Queued — GPU 0 after Exp17 |
| 20 | Multi-seed (seeds 5-9) fp32-sims ST + TOPK=50 | 📋 Queued — GPU 1 after Exp18 |

Key findings so far: ST estimator is the dominant improvement (CE 1.398→0.762 proj, 0.689 HF). Cosine LR annealing hurts. Basin quality matters more than projection CE. Float32 vs bfloat16 sims reach different Voronoi cells even at the same seed.
