# Iterare: Agent-Native Platform Structure
*Version 2.0 — 3-role model, file-native control plane*

---

## Overview

Iterare is an agent-native applied AI research platform. It exists as a vehicle
for the lead developer to leverage agentic capabilities to explore ideas, learn,
and test theories — with outputs oriented toward measurable human benefit.

---

## Core Tenets

1. **Human-centric research**: Research focuses on improving the lives of as many
   humans as possible. Success is defined through measurable, human-centered outcomes.
2. **Agent-centric execution**: Orchestrators distribute work to workers. Each worker
   handles one concrete step at a time.
3. **Transparency and traceability**: All reasoning is stored in README files with
   evidence grades and citations. Claims are traceable via provenance records.
4. **Signal over noise**: Agents produce only what is needed. Thoroughness is measured
   by output quality, not volume.
5. **Durable work across sessions**: Every significant run leaves a file-based control
   plane (run.yaml, events.jsonl, resume.md, checkpoints) so sessions can resume
   without reconstructing intent from conversation history.
6. **Ethical side-constraints**: Hard gates against harm, deception, coercion, privacy
   abuse, and unfair treatment.

---

## Agent Roles

Three roles. No more.

```
Lead Developer (human)
│
Interface Agent     — Sole direct interface for the lead developer. This IS the
│                     Claude Code session. Handles tasks directly or proposes
│                     spawning an Orchestrator for multi-manager scope.
│                     Retains override authority over all spawned agents.
│
Orchestrator        — Plans and delegates. Owns the run lifecycle: creates run.yaml,
│  (Master Agent)     issues checkpoints, submits approvals, archives on completion.
│                     Sole writer to the system of record (outside tasks/).
│                     Spawned by Interface Agent for large or overnight tasks.
│
Manager Agent       — Breaks orchestrator sub-tasks into atomic worker steps.
│                     Owns a leased subtree. Logs events, acquires write lease.
│                     Returns write requests to Orchestrator for out-of-scope writes.
│
Worker Agent        — Executes one concrete step. Owns a leased scratch directory.
                      Records provenance for significant findings and artifacts.
                      On failure: log, try different approach, escalate.
```

**The Interface Agent often acts as Orchestrator directly.** A dedicated Master
agent is spawned only when task scope warrants it: multi-manager work, overnight
unsupervised execution, or when isolation from the main session matters.

---

## File-Native Control Plane

Every significant execution session runs against a control document and event log.
These are the files that make long-running work resumable without prompt archaeology.

### Per-task control files

```
tasks/<task-id>/
  run.yaml          — objective, status, budgets, stop rules, write scope,
                      active delegates, checkpoint pointer, open approvals
  events.jsonl      — append-only event log (decision + side-effect boundaries only)
  resume.md         — human-readable hydration packet for the next session
  provenance.jsonl  — PROV-lite: entity → activity → agent → source → grade
  checkpoints/      — state snapshots at durable boundaries
  approvals/        — pending and resolved approval records
  leases/           — active write leases per agent
  state.yaml        — task-level metadata (created, status, description)
```

### Checkpoint policy

Checkpoint at **decision and side-effect boundaries**, not every model turn:
- Plan finalized
- Delegation issued
- Tool result accepted
- Approval requested / resolved
- Artifact written
- Evidence updated
- Stop rule evaluated

### Resume policy

`resume.md` must let the next session reconstruct intent without reading
conversation history. It contains: objective, last checkpoint, stop rules,
write scope, and a human-readable summary of progress.

---

## Directory Structure

```
iterare/
├── code/           — Finished, working code (organized by project)
│   └── tools/      — Finished tool implementations
├── tasks/          — Task trees + run control plane
│   └── <task-id>/
│       ├── run.yaml
│       ├── events.jsonl
│       ├── resume.md
│       ├── provenance.jsonl
│       ├── state.yaml
│       ├── README.md       ← Orchestrator summary (written on completion)
│       ├── checkpoints/
│       ├── approvals/
│       ├── leases/
│       ├── shared/         ← Lateral worker communication
│       ├── manager-1/
│       │   ├── README.md   ← Manager summary
│       │   └── worker-1/
│       │       └── README.md  ← Worker reasoning + results
│       └── manager-2/ ...
├── archive/        — Completed tasks: manifest + frozen artifacts
│   └── <task-id>/
│       ├── manifest.yaml   ← Final status, metrics, artifact list
│       ├── run.yaml
│       ├── events.jsonl
│       ├── provenance.jsonl
│       └── checkpoints/
├── templates/      — Agent prompts (git-versioned; evolve over time)
├── tools/
│   ├── built/      — Registered tools (YAML metadata → code/tools/)
│   └── requests/   — Tool request queue
├── logs/           — Legacy JSONL logs (use events.jsonl for new tasks)
├── docs/           — System documentation
└── src/iterare/    — Python utilities
    ├── utils/      — run.py, events.py, approvals.py, provenance.py,
    │                 leases.py, archive.py, task.py, log.py
    └── tools/      — file_tools.py, tool_request.py, tide_runner.py, ...
```

**Code promotion**: experimental work lives in `tasks/`. Finished code moves to
`code/`. Tools also register in `tools/built/`.

---

## Write Authority

Write authority is **role-bounded and lease-tracked**.

| Role | Can write to |
|------|-------------|
| Worker | `tasks/<task-id>/manager-N/worker-N/` (own leased directory) |
| Manager | `tasks/<task-id>/manager-N/` (own leased subtree) |
| Orchestrator | anywhere within `tasks/<task-id>/`; system-of-record with approval |
| Interface Agent | full write authority; approvals are for audit, not enforcement |

Workers and Managers that need to write outside their lease submit a write request
to the Orchestrator in their return payload. The Orchestrator reviews and executes.

Leases are file-backed (`tasks/<task-id>/leases/<agent-id>.yaml`) and tracked for
audit. Enforcement is by convention; violations are detectable post-hoc.

---

## Approval Queue

Approvals are **risk-tiered**, not omnipresent.

| Tier | When | Examples |
|------|------|---------|
| low | Auto-proceed inside write scope | File writes within lease |
| medium | Pause for human review | System-of-record writes, tool promotions |
| high | Pause + explicit confirmation | Destructive ops, external APIs, financial |

Approval records live at `tasks/<task-id>/approvals/`. Inspect and resolve via:
```bash
iterare approvals                              # list all pending
iterare approvals <task_id>                    # list for a task
iterare approvals approve <task_id> <apr_id>   # approve
iterare approvals reject  <task_id> <apr_id> "reason"
```

---

## Stop Rules

Every significant run should define explicit stop conditions. Examples:
- `"CE below target threshold (< 0.55)"`
- `"Two consecutive loops with no meaningful evidence delta"`
- `"GPU hours exhausted (> 10h)"`

Stop rules live in `run.yaml` and are checked by the Orchestrator after each
manager completes. When a rule fires, log the event, close the run, return.

---

## Tool Registry

Tools follow an explicit lifecycle: **request → prototype → evaluate → approve → promote → deprecate**.

```
tools/
  requests/   — pending YAML tool requests (written by agents)
  built/      — promoted tools (YAML metadata + reference to code/tools/)
```

Each tool record in `tools/built/` contains: name, version, status, description,
code_path, capabilities, iterare_usage, hardware requirements, configuration.

Use `iterare requests` to review the queue.

---

## Context Hydration

Session startup should be staged:

1. **Boot pack**: task charter (`run.yaml`), latest `resume.md`, active stop rules,
   open approvals, last checkpoint, latest manager summaries.
2. **Task-local artifact pack**: relevant READMEs, evidence tables, tool docs,
   unresolved questions.
3. **Corpus retrieval** (only if corpus exceeds context): hybrid semantic + keyword
   search with reranking. Not the default.

Start from the resume.md, not from raw conversation history.

---

## Evaluation Loops

Workers escalate; they do not self-evaluate. Evals run at Manager and Orchestrator
levels only.

### Manager-level (per task)
- Is the Worker repeating steps without progress? (loop detection)
- Is output converging or diverging?
- Has the Worker exceeded its step or time budget?
- Are outputs meeting quality standards?

### Orchestrator-level (recurring, cross-manager)
- Are Managers producing consistent, high-quality summaries?
- Are task outcomes matching stated goals?
- Are systemic failure patterns emerging?

---

## Documentation Standard (READMEs)

```
tasks/<task-id>/
  README.md              ← Orchestrator summary; metrics; links to manager READMEs
  manager-N/
    README.md            ← Manager summary; worker results with ✓/⚠/✗; eval results
    worker-N/
      README.md          ← Worker reasoning, decisions, citations
```

### Ownership
- Workers write their README as they execute.
- Managers write a summary README on close.
- Orchestrators write the top-level README for the task.

### Evidence quality levels

| Level | Meaning |
|-------|---------|
| `[H]` | High — peer-reviewed or multiple independent credible sources |
| `[M]` | Medium — single credible source, or strong indirect evidence |
| `[L]` | Low — reasoning or inference; no direct citation |

---

## Archive Lifecycle

A completed task is not done until archived. Terminal steps:

1. Orchestrator writes `tasks/<task-id>/README.md` (final summary)
2. `write_manifest(task_id, final_status, summary, metrics, artifacts)`
3. `archive_task(task_id)` — copies control plane files to `archive/<task-id>/`
4. `seal_log(task_id)` — makes the log read-only
5. Return to Interface Agent

The manifest is the permanent record: status, metrics, artifact inventory.

---

## Logging

- `tasks/<task-id>/events.jsonl` — primary event log (durable, append-only)
- `logs/<task-id>.jsonl` — legacy structured log (still supported; use events.jsonl for new tasks)
- Events written at decision/handoff boundaries only — not continuously

---

## Ethical Frame

**Primary objective**: Measurable human-centered flourishing outcomes, prioritizing
groups over individuals.

**Hard gates**:
- No harm
- No deception
- No coercion
- No privacy abuse
- No unfair treatment

---

## What Iterare Is Not

- A headcount replacement system
- A fully autonomous research organization
- A benchmark-maximization engine
- A human organizational structure applied to agents
- A system that generates volume to signal effort
- A framework requiring LangGraph, PostgreSQL, or a heavyweight runtime
