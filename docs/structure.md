# Iterare: Agent-Native Platform Structure
*Version 1.0*

---

## Overview

Iterare is an agent-native public benefit AI-run platform focused on applied AI research and software development. It exists as a vehicle for the lead developer to leverage agentic capabilities to explore ideas, learn, and test theories — with outputs oriented toward measurable human benefit.

---

## Core Tenets

1. **Human-centric research**: Research focuses on improving the lives of as many humans as possible, prioritizing groups over individuals. Success is defined through measurable human-centered flourishing outcomes.
2. **Agent-centric execution**: Tasks are controlled by master planners that distribute work to agents. Each agent handles one thing at a time.
3. **Transparency and traceability**: All reasoning and justification is stored as README files, backed by research and cited sources. Every factor in a reasoning chain is documented and traceable.
4. **Signal over noise**: Agents produce only what is needed to complete the task. No speculative artifacts, no excessive logging, no extraneous comments. Thoroughness is measured by output quality, not output volume.
5. **Ethical side-constraints**: Hard gates against harm, deception, coercion, privacy abuse, and unfair treatment.

---

## Technical Foundation

### Agent Framework: LangGraph (MIT, free)

LangGraph is the orchestration layer for all agents. Selected for:
- **Native hierarchical graphs**: Master → Manager → Worker maps directly to directed graph model with sub-graphs and conditional routing
- **Production checkpointing**: PostgreSQL checkpointer saves complete state at every node execution; overnight tasks survive crashes and resume exactly where they left off
- **Built-in approval gates**: Interrupt system natively pauses at decision points, awaits human input, then resumes
- **Durable audit trail**: Every checkpoint is an immutable execution record
- **MCP support**: Custom tools integrate via Model Context Protocol

### Database
- **PostgreSQL**: LangGraph checkpointer backend; durable task state
- **LanceDB** (Apache 2.0, local, free): Vector store for agent-native semantic retrieval; indexed from markdown, never written to directly

### Systems of Record: Dual-Store

| Store | Format | Purpose | Who writes |
|-------|--------|---------|------------|
| Markdown files | `.md` | Human-readable source of truth | Masters only |
| LanceDB | Vector embeddings | Agent-native semantic retrieval | Re-indexed from markdown only |

**Rule**: Markdown is authoritative. If the two stores diverge, markdown wins. Re-indexing is triggered after every Master write.

**Write escalation**: Workers produce outputs → Managers review and request updates → Masters write to markdown → LanceDB re-indexes.

---

## Directory Structure

```
iterare/
├── code/           — Finished, working code organized by project
│   └── tools/      — Finished tool code
├── tasks/          — Active and completed task trees (README hierarchy + in-progress work)
├── archive/        — Cold-stored sealed task logs
├── templates/      — Agent guideline document templates (git-versioned; evolve over time)
├── tools/
│   ├── built/      — Tool registry; references finished tool code in code/tools/
│   └── requests/   — Tool request queue (YAML files written by agents, polled by Masters)
├── logs/           — Structured hot logs (JSON/YAML, per-task, sealed on completion)
└── docs/           — System documentation
```

### Code promotion path

In-progress and experimental code lives in `tasks/`. When code is finished and working, it is promoted:
- General code → `code/<project>/`
- Tools → `code/tools/<tool-name>/` and registered in `tools/built/`

Nothing lives in `code/` that isn't finished and functional.

---

## Agent Hierarchy

```
Lead Developer (human)
│
Interface Agent     — Sole direct interface for the lead developer. Persistent across sessions;
│                     can be cleared, saved, or restored. Loads context from LanceDB on session
│                     start. Has full Manager capabilities but does NOT spin up agents without
│                     asking first, unless it determines delegation is most efficient.
│                     Proposes via 1-3 line summary; full spec on request. Can override any
│                     Master decision with written justification.
│
Master Agent(s)     — Plans high-level tasks. Spins up agents via fully-specced guideline docs.
│                     Can spawn additional Master agents. Receives initial prompt from the lead
│                     developer. Sole writer to systems of record. Monitors Managers and
│                     intervenes when correction is needed. Runs recurring cross-task evals
│                     (cadence defined per project before execution begins).
│
├── Manager Agent   — Breaks high-level tasks into concrete steps; spins up Workers. Monitors
│   │                 Workers and intervenes when correction is needed or a Worker is stuck.
│   │                 Handles failure triage (retry, reinforce, replace). Writes summary README
│   │                 on task close. May REQUEST system-of-record updates to the Master.
│   │
│   ├── Worker      — Executes one concrete step. May communicate laterally with other Workers
│   │                 but must notify its Manager. On failure: logs output, notifies Manager,
│   │                 attempts self-fix. If unable to fix, escalates. Owns its task README.
│   ├── Worker 2
│   └── Worker 3
│
└── Manager 2 ...
```

---

## Control & Override Rules

| Agent | Authority |
|-------|-----------|
| Interface Agent | Highest. Can override any Master decision with written justification. |
| Master | Directs Managers; may spawn peer Masters. Cannot override Interface Agent. |
| Manager | Directs Workers; handles failure triage. Cannot write to systems of record. |
| Worker | Single-purpose execution. Lateral communication permitted; must notify Manager. |

### Human Approval Gates

The following pause LangGraph execution and require explicit lead developer approval:
- Irreversible actions
- External-impact actions
- Financial actions
- Legal actions
- Security-sensitive actions

---

## Memory & Session Policy

### Interface Agent
Persistent by default. Lead developer can clear, save, or restore sessions. On start, loads relevant context from LanceDB.

### Task-level memory
Decided per task based on agent/framework strengths:
- **Fresh context** — when prior state would introduce noise or drift
- **Persistent** — when historical context meaningfully improves accuracy or continuity

Decision and rationale documented in the task README.

---

## Task Model

Tasks are long-running; some run overnight without supervision. Before any task tree begins:
- Stopping conditions must be defined
- Failure modes and escalation paths must be defined
- Master-level eval cadence must be defined

LangGraph (PostgreSQL checkpointer) ensures crash recovery and full execution history.

---

## Logging Standard

- **Structured**: JSON or YAML — queryable without reading every line
- **Scoped per task**: no single continuously-growing file
- **Sealed on completion**: no writes after a task closes
- **Summarized then archived**: summary stays hot in `logs/`; full log moves to `archive/`
- **Written at decision and handoff points only** — not continuously

LangGraph checkpoints are the primary durable audit trail. Application logs supplement; they do not duplicate.

---

## Documentation Standard (READMEs)

### Task directory structure

```
tasks/
└── task-001/
    ├── README.md              ← Master summary; includes Master-level eval results
    ├── manager-1/
    │   ├── README.md          ← Manager summary (written on task close); includes eval results; links to Worker READMEs
    │   ├── worker-1/
    │   │   └── README.md      ← Worker reasoning, decisions, citations
    │   └── worker-2/
    │       └── README.md
    └── manager-2/
        └── ...
```

### Ownership

- **Workers** write their README as they execute. The deciding agent documents its own reasoning.
- **Managers** write a summary README on task close, referencing Worker READMEs. Primary review layer. Includes eval results.
- **Masters** write a top-level summary for multi-manager task trees. Includes Master eval results.

### Evidence quality levels

All claims in READMEs carry a confidence marker:

| Level | Meaning |
|-------|---------|
| `[H]` | High — peer-reviewed or multiple independent credible sources |
| `[M]` | Medium — single credible source, or strong indirect evidence |
| `[L]` | Low — reasoning or inference; no direct citation |

---

## Evaluation Loops

Workers escalate; they do not self-evaluate. Evals run at Manager and Master levels only.

### Manager-level (per task)

Monitors Workers continuously; intervenes only when needed. Checks:
- Is the Worker repeating steps without progress? (loop detection)
- Is output converging or diverging?
- Has the Worker exceeded its step or time budget?
- Are outputs meeting quality standards?

Monitoring is both **proactive** (watches step counts; intervenes before escalation) and **reactive** (Worker escalates after N failed attempts) — applied based on task criticality.

### Master-level (recurring, cross-task)

Monitors Managers continuously; intervenes when needed. Checks:
- Are Managers producing consistent, high-quality summaries?
- Are task outcomes matching stated goals?
- Are systemic failure patterns emerging?

Cadence defined per project before execution begins.

---

## Tool Usage Policy

1. **Existing free tools**: Manager approves; Master decides if Iterare policy alignment is in question.
2. **Non-existent or bespoke tools**: Master may assign a Manager to build it.
3. **Tool request**: A formal tool request is available to all agents — the primary autonomous path for initiating new tool builds.
4. **Paid tools**: Not available at this time.

### Tool request format

Agents write a YAML file to `tools/requests/`. Masters poll the queue and approve, reject, or assign a build.

```yaml
tool_request:
  name:                        # proposed tool name
  purpose:                     # what it does, one sentence
  why_existing_insufficient:   # specific gap
  inputs:                      # what it takes
  outputs:                     # what it returns
  scope:                       # narrow | moderate | broad
  requester:                   # agent id
  task_context:                # task id this originated from
```

---

## Agent Guideline Document Specification

Every spawned agent receives a guideline document generated from a template in `templates/`. Templates are git-versioned — no duplicate files; history is tracked via git. Masters and Managers update templates when they identify gaps or improvements.

### Required fields

```yaml
agent:
  role:                        # what this agent is
  task:                        # specific work for this instantiation
  tools_available:             # list; always includes tool-request tool
  input_schema:                # format and source of inputs
  output_schema:               # format and destination of outputs
  stopping_conditions:         # list
  failure_escalation_path:     # what to do when stuck or producing bad output
  memory_policy:               # fresh | persistent | hybrid — with rationale
  readme_requirement:          # what to document and where
```

### Interface Agent proposal format

- **Initial**: 1-3 lines — what the agent would do and why
- **Full spec on request**: scope, permissions needed, goals, stopping conditions

---

## Ethical Frame

**Primary objective**: Measurable human-centered flourishing outcomes, prioritizing groups over individuals.

**Hard gates**:
- No harm
- No deception
- No coercion
- No privacy abuse
- No unfair treatment

---

## Research Domain

Research directions are supplied by the lead developer. Iterare does not set its own agenda independently.

---

## What Iterare Is Not

- A headcount replacement system
- A fully autonomous research organization
- A benchmark-maximization engine
- A human organizational structure applied to agents
- A system that generates volume to signal effort
