"""Evaluate and compare results across conditions.

Usage
-----
python tasks/anti-steer-001/eval.py results/run.json
python tasks/anti-steer-001/eval.py results/run.json --judge --agent claude
python tasks/anti-steer-001/eval.py results/run.json --judge --agent codex
python tasks/anti-steer-001/eval.py results/run.json --judge --agent codex --save
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

# Repo root on sys.path so we can import code.tools.*
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def load(path: str) -> list[dict]:
    return json.loads(Path(path).read_text())


def aggregate(results: list[dict]) -> dict[str, dict[str, list[float]]]:
    stats: dict[str, dict[str, list[float]]] = {}
    for task in results:
        for cond, data in task["conditions"].items():
            if cond not in stats:
                stats[cond] = {"mean_pairwise_sim": [], "mean_max_to_prior": []}
            stats[cond]["mean_pairwise_sim"].append(data["diversity"]["mean_pairwise_sim"])
            stats[cond]["mean_max_to_prior"].append(data["diversity"]["mean_max_to_prior"])
    return stats


def print_table(stats: dict[str, dict[str, list[float]]]):
    import statistics

    conds = list(stats.keys())
    header = f"{'condition':<20} {'mean_pair_sim':>14} {'mean_max_prior':>15} {'n_tasks':>8}"
    print(header)
    print("-" * len(header))
    for cond in conds:
        s = stats[cond]
        n = len(s["mean_pairwise_sim"])
        mp = statistics.mean(s["mean_pairwise_sim"])
        mx = statistics.mean(s["mean_max_to_prior"])
        print(f"{cond:<20} {mp:>14.4f} {mx:>15.4f} {n:>8}")


# ── judge backends ────────────────────────────────────────────────────────────

JUDGE_PROMPT = (
    "Issue: {task}\n\nIdea: {idea}\n\n"
    "Rate this idea on a scale from 1 (useless/incoherent) to 5 "
    "(concrete, plausible, actionable). Reply with only the integer."
)


def _parse_score(text: str) -> float:
    m = re.search(r"\b([1-5])\b", text.strip())
    return float(m.group(1)) if m else 3.0


def _judge_claude(task: str, ideas: list[str]) -> list[float]:
    import anthropic
    client = anthropic.Anthropic()
    scores = []
    for idea in ideas:
        msg = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=50,
            messages=[{"role": "user", "content": JUDGE_PROMPT.format(task=task, idea=idea)}],
        )
        scores.append(_parse_score(msg.content[0].text))
    return scores


def _judge_claude_cli(task: str, ideas: list[str]) -> list[float]:
    import subprocess
    scores = []
    for idea in ideas:
        prompt = JUDGE_PROMPT.format(task=task, idea=idea)
        result = subprocess.run(
            ["claude", "-p", prompt],
            capture_output=True, text=True, timeout=30,
        )
        scores.append(_parse_score(result.stdout))
    return scores


def _judge_codex(task: str, ideas: list[str]) -> list[float]:
    from code.tools.codex.runner import CodexRunner
    runner = CodexRunner(default_sandbox="read-only", timeout=60)
    scores = []
    for idea in ideas:
        response = runner.exec(JUDGE_PROMPT.format(task=task, idea=idea), sandbox="read-only")
        scores.append(_parse_score(response))
    return scores


def judge_ideas(task: str, ideas: list[str], agent: str) -> list[float]:
    if agent == "claude":
        return _judge_claude(task, ideas)
    elif agent == "claude-cli":
        return _judge_claude_cli(task, ideas)
    elif agent == "codex":
        return _judge_codex(task, ideas)
    raise ValueError(f"Unknown agent: {agent!r}. Use 'claude', 'claude-cli', or 'codex'.")


def run_judge(results: list[dict], agent: str) -> list[dict]:
    for task_result in results:
        task = task_result["task"]
        for cond, data in task_result["conditions"].items():
            if "usefulness" not in data:
                scores = judge_ideas(task, data["ideas"], agent)
                data["usefulness"] = {"scores": scores, "mean": sum(scores) / len(scores)}
                print(
                    f"  task={task_result['task_idx']} cond={cond} "
                    f"usefulness={data['usefulness']['mean']:.2f}"
                )
    return results


def print_usefulness_table(results: list[dict]):
    import statistics

    cond_scores: dict[str, list[float]] = {}
    for task_result in results:
        for cond, data in task_result["conditions"].items():
            if "usefulness" in data:
                cond_scores.setdefault(cond, []).append(data["usefulness"]["mean"])

    if not cond_scores:
        print("No usefulness scores available. Run with --judge --agent claude|codex.")
        return

    print(f"\n{'condition':<20} {'mean_usefulness':>16}")
    print("-" * 38)
    for cond, scores in cond_scores.items():
        print(f"{cond:<20} {statistics.mean(scores):>16.3f}")


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", help="Path to results JSON")
    parser.add_argument("--judge", action="store_true",
                        help="Score usefulness via an LLM judge")
    parser.add_argument("--agent", choices=["claude", "claude-cli", "codex"], default="claude-cli",
                        help="Judge backend: 'claude' (Anthropic API), 'claude-cli' (claude CLI), or 'codex'")
    parser.add_argument("--save", action="store_true",
                        help="Save judge scores back to the results file")
    args = parser.parse_args()

    results = load(args.results)
    print(f"Loaded {len(results)} tasks\n")

    stats = aggregate(results)
    print("=== Semantic Diversity ===")
    print_table(stats)

    if args.judge:
        print(f"\nRunning usefulness judge (agent={args.agent})...")
        results = run_judge(results, args.agent)
        if args.save:
            Path(args.results).write_text(json.dumps(results, indent=2))
            print(f"Saved updated results to {args.results}")

    print_usefulness_table(results)

    # Per-step max-similarity
    print("\n=== Mean max-to-prior sim by step ===")
    cond_step: dict[str, dict[int, list[float]]] = {}
    for task_result in results:
        for cond, data in task_result["conditions"].items():
            ideas = data["ideas"]
            if len(ideas) < 2:
                continue
            embs = _embed_cached(ideas)
            import numpy as np
            for i in range(1, len(ideas)):
                prior_sims = [float(np.dot(embs[i], embs[j])) for j in range(i)]
                cond_step.setdefault(cond, {}).setdefault(i, []).append(max(prior_sims))

    import statistics
    steps = sorted({k for d in cond_step.values() for k in d})
    header = f"{'step':>5}" + "".join(f"  {c:<18}" for c in cond_step)
    print(header)
    for step in steps:
        row = f"{step:>5}"
        for cond in cond_step:
            vals = cond_step[cond].get(step, [])
            m = statistics.mean(vals) if vals else float("nan")
            row += f"  {m:<18.4f}"
        print(row)


_embed_cache: dict[str, object] = {}

def _embed_cached(texts: list[str]):
    from sentence_transformers import SentenceTransformer
    if "model" not in _embed_cache:
        _embed_cache["model"] = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embed_cache["model"].encode(texts, normalize_embeddings=True)


if __name__ == "__main__":
    main()
