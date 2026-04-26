"""Generate a self-contained HTML report from anti-steer-001 results.

Usage
-----
python tasks/anti-steer-001/visualize.py results/run01.json
python tasks/anti-steer-001/visualize.py results/run01.json results/run02.json --out report.html
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


# ── data helpers ──────────────────────────────────────────────────────────────

def load_runs(paths: list[str]) -> list[dict]:
    runs = []
    for p in paths:
        data = json.loads(Path(p).read_text())
        runs.append({"path": p, "tasks": data})
    return runs


def condition_stats(tasks: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for task in tasks:
        for cond, data in task["conditions"].items():
            s = stats.setdefault(cond, {"mean_pairwise_sim": [], "mean_max_to_prior": [], "usefulness": []})
            s["mean_pairwise_sim"].append(data["diversity"]["mean_pairwise_sim"])
            s["mean_max_to_prior"].append(data["diversity"]["mean_max_to_prior"])
            if "usefulness" in data:
                s["usefulness"].append(data["usefulness"]["mean"])
    return stats


def sim_color(val: float) -> str:
    """Red (high similarity = bad) → green (low similarity = good). Range 0–1."""
    val = max(0.0, min(1.0, val))
    # 0.0 = green, 0.5 = yellow, 1.0 = red
    if val < 0.5:
        r = int(val * 2 * 220)
        g = 180
    else:
        r = 220
        g = int((1 - val) * 2 * 180)
    return f"rgb({r},{g},60)"


def usefulness_color(val: float) -> str:
    """1=red, 3=yellow, 5=green."""
    t = (val - 1) / 4.0
    if t < 0.5:
        r = 220
        g = int(t * 2 * 180)
    else:
        r = int((1 - t) * 2 * 220)
        g = 180
    return f"rgb({r},{g},60)"


# ── HTML generation ───────────────────────────────────────────────────────────

_CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: system-ui, sans-serif; background: #0f1117; color: #e2e8f0; font-size: 14px; }
h1 { font-size: 1.4rem; font-weight: 600; }
h2 { font-size: 1.1rem; font-weight: 600; color: #94a3b8; margin-bottom: 8px; }
h3 { font-size: 0.9rem; font-weight: 600; color: #64748b; text-transform: uppercase; letter-spacing: 0.05em; }
.page { max-width: 1400px; margin: 0 auto; padding: 32px 24px; }
.header { margin-bottom: 32px; }
.header p { color: #64748b; margin-top: 4px; font-size: 0.85rem; }
.run-section { margin-bottom: 48px; }
.run-label { font-size: 0.8rem; color: #475569; font-family: monospace; margin-bottom: 12px; }

/* Summary table */
.summary-table { width: 100%; border-collapse: collapse; margin-bottom: 24px; }
.summary-table th { text-align: left; padding: 8px 12px; color: #64748b; font-weight: 500;
  font-size: 0.8rem; text-transform: uppercase; letter-spacing: 0.04em;
  border-bottom: 1px solid #1e293b; }
.summary-table td { padding: 8px 12px; border-bottom: 1px solid #1e293b; font-family: monospace; }
.summary-table tr:hover td { background: #1e293b; }
.cond-name { font-family: monospace; font-size: 0.85rem; color: #a5b4fc; }
.metric-cell { text-align: right; }
.color-pill { display: inline-block; padding: 2px 8px; border-radius: 4px;
  font-weight: 600; font-size: 0.85rem; color: #0f1117; }

/* Task cards */
.tasks-grid { display: flex; flex-direction: column; gap: 16px; }
.task-card { background: #1e293b; border-radius: 8px; overflow: hidden; }
.task-header { padding: 12px 16px; background: #162032; cursor: pointer;
  display: flex; align-items: center; gap: 12px; user-select: none; }
.task-header:hover { background: #1a2a40; }
.task-idx { font-size: 0.75rem; color: #475569; font-family: monospace; min-width: 24px; }
.task-text { flex: 1; font-size: 0.85rem; color: #cbd5e1; }
.task-chevron { color: #475569; transition: transform 0.2s; font-size: 0.8rem; }
.task-card.open .task-chevron { transform: rotate(90deg); }
.task-body { display: none; padding: 16px; }
.task-card.open .task-body { display: block; }

/* Condition columns */
.cond-grid { display: grid; gap: 12px; }
.cond-col { background: #0f1a27; border-radius: 6px; padding: 12px; }
.cond-header { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
.cond-pills { display: flex; gap: 6px; flex-wrap: wrap; }
.idea-list { display: flex; flex-direction: column; gap: 8px; }
.idea-item { display: flex; gap: 8px; }
.idea-num { font-size: 0.75rem; color: #475569; font-family: monospace; min-width: 18px; padding-top: 2px; }
.idea-text { font-size: 0.82rem; color: #94a3b8; line-height: 1.5; }

/* Responsive grid columns based on condition count */
.cond-grid-2 { grid-template-columns: repeat(2, 1fr); }
.cond-grid-3 { grid-template-columns: repeat(3, 1fr); }
.cond-grid-4 { grid-template-columns: repeat(4, 1fr); }
.cond-grid-5 { grid-template-columns: repeat(5, 1fr); }
@media (max-width: 900px) {
  .cond-grid-2, .cond-grid-3, .cond-grid-4, .cond-grid-5 { grid-template-columns: 1fr; }
}

/* Step chart */
.step-section { margin-top: 24px; }
.step-table { border-collapse: collapse; font-family: monospace; font-size: 0.82rem; }
.step-table th { padding: 6px 16px 6px 0; color: #475569; text-align: right; }
.step-table td { padding: 4px 16px 4px 0; text-align: right; }
"""

_JS = """
document.querySelectorAll('.task-header').forEach(h => {
  h.addEventListener('click', () => {
    h.closest('.task-card').classList.toggle('open');
  });
});
// open first task by default
const first = document.querySelector('.task-card');
if (first) first.classList.add('open');
"""


def _pill(val: float, color_fn, fmt=".3f") -> str:
    color = color_fn(val)
    return f'<span class="color-pill" style="background:{color}">{val:{fmt}}</span>'


def render_summary(stats: dict[str, dict]) -> str:
    has_usefulness = any(stats[c]["usefulness"] for c in stats)
    rows = []
    for cond, s in stats.items():
        mp = statistics.mean(s["mean_pairwise_sim"])
        mx = statistics.mean(s["mean_max_to_prior"])
        n = len(s["mean_pairwise_sim"])
        u_cell = ""
        if has_usefulness and s["usefulness"]:
            u = statistics.mean(s["usefulness"])
            u_cell = f'<td class="metric-cell">{_pill(u, usefulness_color, ".2f")}</td>'
        elif has_usefulness:
            u_cell = '<td class="metric-cell">—</td>'
        rows.append(
            f'<tr>'
            f'<td><span class="cond-name">{cond}</span></td>'
            f'<td class="metric-cell">{_pill(mp, sim_color)}</td>'
            f'<td class="metric-cell">{_pill(mx, sim_color)}</td>'
            f'<td class="metric-cell">{n}</td>'
            f'{u_cell}'
            f'</tr>'
        )
    u_th = '<th>usefulness</th>' if has_usefulness else ''
    header = f'<tr><th>condition</th><th style="text-align:right">mean pairwise sim ↓</th><th style="text-align:right">mean max-to-prior ↓</th><th style="text-align:right">tasks</th>{u_th}</tr>'
    return f'<table class="summary-table">{header}{"".join(rows)}</table>'


def render_step_table(tasks: list[dict]) -> str:
    import numpy as np

    def _embed_all():
        from sentence_transformers import SentenceTransformer
        st = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        cond_step: dict[str, dict[int, list[float]]] = {}
        for task in tasks:
            for cond, data in task["conditions"].items():
                ideas = data["ideas"]
                if len(ideas) < 2:
                    continue
                embs = st.encode(ideas, normalize_embeddings=True)
                for i in range(1, len(ideas)):
                    prior_sims = [float(np.dot(embs[i], embs[j])) for j in range(i)]
                    cond_step.setdefault(cond, {}).setdefault(i, []).append(max(prior_sims))
        return cond_step

    cond_step = _embed_all()
    if not cond_step:
        return ""

    conds = list(cond_step.keys())
    steps = sorted({k for d in cond_step.values() for k in d})
    th_cells = "".join(f'<th>{c}</th>' for c in conds)
    rows = [f"<tr><th>step</th>{th_cells}</tr>"]
    for step in steps:
        cells = []
        for cond in conds:
            vals = cond_step[cond].get(step, [])
            if vals:
                m = statistics.mean(vals)
                cells.append(f'<td>{_pill(m, sim_color)}</td>')
            else:
                cells.append('<td>—</td>')
        rows.append(f'<tr><td style="color:#64748b;padding-right:16px">idea {step+1}</td>{"".join(cells)}</tr>')

    return (
        '<div class="step-section">'
        '<h2>Max similarity to prior ideas by step</h2>'
        f'<table class="step-table">{"".join(rows)}</table>'
        '</div>'
    )


def render_tasks(tasks: list[dict]) -> str:
    conds = list(tasks[0]["conditions"].keys()) if tasks else []
    n = len(conds)
    grid_cls = f"cond-grid cond-grid-{min(n, 5)}"

    cards = []
    for task in tasks:
        idx = task["task_idx"]
        text = task["task"]
        cols = []
        for cond in conds:
            data = task["conditions"].get(cond, {})
            ideas = data.get("ideas", [])
            div = data.get("diversity", {})
            mp = div.get("mean_pairwise_sim", None)
            u = data.get("usefulness", {}).get("mean", None)

            pills = []
            if mp is not None:
                pills.append(_pill(mp, sim_color) + ' <span style="color:#475569;font-size:0.75rem">sim</span>')
            if u is not None:
                pills.append(_pill(u, usefulness_color, ".1f") + ' <span style="color:#475569;font-size:0.75rem">use</span>')

            idea_items = "".join(
                f'<div class="idea-item"><span class="idea-num">{i+1}.</span>'
                f'<span class="idea-text">{_esc(idea)}</span></div>'
                for i, idea in enumerate(ideas)
            )
            pills_html = "".join(f'<span>{p}</span>' for p in pills)
            cols.append(
                f'<div class="cond-col">'
                f'<div class="cond-header">'
                f'<h3>{_esc(cond)}</h3>'
                f'<div class="cond-pills">{pills_html}</div>'
                f'</div>'
                f'<div class="idea-list">{idea_items}</div>'
                f'</div>'
            )

        cards.append(
            f'<div class="task-card">'
            f'<div class="task-header">'
            f'<span class="task-idx">#{idx}</span>'
            f'<span class="task-text">{_esc(text)}</span>'
            f'<span class="task-chevron">▶</span>'
            f'</div>'
            f'<div class="task-body"><div class="{grid_cls}">{"".join(cols)}</div></div>'
            f'</div>'
        )

    return f'<div class="tasks-grid">{"".join(cards)}</div>'


def _esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")


def build_report(runs: list[dict], out_path: Path):
    sections = []
    for run in runs:
        path = run["path"]
        tasks = run["tasks"]
        stats = condition_stats(tasks)
        cond_count = len(stats)

        summary_html = render_summary(stats)
        print(f"  Computing per-step embeddings for {path}...")
        step_html = render_step_table(tasks)
        task_html = render_tasks(tasks)

        sections.append(
            f'<div class="run-section">'
            f'<div class="run-label">{_esc(path)}</div>'
            f'<h2>Summary — {len(tasks)} tasks, {cond_count} conditions</h2>'
            f'{summary_html}'
            f'{step_html}'
            f'<br><h2>Task breakdown</h2>'
            f'{task_html}'
            f'</div>'
        )

    body = "\n".join(sections)
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>anti-steer-001 results</title>
<style>{_CSS}</style>
</head>
<body>
<div class="page">
  <div class="header">
    <h1>anti-steer-001 — Idea Diversity via Residual Anti-Steering</h1>
    <p>Semantic similarity between LLM-generated ideas under different conditions.
       Lower similarity = more diverse ideas. Color: <span style="color:#4ade80">green = low sim (good)</span>,
       <span style="color:#f87171">red = high sim (redundant)</span>.</p>
  </div>
  {body}
</div>
<script>{_JS}</script>
</body>
</html>"""

    out_path.write_text(html)
    print(f"Report written to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results", nargs="+", help="One or more results JSON paths")
    parser.add_argument("--out", default=None, help="Output HTML path (default: report.html next to first input)")
    args = parser.parse_args()

    first = Path(args.results[0])
    out = Path(args.out) if args.out else first.parent / "report.html"

    runs = load_runs(args.results)
    total_tasks = sum(len(r["tasks"]) for r in runs)
    print(f"Loaded {len(runs)} run(s), {total_tasks} tasks total")

    build_report(runs, out)


if __name__ == "__main__":
    main()
