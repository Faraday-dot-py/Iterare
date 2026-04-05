"""
Analyze and compare all steer-001 experiment results.
Run after all experiments complete.
"""

import json
from pathlib import Path

BASE = Path(__file__).parent

# ─── Load results ─────────────────────────────────────────────────────────────

def load(path):
    p = Path(path)
    return json.loads(p.read_text()) if p.exists() else None


exp1 = load(BASE / "manager-1/worker-1/baseline_results.json")
exp2 = load(BASE / "manager-1/worker-1/multiprefixes_results.json")
exp3 = load(BASE / "manager-3/worker-1/layerablation_results.json")
exp4 = load(BASE / "manager-4/worker-1/naturalness_results.json")
exp5 = load(BASE / "manager-5/worker-1/prefixlen_results.json")
exp6 = load(BASE / "manager-6/worker-1/iterative_results.json")
exp7 = load(BASE / "manager-7/worker-1/gumbel_results.json")

# ─── Exp 1 baseline ───────────────────────────────────────────────────────────
if exp1:
    m = exp1["metrics"]
    print("=== EXP 1: BASELINE ===")
    print(f"  Soft CE:      {m['soft_ce_final']:.5f}")
    print(f"  Proj CE:      {m['projection_ce']:.5f}  (gap: +{m['gap_soft_to_proj']:.5f})")
    print(f"  HotFlip CE:   {m['hotflip_ce_final']:.5f}  (recovered: {m['recovery_hotflip']:.5f})")
    print(f"  Final prefix: {exp1['final_prefix_text']!r}")
    print()

# ─── Exp 2 multi-prefix ───────────────────────────────────────────────────────
if exp2:
    print("=== EXP 2: MULTI-PREFIX ===")
    print(f"  {'Prefix':45s} {'Soft CE':>9} {'Gap':>8} {'HotFlip':>9} {'Final prefix'}")
    print("  " + "-" * 100)
    for r in exp2:
        m = r["metrics"]
        print(f"  {r['reference_prefix']:45s} {m['soft_ce_final']:>9.5f} {m['gap_soft_to_proj']:>8.5f} "
              f"{m['hotflip_ce_final']:>9.5f}  {r['final_prefix_text']!r}")
    print()

# ─── Exp 3 layer ablation ─────────────────────────────────────────────────────
if exp3:
    print("=== EXP 3: LAYER ABLATION ===")
    print(f"  {'Layer':>6} {'Proj CE':>10} {'HotFlip CE':>12} {'Proj Nat':>10} {'Final Nat':>10}  {'Final Prefix'}")
    print("  " + "-" * 85)
    for l, r in exp3.get("results_by_layer", {}).items():
        print(f"  {l:>6} {r['projection_ce']:>10.5f} {r['hotflip_ce']:>12.5f} "
              f"{r['projection_naturalness']:>10.3f} {r['final_naturalness']:>10.3f}  {r['final_text']!r}")
    print()

# ─── Exp 4 naturalness penalty ────────────────────────────────────────────────
if exp4:
    res = exp4.get("results", {})
    lr = res.get("lambda_results", {})
    print("=== EXP 4: NATURALNESS PENALTY ===")
    print(f"  Shared soft CE: {res.get('soft_ce', 'N/A'):.5f}")
    print(f"  Shared proj CE: {res.get('projection_ce', 'N/A'):.5f}")
    print(f"  {'λ':>6} {'HotFlip CE':>12} {'Naturalness':>13}  {'Final Prefix'}")
    print("  " + "-" * 80)
    for lam_s, r in lr.items():
        print(f"  {r['lambda']:>6.1f} {r['hotflip_ce']:>12.5f} {r['final_naturalness']:>13.3f}  {r['final_text']!r}")
    print()

# ─── Exp 5 prefix length ──────────────────────────────────────────────────────
if exp5:
    print("=== EXP 5: PREFIX LENGTH ===")
    print(f"  {'Len':>5} {'Soft CE':>9} {'Proj CE':>9} {'Gap':>8} {'HotFlip CE':>12} {'Recovery%':>11}  {'Final Prefix'}")
    print("  " + "-" * 100)
    for pl, r in exp5.get("results_by_length", {}).items():
        rec = r["recovery_fraction"] * 100
        print(f"  {pl:>5} {r['soft_ce']:>9.5f} {r['projection_ce']:>9.5f} "
              f"{r['projection_ce']-r['soft_ce']:>8.5f} {r['hotflip_ce']:>12.5f} {rec:>10.1f}%  {r['final_text']!r}")
    print()

# ─── Exp 6 iterative projection ───────────────────────────────────────────────
if exp6:
    print("=== EXP 6: ITERATIVE PROJECTION ===")
    print(f"  {'Round':>6} {'Soft CE':>9} {'Proj CE':>9} {'HotFlip CE':>12}  {'Final Prefix'}")
    print("  " + "-" * 85)
    for r in exp6.get("rounds", []):
        print(f"  {r['round']:>6} {r['soft_ce']:>9.5f} {r['projection_ce']:>9.5f} "
              f"{r['hotflip_ce']:>12.5f}  {r['final_text']!r}")
    print(f"\n  Best overall: CE={exp6.get('best_overall_ce', 'N/A'):.5f} | {exp6.get('best_text', '')!r}")
    print()

# ─── Exp 7 Gumbel-Softmax ────────────────────────────────────────────────────
if exp7:
    m = exp7.get("metrics", {})
    print("=== EXP 7: GUMBEL-SOFTMAX ===")
    print(f"  Gumbel argmax CE:       {m.get('gumbel_argmax_ce', 'N/A'):.5f}")
    print(f"  Gumbel + HotFlip CE:    {m.get('hotflip_ce_from_gumbel', 'N/A'):.5f}")
    print(f"  Exp 1 baseline CE:      {m.get('exp1_baseline_hotflip_ce', 0.740):.5f}")
    improvement = m.get("improvement_over_baseline", 0)
    print(f"  Improvement over base:  {improvement:+.5f}")
    print(f"  Gumbel argmax prefix:   {exp7.get('gumbel_argmax_text', '')!r}")
    print(f"  Final prefix:           {exp7.get('final_text', '')!r}")
    print()

# ─── Summary table ────────────────────────────────────────────────────────────
print("=== SUMMARY: BEST CE BY METHOD ===")
print(f"  {'Method':40s} {'Best CE':>9}  {'Notes'}")
print("  " + "-" * 80)

if exp1:
    m = exp1["metrics"]
    print(f"  {'Standard pipeline (Exp 1)':40s} {m['hotflip_ce_final']:>9.5f}  baseline")

if exp2:
    best = min(r["metrics"]["hotflip_ce_final"] for r in exp2)
    best_name = min(exp2, key=lambda r: r["metrics"]["hotflip_ce_final"])["reference_prefix"]
    print(f"  {'Best multi-prefix (Exp 2)':40s} {best:>9.5f}  {best_name!r}")

if exp3:
    best_layer = min(exp3.get("results_by_layer", {}).items(),
                     key=lambda x: x[1]["hotflip_ce"], default=(None, None))
    if best_layer[0]:
        print(f"  {f'Best layer proj (Exp 3, layer {best_layer[0]})':40s} {best_layer[1]['hotflip_ce']:>9.5f}")

if exp5:
    best_len = min(exp5.get("results_by_length", {}).items(),
                   key=lambda x: x[1]["hotflip_ce"], default=(None, None))
    if best_len[0]:
        print(f"  {f'Best prefix len (Exp 5, len={best_len[0]})':40s} {best_len[1]['hotflip_ce']:>9.5f}")

if exp6:
    print(f"  {'Iterative projection (Exp 6, best round)':40s} {exp6.get('best_overall_ce', 999):>9.5f}")

if exp7:
    m = exp7.get("metrics", {})
    print(f"  {'Gumbel-Softmax + HotFlip (Exp 7)':40s} {m.get('hotflip_ce_from_gumbel', 999):>9.5f}")
