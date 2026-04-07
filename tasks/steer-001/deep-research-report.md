# Next Research Directions for Discrete Steering Prefixes in steer-001

## Where the project is today and what the repo actually asks for

The repo’s leading question is whether you can “engineer steering prefixes that reliably steer an LLM’s behavior across many suffix prompts, without explicitly stating their intent.” citeturn11view3 The core experimental setup is: pick a reference steering prefix (in the repo writeup, `Talk only about cats.`), sample a batch of user-like suffix prompts, and search for alternative prefixes whose behavior matches the reference when paired with those suffixes. citeturn11view0turn11view3

The repo’s main metric is teacher-forced cross entropy (CE) to a reference completion generated from `(s_ref + z)` for each suffix `z`, with additional “match” proxies like exact match and token overlap. citeturn11view0 The repo also emphasizes that early completion tokens are weighted more heavily (because early divergences compound), and that greedy decoding was used to reduce entropy in generation, with an explicit note that this choice may have tradeoffs. citeturn11view3

Methodologically, the repo writeup already frames the winning baseline as a coarse-to-fine pipeline: optimize a soft prefix in continuous embedding space, project each soft vector to its nearest token embedding, then do a HotFlip-style coordinate descent in token space to repair the damage done by discretization. citeturn11view1

Your new experiment logs (Exp16–20 complete) indicate meaningful but still incremental progress on the repo’s hardest point: the soft-to-discrete gap. In the logs you provided, best discrete CE improved to **0.6794** (Exp19, prefix length 16), down from the repo’s original HotFlip baseline around **0.74**. This is real headway, but it is also consistent with the repo’s own conclusion that discretization is the dominant bottleneck and that purely local repair methods tend to plateau. citeturn11view1

The best next steps should therefore prioritize two things:

Closing the gap by changing the discretization and discrete-search machinery (not just tuning the same knobs), and strengthening “reliably across many suffix prompts” by upgrading evaluation beyond single-trajectory teacher-forced matching, which the repo explicitly flags as a limitation. citeturn11view3turn11view1

## What prior work says about the soft-to-discrete gap and why your behavior makes sense

A key piece of research that matches your observations is **Prompt Waywardness** (Khashabi et al., NAACL 2022). They document a “wayward” phenomenon where continuous prompts can solve a task while their nearest-neighbor discrete projections correspond to arbitrary, even contradictory, text, with only a small drop in task performance. citeturn12view0 This supports two practical interpretations that fit your logs:

First, “projection quality” (nearest-neighbor tokens) is not guaranteed to correlate tightly with behavioral fidelity, because many continuous solutions lie in regions whose nearest discrete representatives are behaviorally misaligned.

Second, a method that only improves local discrete neighbors (HotFlip style single-token replacement) can get stuck, because the discrete region containing behavior-equivalent prompts can be very sparse and discontinuous, which is exactly how the repo describes it. citeturn11view1turn1view0

A closely related line, **Prompt Obfuscation for Large Language Models** (Pape et al., 2024), explicitly leverages “collisions in continuous embedding space” that do not map cleanly back to a meaningful hard prompt, which they use for system prompt IP protection. citeturn14view1 Even though their goal differs, the empirical punchline reinforces your core difficulty: it can be easy to find embedding-space solutions with desired I/O behavior, and hard to realize them as clean discrete text.

Taken together, these findings imply that further progress is likely to come from methods that either:

Optimize in discrete space more globally (better discrete search and better exploration across basins), or

Modify the continuous-to-discrete interface so that the continuous solution is trained to be “quantization friendly,” not just behaviorally optimal.

## High-leverage algorithmic directions to try next

### Replace HotFlip-only repair with more modern discrete prompt optimizers

HotFlip was introduced as a gradient-based method for choosing discrete token edits by approximating the effect of a flip using gradients with respect to one-hot inputs. citeturn0search1turn0search5 AutoPrompt later popularized gradient-guided search over discrete “trigger” tokens for prompting, building directly on gradient-based token search ideas (including HotFlip-style updates). citeturn10search28turn0search2

Your logs show HotFlip is functioning as intended: a strong local optimizer that recovers a meaningful fraction of lost performance, but it saturates. That aligns with the repo writeup’s own diagnosis that HotFlip is effective locally but infeasible as a global search across the full vocabulary. citeturn11view1

The next lever is to adopt prompt optimizers that are still gradient-driven but better behaved than a simple HotFlip loop. A particularly relevant paper is **Hard Prompts Made Easy** (Wen et al., 2023). They propose a method (PEZ) that maintains continuous prompt embeddings, projects to nearest neighbors on the forward pass, and then updates the continuous iterate using the gradient computed through the projected embeddings. citeturn7view0 Conceptually, this is very close to what your project calls “ST-like” optimization, but Wen et al. explicitly connect it to lessons from discrete optimization in quantized networks, including failure modes like stagnation when projection steps are too harsh. citeturn7view0

Why this is a good next step for steer-001:

It directly targets the same soft-to-hard bridge you are working on, but it is positioned as a more robust, less brittle discrete optimizer than prior “project after each step” baselines. citeturn7view0

It is designed for efficiency and portability of hard prompts, which matches your repo’s motivation around equivalence classes and prompt-only control. citeturn11view2turn7view0

A second branch is to incorporate “coordinate-gradient” style prompt optimization used for universal triggers, but only as an optimization primitive, not as a misuse tool. The **universal adversarial triggers** line shows that short token sequences can be optimized to cause consistent, input-agnostic effects across many examples, using gradient-guided token search, and that these triggers can transfer across models. citeturn0search3turn0search23 The high-level applicability is that robust discrete control strings exist and can be found by optimizing across multiple inputs, which is structurally similar to your “many suffix prompts” requirement.

If compute becomes the bottleneck for these more global coordinate methods, the **Probe sampling** work (Zhao et al., 2024) is worth reading: it proposes a way to accelerate Greedy Coordinate Gradient style prompt optimization by filtering candidate prompts using a smaller draft model when draft and target predictions are similar, reporting large speedups for prompt optimization routines. citeturn17view0turn9search6 Even if you do not adopt the full method, the general idea of using a cheaper proxy model to pre-rank token candidates is directly relevant to scaling beyond TOPK 50–200.

### Make discretization quantization-aware using vector quantization style regularizers

Your repo language about “Voronoi basins” and boundary effects is a geometric way to describe quantization. The most mature conceptual toolkit for this is vector quantization methods, where the training objective explicitly includes terms that stabilize discrete assignment and prevent chattering between codebook entries.

Two especially relevant references:

VQ-VAE (van den Oord et al.) is a canonical example of learning with discrete latent variables via vector quantization. It uses an encoder whose outputs are quantized to a codebook, and training uses tricks like straight-through gradients and codebook-related losses to make learning stable. citeturn12view3turn2search7

VIP (Vector-Quantized Input-Contextualized Prompts) explicitly applies vector quantization to prompt representations, mapping prompt tokens to learnable codebook vectors to stabilize prompt learning, and it frames quantization as controlling representational capacity and reducing variance across diverse inputs. citeturn16view0turn16view2

VQ-Prompt (Li et al., 2024) is another example of replacing a continuous prompt with the nearest element from a discrete prompt pool and then using gradient estimation plus VQ regularization terms to make the process end-to-end trainable. citeturn4view1

Why this is a high-value next direction for steer-001:

Your best results appear sensitive to initialization and numeric details (seed and fp32 vs bfloat16 effects in your narrative). VQ-style commitment and codebook regularization is specifically intended to reduce unstable assignment dynamics and variance. citeturn16view2turn4view1

Unlike a generic “naturalness penalty,” VQ regularization is directly about discretization geometry, which is where you are stuck. Your earlier attempts suggest that optimizing “English-likeness” can fight the steering objective; VQ regularizers instead push continuous solutions toward stable discrete representatives.

Concretely, the next experiments that follow the literature most closely are:

Fix the Exp16 crash and actually run the margin regularization settings you planned, because the idea matches the purpose of commitment losses and boundary avoidance; this is aligned with VIP’s motivation that quantization can reduce noise and stabilize performance. citeturn16view2turn4view1

Prototype a “codebook of allowed tokens” variant: rather than projecting into the full vocabulary embedding table, project into a smaller learned or curated codebook, as used in VIP and VQ-Prompt style methods. citeturn16view2turn4view1 This can also be a clean way to trade off readability, dissimilarity constraints, and search tractability.

### Use exploration methods designed to produce a distribution of prompts rather than one optimum

A recurring theme in your logs is that a single run converges to a deep local minimum and that different seeds land in very different basins. This is exactly the scenario where “single optimum search” underdelivers and “sample a distribution of good prompts” can do better.

Two relevant families:

FluentPrompt (Langevin dynamics with a fluency constraint) is explicitly designed to generate a diverse set of effective prompts rather than a single one, by adding progressive noise and maintaining fluency through a perplexity constraint; it positions itself as a way to avoid the disfluent gibberish prompts common in gradient-based methods like AutoPrompt. citeturn12view1turn9search0 Even if your end goal is not readability, the reason this matters is exploration: Langevin-style noise can help traverse energy barriers between basins.

Genetic Prompt Search (GPS) uses a genetic algorithm to search for high-performing hard prompts and is gradient-free, using a dev set score for selection. citeturn13view0turn13view1 This is slower per improvement, but it is valuable precisely when gradients are unhelpful or lead to local-minimum plateaus.

For steer-001, a practical compromise is a two-stage approach:

Use your current best gradient-based method to generate a pool of candidate discrete prefixes (multiple seeds, multiple runs, possibly with injected noise), then run a shallower gradient-free or mutation-based search only in a small neighborhood of those candidates. This keeps compute bounded while increasing the chance of escaping “good but not best” discrete basins.

## Evaluation upgrades that directly match the repo’s stated concerns

The repo explicitly warns that teacher-forced matching to a single reference completion can reward matching one deterministic trajectory rather than robust steering, and it suggests using task-specific topic adherence metrics across multiple unrelated prompts as a better evaluation direction. citeturn11view0turn11view1 This matters because your optimization is only as good as the signal it is given. If the metric is too narrow, you can overfit to brittle behaviors.

Two research directions that are worth prioritizing, because they change how you measure success:

### Move from single-completion CE to distribution matching

Prompt obfuscation work uses a combined objective with both cross entropy (token correctness) and KL divergence between output probability distributions as a regularizer for “functional equivalence.” citeturn14view1 Even though their context differs, the key lesson generalizes: matching only one sampled output can be too narrow, and matching distributions can be a stabilizer.

For steer-001, “distribution matching” can mean:

For each suffix, keep multiple reference completions or multiple decoding settings, and score the candidate prefix against a mixture of references, not a single output.

Add a KL term that matches model output distributions under `(s_ref + z)` versus `(s + z)` for a window of early tokens, instead of only matching the argmax trajectory.

This aligns with the repo’s own emphasis that early tokens dominate divergence and are weighted heavily. citeturn11view3

### Split suffix prompts into optimization and held-out evaluation sets

The repo’s core requirement is “reliably across diverse user prompts,” phrased as prefix-suffix equivalence for many suffixes `z`. citeturn11view2turn11view3 If the current CE is computed on the same fixed batch used for optimization, it is difficult to know how much you are learning robust steering versus memorizing idiosyncrasies of that suffix set.

A simple but high-impact research step is to build a standardized evaluation protocol:

A train set of suffixes used in optimization.

A validation set used for early stopping, seed selection, and hyperparameter selection.

A test set never used in optimization, used only for reporting.

This is less glamorous than new optimizers, but it sharply improves your ability to judge whether Exp19-level gains are genuine progress toward “reliability.”

## A prioritized portfolio of next experiments

The following portfolio is ordered by expected value per unit of TIDE compute, given what your logs show and what the literature suggests.

### Quantization-aware stabilization of ST and projection

Your current narrative already points at “Voronoi boundary” effects, and VQ-style methods exist specifically to stabilize discrete assignments and reduce variance. citeturn16view2turn4view1turn12view3 Fixing and completing margin regularization runs is a fast, direct test of whether boundary avoidance yields consistent improvements. If it works, it is a general tool, not a one-off trick.

### Replace HotFlip with a more robust discrete optimizer, or hybridize with PEZ-like updates

HotFlip is a crisp local method but tends to plateau. citeturn0search1turn11view1 Hard Prompts Made Easy describes a projection-in-the-loop optimizer intended to be more robust and less prone to stagnation when discrete projection is harsh. citeturn7view0 If you can reproduce any of its stability benefits in your setting, it directly targets your current bottleneck.

### Basin exploration on purpose

Your logs show that “good seeds are unusually good,” and that multi-seed runs are not reliably beating the SOTA. The most direct research framing is: treat the problem as one of basin discovery. FluentPrompt-style noise injection provides a principled way to explore a distribution of prompts, not just descend to one optimum. citeturn12view1 GPS provides a gradient-free alternative if gradients keep returning you to the same attractor. citeturn13view0turn13view1

### Evaluation improvements that reward robust steering rather than one trajectory

The repo itself flags this as an open issue. citeturn11view1turn11view0 Adding distribution matching (for example CE plus KL) is also consistent with the prompt obfuscation literature, which uses exactly that combination to preserve functional similarity. citeturn14view1

### Scaling prefix length beyond 16, but only after improving the search signal

Your SOTA improves with longer prefixes, but the gains are currently incremental. In the literature, longer prompts often increase capacity, but do not automatically solve discrete search and stability issues. citeturn5view1turn2search1turn2search2 Treat prefix length 24 or 32 as a “multiplier” that is most valuable after you have a better optimizer and a better evaluation signal to exploit that added capacity.

## Why these are the “next best steps” given your results so far

Your experimental arc matches the broader field’s story:

Continuous prompts are easy to optimize and can be effective, but discretizing them is difficult and can be “wayward,” meaning nearby discrete projections may not reflect the same function. citeturn12view0turn11view1

Local discrete edits (HotFlip style) help but plateau, which is why multiple later works either improve discrete optimization algorithms (AutoPrompt, Hard Prompts Made Easy) or change the representation and training scheme (VQ-style prompt quantization) to make discrete representations stable and learnable. citeturn10search28turn7view0turn16view2turn4view1

The repo itself anticipates that evaluation can mislead if it over-rewards a single deterministic trajectory, which suggests that further “CE improvements” might not translate into the kind of reliability the repo wants unless evaluation also evolves. citeturn11view0turn11view1

Finally, there is strong evidence from universal trigger research that short discrete token sequences can produce consistent and transferable behavior changes across many inputs, which is encouraging for the existence of robust discrete steering prefixes in principle. citeturn0search3turn0search23 The open problem for steer-001 is therefore less about whether such strings exist, and more about whether your optimization and evaluation pipeline is set up to find them efficiently and verify their robustness.

If you pursue the portfolio above, you should get clearer answers to the most decision-relevant questions:

Is the remaining gap mainly an optimization limitation (better search could reduce CE further), or is it largely a metric and evaluation limitation (you are optimizing the wrong signal), or is it a representational limitation (nearest-neighbor projection is the wrong discretization interface)?

Those answers will determine whether the next major investment should be in optimizer engineering, evaluation redesign, or representational changes like codebooks and quantization-aware training.