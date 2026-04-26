# Idea Anti-Repetition via Internal Suppression

## Purpose

Test whether an LLM can be pushed to generate meaningfully new ideas by suppressing internal representations associated with ideas it has already produced, rather than relying only on prompt-based requests for novelty.

This document scopes a quick and dirty viability experiment for an agentic coding platform. The goal is not to prove a full theory of representation-level novelty control. The goal is to determine whether a lightweight intervention can measurably reduce repeated ideas without destroying usefulness.

## Core hypothesis

After an LLM proposes an idea, it may be possible to discourage closely related follow-up ideas by applying a targeted negative intervention during the next generation pass.

The intervention should work better when it targets a representation of the prior idea than when it bluntly damages the model. In practice, that means feature suppression or anti-steering is more promising than whole-layer ablation.

## Motivation

Agentic coding systems often need to propose multiple candidate approaches to a problem:

* bug fixes
* refactor plans
* experiment designs
* infrastructure changes
* optimization strategies
* product or research ideas

In many cases, repeated generations are superficially different but conceptually redundant. A system that can explicitly avoid previously explored idea regions could improve search breadth, reduce wasted tool calls, and surface more diverse candidate solutions.

## Non-goals

* Build a production-safe interpretability stack
* Prove that specific layers cleanly correspond to syntax vs semantics
* Solve novelty evaluation in a general setting
* Guarantee that generated ideas are globally novel relative to the literature

## Main question

Can we cheaply reduce semantic repetition across repeated idea-generation attempts by suppressing internal representations associated with earlier outputs?

## Framing

There are three levels of intervention worth separating:

### 1. Prompt-only novelty pressure

Ask for a different idea and provide previous ideas in context.

This is the easiest baseline and should almost certainly be included.

### 2. Output-level anti-similarity

Generate several candidate ideas and penalize candidates that are too similar to earlier ideas according to embedding similarity or reranking.

This is likely the strongest simple baseline.

### 3. Internal anti-repetition

During generation of the next idea, suppress internal activations associated with the previous idea or with latent features activated by it.

This is the research target.

## Why not whole-layer ablation

Whole-layer ablation is probably too coarse for a first pass. It is likely to:

* reduce fluency
* damage task performance
* remove useful capabilities unrelated to the previous idea
* produce fake novelty by making the model worse rather than more exploratory

A better first target is the residual stream at selected layers, where a negative steering vector or other localized intervention can be applied.

## Minimal viable mechanism

The simplest viable mechanism is:

1. Ask the model for an idea.
2. Record hidden states while generating a compressed summary of that idea.
3. Construct a steering signal from those activations.
4. During the next decode, subtract that signal at one or more chosen layers.
5. Generate a new idea.
6. Measure whether the new idea is less semantically similar to the prior one while remaining useful.

## Candidate intervention families

### A. Residual anti-steering

Construct a vector representation of the previous idea and subtract it during the next generation pass.

Possible crude version:

* run the model on a short canonicalization prompt such as `Summarize the core idea in one sentence`
* average token activations across selected layers
* subtract a scaled version of that average from the residual stream during the next idea generation

Pros:

* simple to implement
* easy to sweep over layers and strengths

Cons:

* likely entangles idea content with wording and style
* can over-steer and degrade coherence

### B. Contrastive anti-steering

Instead of using the raw idea activation, build a difference vector between:

* a prompt with the previous idea included
* a matched prompt without that idea

Then subtract the difference.

Pros:

* may isolate the idea signal more cleanly

Cons:

* requires prompt pairing
* still fairly blunt

### C. Sparse feature suppression

If the platform already has access to sparse autoencoder features or similar latent features, identify the most activated features during the previous idea summary and suppress only those.

Pros:

* more targeted
* better fit to the intuition that concepts live in features rather than whole layers

Cons:

* more setup
* harder for a quick first test unless feature tooling already exists

## Recommendation for the first experiment

Start with residual anti-steering, because it is easy to wire into an agentic coding platform with model hooks and produces a fast read on viability.

Do not start with literal layer ablation.

## Quick and dirty experiment

### Objective

Test whether internal anti-steering produces more diverse follow-up ideas than prompt-only generation, without a major drop in usefulness.

### Task

Use a constrained coding-ideas task where semantic repetition is common and evaluation is easy enough to automate.

Recommended task template:

> Given a repository issue, propose 5 distinct implementation approaches.

Good input sources:

* synthetic repo tasks
* a curated set of real GitHub issues
* internal benchmark tasks for code agents

Pick around 30 to 50 tasks for a fast first pass.

### Experimental conditions

For each task, generate a sequence of 5 ideas under each condition.

#### Condition 1: Prompt-only baseline

Each round includes prior ideas in the prompt and asks for a substantially different new idea.

#### Condition 2: Output-level novelty baseline

Generate multiple candidates each round and rerank away from prior ideas using embedding similarity.

#### Condition 3: Internal anti-steering

Generate each new idea with negative steering derived from the immediately previous idea.

#### Optional condition 4: Hybrid

Apply internal anti-steering and then rerank candidates by output-level novelty.

This hybrid may end up strongest even if internal steering alone is weak.

### Generation protocol

For a given task:

1. Generate idea 1 normally.
2. Compress idea 1 into a one-sentence canonical representation.
3. Build the intervention signal from that canonicalization pass.
4. Generate idea 2 with the intervention active.
5. Repeat using memory of all prior ideas.

Two simple memory variants:

* subtract only the most recent idea representation
* subtract the mean of all prior idea representations

### Intervention details

#### Layers to test

Use a small sweep over mid and late layers, not all layers.

Example:

* one middle layer
* one late-middle layer
* one late layer
* two-layer combination

#### Strengths to test

Use a small scalar sweep.

Example:

* 0.25
* 0.5
* 1.0
* 2.0

Keep this coarse. The purpose is to detect a signal, not optimize perfectly.

#### Canonicalization prompt

Use a stable prompt that tries to strip away phrasing and capture only the proposal.

Example:

> State the core implementation idea in one sentence, focusing on the main mechanism and ignoring wording details.

This should make the intervention target less style-dependent.

## Evaluation

### Primary metric: intra-sequence semantic diversity

For each 5-idea sequence, compute pairwise embedding similarity between all ideas.

Measure:

* mean pairwise cosine similarity
* max similarity to any prior idea for each step

Lower is better, assuming usefulness is preserved.

### Secondary metric: structural diversity

Use an LLM judge or a simple rubric to classify whether two ideas are actually different at the mechanism level.

Example categories:

* algorithmic change
* retrieval change
* planner change
* test-time search change
* infrastructure change
* prompt-only change

If two ideas fall in the same mechanism bucket with minor wording changes, count them as redundant.

### Tertiary metric: usefulness

Use either an LLM judge or a cheap hand-review sample to score each idea on:

* plausibility
* relevance to the task
* implementation usefulness

A simple 1 to 5 scale is enough for the first pass.

### Failure metric: degeneration

Track signs that the model is being damaged rather than diversified:

* shorter outputs
* incoherent outputs
* vague or generic ideas
* repeated disclaimers
* syntax damage or malformed code-like text

## Success criteria

Call the experiment promising if internal anti-steering shows both of these:

1. materially lower semantic similarity than prompt-only baseline
2. little or no drop in usefulness relative to baseline

A weaker but still interesting outcome is:

* internal anti-steering alone is noisy
* hybrid internal plus reranking beats output-level reranking alone

A negative result would be:

* novelty only rises when usefulness clearly falls
* or output-level reranking dominates with much less complexity

## Implementation sketch for an agentic coding platform

### Components

#### 1. Idea generator

A standard chat completion call that produces one candidate idea at a time.

#### 2. Canonicalizer

A second pass that compresses each idea to a one-sentence core mechanism statement.

#### 3. Activation capture hook

Capture residual stream activations for the canonicalizer pass at chosen token positions and layers.

#### 4. Intervention hook

During the next generation call, subtract a scaled steering vector at the same layers.

#### 5. Novelty scorer

Compute embedding similarity against prior ideas and log results.

#### 6. Judge

Use a lightweight evaluator to score usefulness and mechanism-level distinctness.

### Pseudocode

```python
for task in tasks:
    prior_ideas = []
    prior_vectors = []

    for step in range(5):
        if step == 0:
            idea = generate_idea(task, prior_ideas, intervention=None)
        else:
            steering = mean(prior_vectors)
            idea = generate_idea(task, prior_ideas, intervention=negative(steering))

        summary = canonicalize_idea(idea)
        vector = capture_summary_vector(summary, selected_layers)

        prior_ideas.append(idea)
        prior_vectors.append(vector)

    score_diversity(prior_ideas)
    score_usefulness(task, prior_ideas)
```

## Design choices that matter

### What tokens to average

Three plausible options:

* average all summary tokens
* average only the final token
* average content tokens only

For the first pass, average all summary tokens and keep it simple.

### Whether to target one idea or all prior ideas

For a quick test, compare:

* previous idea only
* average of all prior ideas

The first may improve local novelty. The second may broaden search more aggressively.

### Whether to intervene during all decoding steps

For simplicity, yes. More surgical scheduling can come later.

### Whether to intervene on the generator or canonicalizer context

Intervene only on the generator.

## Risks and expected failure modes

### 1. Surface novelty without conceptual novelty

The model may produce different wording for the same underlying mechanism.

Mitigation:

* use canonicalization
* use mechanism-level judging
* inspect false positives manually

### 2. Capability damage

The intervention may reduce quality instead of encouraging exploration.

Mitigation:

* prefer vector subtraction over ablation
* sweep low strengths first
* track degeneration metrics

### 3. Representation entanglement

The captured vector may mix topic, style, and quality with the idea itself.

Mitigation:

* use contrastive vectors in a follow-up experiment
* compare summary-based vectors to raw idea vectors

### 4. Output-level baselines may already solve most of the problem

That would reduce the value of internal methods.

Mitigation:

* compare against reranking directly
* focus on whether internal methods improve the search frontier beyond reranking

## If the first experiment works

Next steps:

1. move from raw residual anti-steering to contrastive vectors
2. test sparse feature suppression if tooling exists
3. evaluate on longer agent trajectories, not just isolated idea lists
4. study whether broader idea search improves downstream coding outcomes
5. test whether novelty control helps tree search or planner branching in code agents

## If the first experiment fails

Most likely interpretations:

* the intervention is too blunt
* the vector extraction is too entangled with phrasing
* output-level novelty control is the better engineering solution

Useful fallback:

* keep the novelty scorer and reranker
* drop internal interventions unless stronger feature tooling becomes available

## Minimal deliverable for the platform team

A prototype is enough if it can do all of the following:

* run repeated idea generation on a small benchmark set
* capture activations for one model
* apply negative steering at selected layers
* log idea text, summaries, similarity scores, and judge scores
* export a simple comparison table across conditions

## Bottom line

This is a reasonable experiment.

The sharp version is not "ablate layers to force new ideas." The sharp version is "suppress internal representations of already explored ideas and test whether that expands the search space without damaging usefulness."

For a fast build on an agentic coding platform, residual anti-steering plus output-level novelty scoring is the right first pass.
