# Titans — Claim Extraction (pass 1)

Sources: `titans.pdf` (abstract, p.1)

## Key Claims (early)
- Introduces a neural long-term memory module that learns to memorize historical context, enabling attention to use long past information. (p.1)
- Provides fast, parallelizable training while maintaining fast inference. (p.1)
- Aims to beat quadratic-attention limitations on context length while preserving modeling accuracy.

## Intuition to Reproduce
- Memory module separates storage from attention: compress long history into persistent state; attend over current context plus memory summary.
- Parallelizable training via structured updates; inference remains lightweight because memory lookups are cheaper than full attention over long windows.

## Potential Failure Cases / Why It Might Not Work
- Memory drift/staleness: persistent state may become outdated without good refresh or surprise-triggered writes.
- Interference: contextual vs persistent memories may overwrite each other if gating is weak.
- Capacity mismatch: too small memory -> loss of detail; too large -> compute/memory cost erodes gains.
- Training instability: writing to memory during training can cause exploding activations if not normalized.
- Benchmark sensitivity: advantages may disappear on moderate lengths where attention cost is acceptable.

## Proposed Tests (marimo/MLX)
- Long-sequence synthetic recall (e.g., pointer/far retrieval at 8k–64k tokens) vs Transformer and RetNet; track accuracy and runtime/memory.
- Ablation: contextual vs persistent memory components to measure contribution.
- Throughput benchmark (tokens/s, memory footprint) under fixed hardware.
- Visuals: recall vs distance curves; throughput vs sequence length; memory utilization heatmaps.
- Specific test cases:
  - Far-pointer retrieval at 8k/32k/64k tokens; accuracy vs distance.
  - Copy/induction with distractors; measure interference/forgetting.
  - Staleness: delayed refresh vs surprise-triggered writes to assess drift.
  - Capacity sweep: memory size/sparsity vs accuracy/cost.
  - Gating strength sweep for contextual/persistent writes.
  - Baselines: Transformer and RetNet at matched compute/params.

## Reviewer Prompts
- Does the memory introduce drift or staleness over very long horizons?
- How sensitive is performance to memory size/refresh policy?
- Are gains architecture-specific or training-schedule-dependent?

## Links/Artifacts
- Will add marimo notebook: `notebooks/titans_claims.py` (stub in repo).
