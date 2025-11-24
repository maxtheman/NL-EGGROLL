# Titans Explained — Claim Extraction (pass 1)

Sources: `titans-explained.pdf` (abstract, p.1)

## Key Claims (early)
- Provides mathematical formalization of the Titans architecture (memory dynamics, surprise metrics, state spaces). (p.1)
- Highlights hierarchical memory systems (contextual and persistent) and argues for computational scalability vs Transformers. (p.1)
- Offers comparative analyses to demonstrate adaptability and theoretical robustness. (p.1)

## Intuition to Reproduce
- “Surprise” as a driver for memory updates: larger mismatch triggers stronger writes.
- Contextual vs persistent memory: fast-changing vs slow-consolidating components to balance cost and recall.
- Formalism clarifies where complexity reductions arise (structured memory operations vs full attention).

## Potential Failure Cases / Why It Might Not Work
- Surprise metric may be noisy/unreliable, leading to unstable writes or overfitting recent tokens.
- Assumptions (stationarity, bounded surprise) may not hold in real data; theory-practice gap.
- If metrics cannot be measured cleanly in code, implementation may diverge from math guarantees.
- Complexity claims may hinge on specific parameter regimes; could regress to attention-like costs otherwise.

## Proposed Uses in Tests
- Extract equations to parameterize synthetic surprise-driven tasks and evaluation metrics.
- Validate that empirical Titans experiments align with the defined dynamics (e.g., update rules, stability conditions).
- Visuals: plot surprise signals over time; compare to attention weights; show memory state trajectories.

## Reviewer Prompts
- Are the formal assumptions (e.g., stationarity, bounded surprise) realistic for target tasks?
- Do the defined metrics map cleanly to measurable quantities in experiments?
- Any gaps between math formalism and the implementation in Titans code?

## Links/Artifacts
- Will add marimo notebook: `notebooks/titans_explained_claims.py` (stub in repo).
