# Nested Learning / HOPE — Claim Extraction (pass 1.5)

Sources: `nestedlearning.pdf` (pp.1–5, abstract, intro, assoc. memory section)

## Key Claims (with page anchors)
- NL models = nested/parallel optimization problems, each with its own context flow; deep learning is a flattened image of this (p.1, Fig 2).  
- Existing models compress their own context flow; in-context learning emerges from this compression (p.1).  
- Optimizers (SGD+Mom, Adam) are associative memory modules compressing gradients (p.1, 2.1).  
- Multi time-scale updates (brain-wave analogy) unlock continual learning; Transformers can be seen as linear layers with different frequency updates (p.2).  
- Formalizes learning as associative memory: memory is a neural update caused by input; learning is acquiring effective memory (p.4).  
- Backprop itself can be cast as solving an associative memory mapping from inputs to gradient “surprise” signals (p.5 eqs.3–6).

## Intuition to Reproduce
- Optimizer-as-memory: gradient history is compressed into state; learning rule is a write policy.  
- Context flow: every level has its own data stream and write/read; nesting these flows yields higher-order in-context learning.  
- Multi-frequency: fast writes for immediate context, slow consolidation for long-term; think gamma vs delta bands.  
- Backprop as memory fitting: weight update solves a proximal problem storing surprise signals.

## Potential Failure Cases / Why It Might Not Work
- **Stability of nested optimizers:** inner loops can explode/oscillate; need damping or trust-region style updates.  
- **Interference across frequencies:** fast updates may corrupt slow memories if write isolation is weak.  
- **Credit assignment leakage:** without explicit gating, nested loops may overwrite useful structure (catastrophic interference).  
- **Compute/latency:** multi-frequency updates increase per-step cost; may negate gains unless sparsified.  
- **Surprise signal noise:** if local surprise is noisy, memory updates may drift; needs normalization/clipping.  
- **Theoretical gaps:** associative-memory framing may not guarantee convergence; lacking bounds for stacked optimizers.  
- **Data regime sensitivity:** benefits might vanish on stationary tasks where simple attention suffices.

## Proposed Tests (marimo/MLX; small, runnable)
- Copy/induction toy (length 64–256): HOPE-style fast-weight layer vs Transformer/RNN; metrics = loss, adaptation speed, interference.  
- Optimizer swap ablation: learned optimizer vs fixed momentum/Adam; check stability and forgetting.  
- Frequency ablation: single vs dual-frequency updates; measure long-horizon recall vs runtime.  
- Noise stress: inject noise into surprise signals to see drift/robustness.  
- Visuals: weight-update heatmaps per frequency; surprise magnitude over time; forgetting curves.

## Reviewer Prompts (for third-party critique)
- What stability controls are in place for nested updates (norm caps, EMA, projection)?  
- Do multi-frequency writes actually improve retention after controlling for compute?  
- How does the method behave on stationary vs non-stationary tasks?  
- Are gains architecture-driven or due to training tricks (curriculum, normalization)?

## Links/Artifacts
- Marimo notebook stub: `notebooks/nested_learning_claims.py` (to be populated with MLX code).
