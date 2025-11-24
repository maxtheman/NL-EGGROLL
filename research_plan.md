# Mind Map & Validation Plan for Nested Learning + EGGROLL

## Goals
- Build a mind map that links all papers/resources to their claims, required background, and validation tasks.
- Produce per-claim test plans (unit-style) and an integration test that combines the ideas.
- Keep onboarding-friendly: fast.ai–style path from first principles to hands-on labs, with rich visualizations.
- Ground every claim with intuition and invite third-party critique at each pass.

## Source Inventory (quick scan)
- `EGGROLL.pdf` — evolution strategies with low-rank perturbations; backprop-free scaling.
- `nestedlearning.pdf` — Nested Learning / HOPE; optimizers as memory; multi-frequency loops.
- `titans.pdf` — Titans memory module; memorization at test time; long-context efficiency.
- `titans-explained.pdf` — mathematical exposition of Titans (surprise metrics, memory dynamics).
- `other_resources.md` — link to official eggroll technical docs.

## Mind Map Structure (intended outline)
- Root: “Nested Learning + EGGROLL Stack”
- Level 1: Papers/resources (EGGROLL, Nested Learning/HOPE, Titans, Titans-explained, docs).
- Level 2: Claim buckets per paper
  - Algorithmic claims (update rules, memory forms, complexity).
  - Theoretical claims (optimizer-as-memory, convergence, surprise metrics).
  - Empirical claims (benchmarks, scaling, ablations).
- Level 3: Validation nodes per claim
  - What to reproduce (task/dataset), metrics, baselines, acceptance criteria.
  - Implementation hooks (pseudo/code refs, parameters to sweep).
- Level 4: Integration nodes
  - Cross-paper synthesis (e.g., HOPE fast weights trained via EGGROLL).
  - Onboarding assets (notebooks, labs, explainer slides).

## Research Workflow (before deep dive)
1) Claim harvesting: skim each paper to extract explicit claims/tables/algorithms; log page refs; capture intuition text for every claim.
2) Prereq mapping: note background concepts and external refs needed per claim (pull offline docs for MLX, marimo, uv; use `exa-code`/`curl` sparingly to mirror docs into `docs/`).
3) Test design per claim: define dataset/task, baseline, metric, minimal reproduction target; add visualization hook and “third-party critique” questions.
4) Integration design: plan an end-to-end demo combining methods; specify marimo notebook stubs for runnable experiments.
5) Packaging: convert mind map into onboarding artifacts (readme, labs, slides/notebooks); ensure JSON Canvas map links out to markdowns/notebooks.

## Per-Paper Validation Plan (initial)
### EGGROLL (ES with low-rank perturbations)
- Claims to capture: scalable ES without backprop; low-rank noise `E = (1/√r) A B^T`; better throughput/stability at large populations.
- Tests:
  - Compare ES (EGGROLL) vs backprop on small model/task (e.g., CIFAR-10 subset or synthetic control): measure wallclock, sample efficiency, variance.
  - Non-differentiable toy task (e.g., discrete control, Parity/XOR) to show ES advantage.
  - Scaling probe: population size vs performance/stability; monitor gradient-free update quality.
  - Intuition to reproduce: ES as population-level memory/aggregation; why low-rank noise is efficient.
- Fitness metrics: average fitness/reward per iteration, sample efficiency (fitness vs evaluations), variance across population, norm of aggregated update, divergence incidents, tokens/evals per second, communication cost proxy (rank vs bandwidth).
- Ablations: rank `r` sweep, noise scale sweep, population size sweep, low-rank vs full-rank noise, fitness smoothing on/off, antithetic sampling on/off, ES vs backprop baseline.
  - **Pushback**: Population 100+ is likely infeasible on local Apple Silicon; target 8–32 pop for first passes, with tiny models, before scaling.
### Nested Learning / HOPE
- Claims: optimizer-as-memory; fast/slow frequency loops; continuum memory beats pure attention/RNN on long/continual tasks.
- Tests:
  - Copy/induction or algorithmic task to measure continual adaptation vs Transformer/RNN baseline.
  - Ablate learned optimizer vs fixed momentum/Adam to verify “optimizer-as-memory” effect.
  - Frequency ablation: single frequency vs multi-frequency to test long-horizon retention.
  - Intuition to reproduce: optimizers compress gradient history as memory; brain-wave frequency analogy for nested loops.
  - Failure/critique hooks: stability of nested optimizers, interference across frequencies, compute overhead, noisy surprise signals. Ablations: learned vs fixed optimizer, frequency on/off, gating/normalization on/off, surprise noise injected.
### Titans
- Claims: neural memory module enabling long-context recall with parallelizable training; better cost than quadratic attention.
- Tests:
  - Long-sequence synthetic recall (e.g., retrieval at 8k–64k tokens) vs Transformer/RetNet; track accuracy and runtime/memory.
  - **Data Strategy**: Use **BABILong** or **Long Range Arena (LRA)** (specifically Associative Recall) to rigorously test long-context. **Pushback:** LRA/BABILong are heavy; start with synthetic far-pointer/copy tasks at 8k–16k tokens and only move to LRA subsets if feasible on Mac.
  - Ablate contextual vs persistent memory pieces; measure degradation.
  - Throughput benchmark (tokens/s) under fixed hardware.
  - Intuition to reproduce: separating contextual vs persistent memory to manage cost/recall tradeoff.
  - Specific tests: far-pointer retrieval at 8k/32k/64k; copy/induction with distractors; staleness test (delayed refresh vs surprise-triggered writes); capacity sweep (memory size/sparsity); gating strength sweep; compare Titans vs Transformer vs RetNet at matched compute.
  - Failure/critique hooks: memory drift/staleness, interference, capacity mismatch, training instability, benchmark sensitivity.
### Titans-Explained (math deep dive)
- Role: provides formal definitions (surprise metrics, memory dynamics) to ground tests.
- Action: extract equations to parameterize synthetic tasks and evaluation metrics; ensure consistency with Titans experiments.
  - **Unit Test**: Implement a specific unit test for the "Surprise" metric to verify it correlates with loss on a toy sequence before full training.
  - Intuition to reproduce: how “surprise” drives updates and stabilizes memory.

## Cross-Paper Integration Test (Capstone)
> [!WARNING]
> This integration is a novel research endeavor with high risk. We will adopt a phased approach.

### Phase 1: HOPE + EGGROLL
- **Goal**: Train a simple fast-weight network with ES.
- **Verification**: Verify "optimizer-as-memory" works without gradients.
- **Fallback**: If ES fails, analyze gradient variance vs ES noise scale.

### Phase 2: Titans + Backprop
- **Goal**: Verify Titans implementation works with standard training first.
- **Baselines**: Compare against **GPT-2 small (RoPE)** and `mlx-examples` LLM implementation.
- **Verification**: Reproduce Titans performance on Associative Recall (BABILong).

### Phase 3: Full Integration
- **Goal**: Construct a HOPE/Titans-style sequence model trained with EGGROLL ES.
- **Condition**: Only proceed if Phase 1 and 2 succeed.
- **Curriculum**: Synthetic tasks combining continual adaptation + long-context retrieval (copy/induction + far-pointer).
- **Metrics**: (1) adaptation speed within episode, (2) long-horizon recall, (3) throughput/energy vs backprop baseline.
- **Fallback**: If ES fails to train the memory module, revert to training only the "fast weights" (HOPE) with ES while keeping Titans memory fixed or backprop-trained.

## Compute Budget & Resource Planning
- **Constraint**: Local Apple Silicon (Mac).
- **Estimation**:
  - ES Population: target 8–32 for initial runs; only scale if perf/memory allow.
  - Memory: Titans state is $O(N)$ or $O(1)$ depending on implementation, but population adds $O(P)$.
  - **Action**: Estimate memory usage per instance. Use `vmap` in MLX. If memory is tight, scale down model size (width/depth) for ES experiments, or reduce population size.
  - **Pushback:** GPT-2 small baseline may be too heavy for long-context on local Mac; prefer tiny Transformer/RetNet baselines sized similarly to Titans-lite.

## Onboarding Path (fast.ai–style modules)
1) ES fundamentals → implement tiny ES and visualize low-rank noise.
2) Optimizer-as-memory → build fast-weight layer; show learned vs fixed optimizer behavior.
3) Continuum/fast weights → recreate HOPE-style loop on toy tasks.
4) Titans memory math → translate equations to code; run long-context benchmark.
5) Combine with EGGROLL → train HOPE/Titans-lite with ES; compare to backprop.
6) Capstone lab → run integration test, log metrics, write findings.

## Tooling & Local Docs
- Implementation stack: MLX for models; marimo notebooks for experiments; `uv` for environment/bootstrap.
- Action: mirror key docs into `docs/` (MLX, marimo, uv quickstart/CLI) using `curl`/`exa-code`; keep versions noted.
- Visualization: prioritize plots in marimo (loss curves, recall vs distance, throughput vs population size).

## Folder Layout (initial)
- `research_plan.md` — master plan (this file).
- `mind_map/roadmap.canvas` — JSON Canvas mind map linking resources.
- `docs/` — mirrored docs (MLX, marimo, uv, eggroll site snippets).
- `notebooks/` — marimo notebooks (per-claim tests, integration lab).
- `notes/` (future) — claim extraction summaries with page refs.
- `third_party_references/` — external links and local clones (`nested_learning`, `Titans-Learning-to-Memorize-at-Test-Time`, `nano-egg`).

## Review & Critique Loop
- For every claim/test: include intuition write-up + “questions for reviewer” section.
- Third-party critique scheduled each pass; keep changelog of feedback and responses.

## Deliverables to produce next
- Populate mind map (`mind_map/roadmap.canvas`) with page refs and per-claim test specs.
- Short readme/guide for new contributors on how to run each test and where to start.
- Minimal reproducible scripts/notebooks for each test node.
- Progress (Phase 1): uv env created (`uv venv` + `uv pip install marimo mlx matplotlib`); `notebooks/eggroll_claims.py` now includes low-rank ES demo, SGD baseline on a quadratic toy (pop=8, rank=2), and an ES-only non-differentiable step-reward toy. Quick CLI sanity: ES reward -7.9→-5.1 (20 steps); SGD loss 6.05→0.14; non-diff ES acc 0.31→0.53.

## Pushback / Open Concerns
- Compute realism: keep ES populations small and models tiny for local Apple Silicon; avoid heavy LRA/BABILong unless subsets are manageable.
- Baseline choice: GPT-2 small long-context may not be viable locally; use scaled-down Transformer/RetNet baselines.
- Integration risk: phase gates must be strict; do not attempt full HOPE+Titans+ES until Phase 1/2 are validated.
