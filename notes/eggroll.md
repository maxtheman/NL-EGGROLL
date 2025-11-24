# EGGROLL (Evolution Strategies at the Hyperscale) — Claim Extraction (pass 1)

Sources: `EGGROLL.pdf` (abstract, p.1)

## Key Claims (early)
- Backprop-free optimization that scales to large population sizes and large neural networks (billions of parameters) using evolution strategies. (p.1)
- Uses low-rank perturbations `E = (1/√r) A B^T` to reduce communication/compute overhead in ES. (p.1)
- Claims excellent scaling potential on non-differentiable or noisy objectives. (p.1)

## Intuition to Reproduce
- ES as population-level memory/aggregation: each perturbation explores parameter space; fitness aggregation approximates gradient direction without backprop.
- Low-rank noise reduces dimensionality of search directions, lowering bandwidth while retaining exploration diversity.
- Suited for non-differentiable objectives because fitness signals replace gradients.

## Potential Failure Cases / Why It Might Not Work
- Communication bottlenecks if low-rank noise still too large for big models; effectiveness depends on rank choice.
- Fitness signal sparsity/noise can mislead updates; may require heavy smoothing or large populations.
- Sample efficiency gap vs backprop on smooth differentiable tasks; ES may lag without variance reduction.
- Covariance mismatch: low-rank perturbations may miss important directions; risk of mode collapse.
- Hyperparameter brittleness (population size, noise scale, rank); stability issues on ill-conditioned losses.

## Paper specifics (fresh read-through)
- Algorithm 1 (Eq. 8): sample A∈R^{m×r}, B∈R^{n×r}, set E = (1/√r) A Bᵀ, update µ ← µ + α (1/N) Σ_i E_i f(µ+σE_i). Gaussian score approximation uses limit r→∞; theoretical error to full ES is O(1/r).
- Key assumption: entries of A,B i.i.d. zero-mean with finite 4th moment; add vanishing Gaussian ϵ for analysis. Overall update rank is min(Nr,m,n) so not actually low-rank.
- Hardware trick: counter-based RNG to reconstruct perturbations on demand (no storage); batched low-rank matmul uses xBi then (·)Aᵀ, cost O(r(m+n)) vs O(mn). For rank1, reduces to vector-dot + outer. They avoid materializing E when summing E_i f_i.
- Empirics: claims 100× throughput vs naïve ES at billion-param scale; tasks include tabula-rasa RL, LLM reasoning (GRPO comparison), pure-integer RNN pretraining with huge populations (64–262k).
- Hyperparams to mirror: σ as noise scale; α absorbs 1/σ and 1/σ₀⁴; population N large; rank r≪min(m,n); recommend antithetic sampling + baselines not central in paper but compatible.

## Proposed Tests (marimo/MLX, uv environment)
- Small-model benchmark: CIFAR-10 subset or synthetic regression; compare EGGROLL ES vs backprop on wallclock, sample efficiency, variance.
- Non-differentiable toy: discrete control or step-function loss to show ES robustness vs gradient failure.
- Scaling probe: population size vs performance/stability; measure throughput and fitness variance.
- Visuals: fitness vs iteration; population diversity; low-rank noise spectrum.
- Fitness metrics to log: avg fitness/reward, fitness vs evals (sample efficiency), variance across population, norm of aggregated update, divergence incidents, evals per second, rank/bandwidth proxy.
- Progress: marimo notebook `notebooks/eggroll_claims.py` contains a low-rank ES demo (pop=8, rank=2) on a quadratic toy, an SGD baseline, and an ES-only non-differentiable step-reward toy.
- Quick local sanity (CLI, tiny quadratic): ES reward improved from -7.9 to -5.1 over 20 steps; SGD loss improved from 6.05 to 0.14 over 20 steps (pop=8, rank=2, lr=0.1).
- Quick local sanity (CLI, non-diff step reward): ES accuracy improved from 0.31 to 0.53 over 20 steps (pop=8, rank=2, noise_scale=0.1).
- Observation: ES improves but is sample-inefficient vs SGD on the smooth quadratic—expected; robustness on non-diff tasks is the advantage case.
- Micro ablation (rank on quadratic, pop=8, noise_scale=0.05): rank1 reward -8.44→-5.20; rank2 -4.39→-2.47; rank4 -4.72→-2.40 (20 steps).
- How to run (local): `uv venv && source .venv/bin/activate && uv pip install marimo mlx matplotlib`; then `uv run python -m py_compile notebooks/eggroll_claims.py` or `uv run marimo run notebooks/eggroll_claims.py`.
- Noise scale sweep (quadratic, pop=8, rank=2, 20 steps): sigma 0.01 → -8.55→-7.99; 0.05 → -7.17→-4.25; 0.10 → -7.70→-2.84; 0.20 → -5.49→-0.37.
- Population sweep (quadratic, sigma=0.1, rank=2, 20 steps): pop 4 → -5.54→-1.91; pop 8 → -4.60→-1.18; pop 16 → -4.66→-0.76.
- Low-rank vs full noise (hi-dim linear d=16, pop=16, rank=4, scale=0.05, 20 steps): low-rank start/end ≈ reported in notebook; full noise also tracked (see `notebooks/eggroll_claims.py`).
- Ill-conditioned quadratic (feature scale 1 vs 100): ES converged from ~-18k to ~-21 / -70 / -524 depending on sigma (0.05/0.1/0.2) with pop=16; SGD with naive lr=0.01 diverged to NaN. Suggests ES robustness to ill-conditioning vs naive SGD.
- Two-moons (non-convex, MLP hidden=8, 50 steps): ES accuracy 0.50→0.88; SGD (MSE surrogate) 0.50→0.84. ES competitive/slightly better on accuracy; SGD lowers loss more. Useful sanity for non-convex setting.
- Quadratic sweeps (50 steps, pop=16 unless noted):  
  - scale2=1: ES best end ~-0.02 (sigma=0.05) vs SGD lr=0.1 hitting ~1e-8 loss in 0.013s (50 evals).  
  - scale2=10: ES end ~-1.15 (sigma=0.1) vs SGD lr=0.01 end 0.24 loss; lr=0.1 blows up.  
  - scale2=100: ES ends [-46, -84, -159] (sigma 0.05/0.1/0.2); SGD diverges for lr≥1e-3. ES stable where SGD unstable.
- Two-moons noise sensitivity (noise=0.05/0.1/0.2, pop=16, sigma=0.1, 50 steps):  
  - ES acc: 0.625→0.885 / 0.885→0.875 / 0.5→0.84, time ~0.28–0.31s, evals 800.  
  - SGD acc: 0.33→0.855 / 0.5→0.805 / 0.545→0.83, time ~0.03s, evals 50; loss drops ~1.2→0.38–0.44.  
  - ES matches/beats accuracy but costs ~10x time and ~16x evals.
- Two-moons sigma/pop grid (noise=0.1, 50 steps):  
  - sigma 0.05 pop 8: 0.50→0.50 (stalled); pop 16/32: 0.50→0.875.  
  - sigma 0.1 pop 8/16/32: ~0.24–0.54→~0.875–0.90.  
  - sigma 0.2 pop 8/16/32: 0.49–0.81→~0.89–0.90.  
  - SGD baseline: 0.58→0.845 in 0.036s (50 evals).

## Nano-eggroll baseline (code read, `third_party_references/repos/nano-egg/run.py`)
- Objective: int8-only minGRU LM on minipile; ES-only training with low-rank LoRA-style perturbations.
- Perturbation scheme: pre-sampled BIG_RAND_MATRIX, deterministic slicing per epoch/thread; rank defaults to 1 (get_lora_update_params) with antithetic pairing (thread_id even/odd sign flip). Also supports full elementwise perturbations for non-LoRA params.
- Fitness handling: `fast_fitness` uses sign of paired score diff; else CLT-style normalization. Update threshold from Z-test quantile (`alpha`); can decay α over epochs. `sigma_shift` sets σ=2^{-sigma_shift}. `noise_reuse` reuses perturbations across epochs; `group_size` controls data reuse for antithetic pairs.
- Training loop: batch generates `parallel_generations_per_gpu` sequences (grouped) for `tokens_per_update`; uses vmapped `generate_thread` to accumulate token-level log-likelihood (bits/byte) and vmapped update. Validation optional every `validate_every`.
- Data: builds cached numpy arrays of minipile text; prepends 0 token per sequence; uses uint8 vocab_size=256. Logging via wandb optional.

## MLX adaptation notes
- Need counter-based or reproducible RNG: mimic JAX fold_in by hashing (epoch, thread_id) into `mx.random.key` seeds; avoid storing perturbations; reconstruct A,B per call.
- Low-rank forward pass: implement xB then outer with Aᵀ; cache base activations; use `mx.einsum`/`@` with small rank; avoid materializing E. For rank1, use `(x @ B)[:, None] * A[None, :]`.
- Antithetic sampling: pair perturbations via sign flips; store thread_id parity to negate A,B (nano-egg uses ±1). Pair fitnesses before update.
- Fast fitness: sign of paired reward difference; thresholded update for integer-style step not necessary unless matching nano-egg exactly; can start with standard ES averaging, add sign option if needed.
- Data pipeline: no JAX datasets; use numpy/pyarrow to stream tiny text/csv; for minipile-style uint8 tokens, load to `mx.array` and slice batches; keep sequence length small for Mac memory.
- Integer-only path: MLX int8 matmuls exist but training likely easier in float32; if exploring integer fidelity, mirror nano-egg fixed-point scaling with bitshifts and clip to int8.

## Gaps vs Paper (EGGROLL intent)
- Paper emphasizes communication/computation savings with low-rank perturbations for very large models/populations; current tests are small-scale. Need a proxy metric (bandwidth = params vs rank) and possibly a larger synthetic model.
- Need variance reduction tricks (antithetic sampling, baselines) and compare low-rank vs full ES directly (added initial hi-dim test).
- Should measure wall-clock/token throughput per update to see practical speedups, even on Mac.

## Next Experiments to Align Better
- Expand low-rank vs full ES on a larger synthetic model (higher d) and record both convergence and approximate bandwidth (rank* (rows+cols) vs full params).
- Add antithetic sampling / reward baselines to reduce variance; compare.
- Track compute time per step (even rough) to relate to efficiency claims.

## Reviewer Prompts
- Does low-rank noise preserve exploration in high dimensions? Any mode collapse?
- How sensitive is performance to rank `r` and noise scale?
- Are communication savings clear in the benchmark setup?

## Links/Artifacts
- Will add marimo notebook: `notebooks/eggroll_claims.py` (stub in repo).
