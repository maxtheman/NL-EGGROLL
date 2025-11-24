## Nano-Egg Architecture (with line refs)

- **Purpose & setup** — JAX-only evolutionary trainer; int8 fixed-point weights/activations; uses large precomputed RNG table and antithetic pairing for ES updates. Key args: population (`parallel_generations_per_gpu`), alpha/z-threshold schedule, rank, sigma shift, noise reuse, token chunking (`tokens_per_update`). `Args` and env setup: `third_party_references/repos/nano-egg/run.py:1-120`.

- **Model scaffold** — Generic model helpers build trees of params plus `scan_map` and `es_map` for ES classification; `simple_es_tree_key` makes per-parameter RNG keys; `merge_inits`/`scan_init` stack layers with scan-aware shapes. See `third_party_references/repos/nano-egg/run.py:123-195`.

- **Modules** — Param (`Parameter`), embedding (`Embedding`), matmul (`MM`), linear/MLP, fixed-point LayerNorm (`EGG_LN`), GRU cell (`EGG_GRU`), residual block (`LayerEGG`), and top-level language model (`EGG`) with emb → stack of GRU blocks (scan) → LN → head. Residual adds use `clipped_add` to stay within int8 range. References: `third_party_references/repos/nano-egg/run.py:222-418`.

- **Noise indexing & antithetic pairing** — Deterministic slice into `BIG_RAND_MATRIX` per (epoch, thread, param) via `get_common_start_idx`; even/odd thread flips sign for antithetic samples. Low-rank LoRA adapters pulled as contiguous slice reshaped into B (in×r) and A (out×r); full noise uses paired int8 multipliers. See `third_party_references/repos/nano-egg/run.py:424-457`.

- **ES update rules (QEggRoll)** — Two paths: full-parameter (`_simple_full_update`) and low-rank (`_simple_lora_update`). Scores are pairwise sign (fast fitness) or CLT-weighted; updates are sign steps ±1 on int8 params, gated by per-epoch `update_threshold` and optional CLT scaling. `_noop_update` for frozen params. Update dispatcher uses `es_map` to choose full vs LoRA vs noop. References: `third_party_references/repos/nano-egg/run.py:458-567`.

- **Forward-time noise injection** — Matmul/embedding/parameter accessors (`do_mm`, `do_emb`, `get_noisy_standard`) add perturbation during forward if `iterinfo` is present: base int8 op + low-rank perturbation shifted by `FIXED_POINT + sigma_shift`, then clipped. All math uses int32 accumulators, returns int8. Lines: `third_party_references/repos/nano-egg/run.py:496-538`.

- **Fixed-point log-likelihood** — Precomputed EXP2/LOG2 tables convert int logits to approximate log2 likelihood per token (`get_int_ll`). Lines: `third_party_references/repos/nano-egg/run.py:569-590`.

- **Dataset prep** — `build_dataset` downloads JeanKaddour/minipile, encodes bytes with prepended 0 token, saves train/valid/test as `.npy`. Lines: `third_party_references/repos/nano-egg/run.py:754-781`.

- **Training loop (`run_evolution`)** — Loads or initializes params, builds `es_tree_key`, precomputes per-epoch alpha → z-threshold, seeds noiser (rank, sigma_shift, noise_reuse). Compiles vmapped `generate_thread` (per-thread ES rollout) and `jit_update` (ES param update). Data is chunked into `tokens_per_update`; `group_size` repeats sequences to reduce unique perturbations. Each epoch: slice dataset segment, generate raw scores across threads, sanitize NaNs, convert to fitness, apply ES update, log param change rates (LoRA vs non-LoRA), throughput, optional validation. Lines: `third_party_references/repos/nano-egg/run.py:592-752`.

## Community Reference: egg.c (C/NEON implementation)

- **Source & config** — Downloaded to `third_party_references/repos/egg.c/full_trained_egg.c`. Byte-level LM, int8 fixed-point (FIXED_POINT=4), GRU + MLP stack, population ES. Key hyperparams: vocab=256, hidden=512, layers=4, seq_len=4096, pop=128, sigma_shift=4, update_threshold=160. Lines: `third_party_references/repos/egg.c/full_trained_egg.c:10-52`.
- **Perturbed matmul** — `matmul_perturbed` generates rank-1 noise (A,B) from xorshift RNG per layer/step, computes `x@W + noise_sign * (xB * A)>>shift`, NEON-optimized dot products. Mirrors nano-egg low-rank perturbation but done on-the-fly without precomputed BIG_RAND. Lines: `third_party_references/repos/egg.c/full_trained_egg.c:108-177`.
- **Forward with noise** — `forward_pass` injects perturbations into GRU gates, MLP expand/project, and head; applies fixed-point LayerNorm; uses GRU-style residual block similar to nano-egg `LayerEGG`. Lines: `third_party_references/repos/egg.c/full_trained_egg.c:212-296`.
- **ES update** — `update_matrix` accumulates votes from antithetic pairs (fitness sign) using pre-transposed noise buffers; updates weights by ±1 if vote crosses `UPDATE_THRESHOLD`. Functionally analogous to nano-egg `_simple_lora_update`/`_simple_full_update` sign-step gating. Lines: `third_party_references/repos/egg.c/full_trained_egg.c:332-408`.
- **Sampling/loglik** — Fixed-point softmax/log2 via `EXP2_TABLE` + `log2_fixed`, same idea as nano-egg `get_int_ll`. Lines: `third_party_references/repos/egg.c/full_trained_egg.c:27-60`, `321-330`, `411-463`.
- **Training loop** — Reads raw bytes (`input.txt`), initializes params with noise, maintains per-population recurrent states, runs antithetic rollouts over dataset slices, updates per layer, reports loss/tokens per sec, samples text every 10 steps. Lines: `third_party_references/repos/egg.c/full_trained_egg.c:465-583`.

## Cross-reference (nano-egg vs egg.c)

- Fixed-point int8 everywhere; both use `FIXED_POINT=4` scaling and clip to [-127,127] (nano-egg: `MAX`; egg.c: `MAX_VAL/MIN_VAL`).
- Low-rank noise: nano-egg slices deterministic `BIG_RAND_MATRIX` with antithetic thread pairing (`get_common_start_idx`); egg.c generates A/B per call via xorshift seeds; both use rank-1 LoRA-style perturbations.
- Fitness/sign updates: nano-egg converts paired scores to sign (`convert_fitnesses`) and applies sign ±1 with z-threshold gating; egg.c uses `UPDATE_THRESHOLD` votes from population pairs before flipping a weight.
- Architecture: both use emb → GRU-ish block with LN and MLP → head; egg.c hardcodes NEON paths and fixed dims, nano-egg is JAX/scan-parametric.
- Data: nano-egg uses minipile bytes with BOS=0; egg.c streams raw bytes from `input.txt`. Both operate byte-level vocab=256.

## Quotes from EGGROLL (paper, pp.1–10)

- Abstract (p.1): “EGGROLL overcomes these bottlenecks by generating random matrices A∈Rm×r, B∈Rn×r with r≪min(m,n) to form a low-rank matrix perturbation AB⊤ … reducing the auxiliary storage from mn to (m+n) per layer … resulting in a hundredfold increase in training throughput for billion-parameter models at large population sizes, nearly reaching the throughput of pure batch inference.”
- Intro (p.2): “ES is highly amenable to scaling through parallelisation, since fitness evaluations are independent across population members and require only the communication of scalar fitnesses, which maps cleanly onto modern inference infrastructure and yields near-linear speedups on large clusters.”
- Low-rank batching (p.2): “When evaluating the fitness of members of multiple perturbations in parallel, EGGROLL batches a population of low-rank adapters and shares the base activations, enabling a single forward pass that applies all AB⊤ updates via specialized batched matrix multiplications.”
- Theory (p.6): “We provide a rigorous theoretical analysis … proving that EGGROLL updates converge to the full rank Gaussian ES updates at an O(1/r) rate.”
- Integer RNN result (p.8): “We develop a nonlinear RNN language model that operates purely in integer datatypes, and demonstrate that EGGROLL can stably pretrain this language model … Our largest population size is 2^18 = 262144 … while only requiring a single GPU to train.”
