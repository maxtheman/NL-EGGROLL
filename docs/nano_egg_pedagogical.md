## Building Nano-Egg From Scratch (Pedagogical Guide)

This is a plain-language walkthrough for re-implementing the core nano-egg idea (ES with low-rank perturbations and optional int8 fixed-point). The low-rank Eggroll update works in any dtype (fp32/fp16/bf16); nano-egg’s reference happens to use int8 for speed and hardware efficiency. Use alongside `third_party_references/repos/nano-egg/run.py` for code details.

### 1) Core ideas to keep in mind
- Precision is flexible: the Eggroll update works in fp16/bf16/fp32; int8 is an implementation choice for maximal throughput. In int8 mode, store weights/activations as int8 (scale 2^4) and accumulate in int32.
- Low-rank ES noise: perturb each weight matrix W with A (out×r) and B (in×r); delta = (A @ B^T) / 2^(FIXED_POINT + sigma_shift) in fixed-point, or just scale by sigma in floating point.
- Antithetic pairing: even/odd threads share A,B but flip sign to reduce variance.
- Deterministic noise lookup: precompute one giant noise table (`BIG_RAND_MATRIX`) and slice by (epoch, thread_id, parameter) so you never store per-member perturbations.

### 2) Data pipeline
- Use a byte-level dataset (vocab=256). Nano-egg uses minipile; each sequence is an array of uint8 with a 0 prepended as BOS.
- Store datasets as `.npy` for fast mmap loading; reshape to batches `[num_sequences, tokens_per_sequence]`.

### 3) Model architecture (int8-friendly)
- Embedding: int8 table `vocab_size × hidden_dim`.
- Block (repeat `n_layer` times):
  - LN variant (`EGG_LN`): L1-style normalization implemented via lookup table `DIVISION` to avoid sqrt.
  - GRU-ish cell (`EGG_GRU`): gates computed with int8 matmuls + bias; nonlinearities are implicit via clipping/saturation.
  - Residual add with `clipped_add`.
  - MLP: linear expand (hidden → 4×hidden) then project back (4×hidden → hidden), both int8 matmul with low-rank noise.
- Output: LN + linear head to vocab logits (still int8 logits).
- State: per-layer hidden state kept as int8; `default_state` is zeros.

### 4) Noise and ES plumbing
- Generate per-parameter RNG keys deterministically:
  - Build `scan_map` over params; call `simple_es_tree_key` to split a base key into per-parameter trees.
  - For a given (epoch, thread_id), `get_common_start_idx` picks a slice start in `BIG_RAND_MATRIX` and an antithetic sign.
- Low-rank perturbation:
  - Slice `(a+b)*r` int8s → reshape to B (b×r), A (a×r).
  - During forward: `x @ W.T` (int32 accumulate) + `(x @ B) @ A.T` shifted and clipped to int8.
- Full-parameter perturbation (for non-matrix params): slice `param.size*2`, reshape to `param.shape + (2,)`, take product along last dim to get signed noise.

### 5) Forward pass with noise
- Each thread processes its token slice; `iterinfo=(epoch, thread_id)` enables noise injection.
- For every linear/embedding/parameter access, call the noiser’s `do_mm`, `do_emb`, or `get_noisy_standard` to add low-rank/full noise on-the-fly.
- Compute token log-likelihood with precomputed EXP2/LOG2 tables (`get_int_ll`), staying in fixed-point.

### 6) Fitness and updates
- Run population members in parallel (`parallel_generations_per_gpu` threads). Antithetic pairing halves distinct perturbations.
- Fitness = summed int log-likelihood over each member’s token window.
- Convert paired scores to signs (`convert_fitnesses`); optionally CLT-weighted if `use_clt=False`.
- Apply updates:
  - For each param, dispatch to full or low-rank updater via `es_map`.
  - Update is sign-step ±1 (or gated) per weight; low-rank votes computed as sum over `score * A @ B^T`.
  - Clip back to int8 bounds.

### 7) Efficiency tricks
- Do not materialize E; reconstruct A and B per batch from the noise table.
- Batch all threads: vmapped `generate_thread` (forward) and jitted `do_updates`.
- Avoid recomputing noise: `noise_reuse` lets multiple epochs share slices.
- Keep `group_size` > 1 to reduce unique perturbations when data is short.

### 8) Minimum viable config to test
- Hidden=256, layers=6, vocab=256, rank=1–2, sigma_shift=4, FIXED_POINT=4.
- `tokens_per_update=100`, `parallel_generations_per_gpu=128` (or more if memory allows).
- Population fitness sign updates; `alpha` schedule → z-threshold for gating.

### 9) What to log
- Throughput (tokens/s), generate vs update time.
- Min/mean/max fitness per epoch; parameter change rates (LoRA vs non-LoRA) if you split `es_map`.
- Validation bits/byte if you hold out a shard.

### 10) How it differs from egg.c
- egg.c generates noise on-the-fly with xorshift; nano-egg slices from a giant precomputed table.
- egg.c uses NEON hand-tuned matmul; nano-egg leans on JAX/XLA vmaps/scans.
- Both use int8 fixed-point, antithetic pairs, rank-1 perturbations, and sign-step updates gated by a threshold.
