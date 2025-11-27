# Findings

# Findings

## Recent ES observations (fp16 path)
- Hyperparameters are very finicky; Optuna sweeps and manual tuning gave sensitive results without a stable “best” set.
- Full-pop forward scales linearly; pop 5120 takes ~3s/step on M-series, so large pops are slow.
- Convergence is not clearly demonstrated; loss/reward often bounce or stay near baseline, and reward ranges can stay flat.
- Rank 1 low-rank updates did not converge; higher rank helps but remains noisy.
- The same GRU architecture with standard backprop/AdamW converges quickly to ~1e-3 loss on the count task.

## 1. Threshold Scaling and Update Rate
We investigated why the model was not updating (0% update rate) and found two key issues:

### A. Threshold Scaling Logic
The original implementation missed a scaling factor present in the reference code (`nano-egg`).
- **Reference Logic:** `Effective Threshold = Base Threshold * sqrt(Pop) * (2^FixedPoint if UseCLT else 2^-FixedPoint)`
- **Our Logic (Fixed):** We updated `eggroll_mlx.py` to match this.

### B. Base Threshold Value
The default `update_threshold` of **512** was too high.
- **Analysis:** `update_threshold` represents the Z-score cutoff scaled by `2^FixedPoint`.
    - `512 / 16 = 32`. This corresponds to a **32-sigma** cutoff.
    - The probability of a random signal exceeding 32 sigma is effectively zero.
- **Reference Value:** The reference starts with `update_threshold = 0` and anneals it up to ~30 (approx 2 sigma).
- **Conclusion:** We must lower `update_threshold` to a reasonable range (e.g., **32**, which is 2 sigma) to allow updates.

## 2. Steps Per Second (SPS) Benchmark
We benchmarked the training loop speed at different population sizes with `seq_len=16`.

| Pop Size | Time/Step | SPS | Tokens/Sec | Speedup |
| :--- | :--- | :--- | :--- | :--- |
| **1,024** | 0.19s | **5.35** | 87.7k | 1.00x |
| **32,768** | 5.42s | **0.18** | 96.7k | 1.10x |
| **262,144** | 47.27s | **0.02** | 88.7k | 1.01x |

**Result:** We confirmed linear scaling up to 262k populations, achieving the target scale on a single GPU.

## 3. Update Mechanics
The core of the EGGROLL algorithm (Evolutionary Gradient Generation via Low-Rank Perturbations) works as follows:

1.  **Perturbation (Forward Pass):**
    *   Each parameter matrix $W$ is perturbed by a low-rank noise term: $W' = W + \sigma (A \times B^T)$.
    *   $A$ and $B$ are slices from a large, shared, fixed random tensor (`BIG_RAND`), determined deterministically by `(epoch, thread_id)`.
    *   This allows each member of the population (thread) to explore a different direction in parameter space without storing unique weights.

2.  **Evaluation:**
    *   The population runs the forward pass with these perturbed weights.
    *   We calculate the loss (negative reward) for each member.
    *   We convert these rewards into **fitness scores** (e.g., +1 if better than peer, -1 if worse).

3.  **Gradient Estimation (Update Pass):**
    *   We estimate the "gradient" by aggregating the correlation between the noise direction and the fitness.
    *   Signal $Z = \sum_{i=1}^{Pop} \text{Fitness}_i \times (A_i \times B_i^T)$.
    *   Intuitively, if a specific noise direction $A \times B^T$ consistently leads to higher fitness, $Z$ will be large and positive. If it leads to lower fitness, $Z$ will be large and negative. If uncorrelated, $Z$ will be near zero (random walk).

4.  **Gating and Update:**
    *   **Noise Gating:** We only update if the signal $|Z|$ exceeds a **Threshold**.
        *   $|Z| > \text{Threshold} \implies \Delta W = \text{sign}(Z)$.
        *   $|Z| \le \text{Threshold} \implies \Delta W = 0$.
    *   This acts as a high-pass filter, ignoring random noise and only stepping in directions with strong consensus from the population.
    *   **Crucial Tuning:** The Threshold must be scaled by $\sqrt{\text{Pop}}$ to match the growth of the random signal $Z$. If set too high (e.g., 512 for small pop), no updates occur. If set correctly (e.g., ~2 sigma), valid gradients pass through.

## 4. Datatypes and Locations
We confirmed that the implementation uses **int8** for both weights and activations to maximize performance on Apple Silicon (M-series) hardware.

*   **Weights:** Stored as `int8` in `BlockWeights` (e.g., `gru_x.weight`, `mlp_in.weight`).
*   **Activations:**
    *   **LayerNorm:** `l1_layernorm_int8` (scripts/train_tinystories_gru_full.py:70) normalizes and re-quantizes to `int8`.
    *   **MatMul:** `matmul_with_noise` (scripts/train_tinystories_gru_full.py:241) accepts `int8` inputs and returns `int8` outputs.
    *   **Residuals:** Clipped to `[-127, 127]` and cast to `int8`.
    *   *Note:* Non-linearities (sigmoid, tanh) use `float32` for precision but are immediately quantized back to `int8`.

## 5. Performance Estimates
Measured and estimated performance for different population sizes (SeqLen=16):

| Pop Size | Time/Step | SPS | Tokens/Sec | 5-Min Capacity |
| :--- | :--- | :--- | :--- | :--- |
| **1,024** | 0.19s | **5.35** | 87.7k | ~1600 steps |
| **10,240** | 1.69s | **0.59** | 97.0k | **~177 steps** |
| **32,768** | 5.42s | **0.18** | 96.7k | ~55 steps |
| **262,144** | 47.27s | **0.02** | 88.7k | ~6 steps |

**Conclusion:** We can run a meaningful experiment (~177 steps) with **10k population** in 5 minutes.

## 6. Memory Analysis
We identified a significant memory spike caused by the `cross_entropy` calculation during the forward pass.
*   **Cause:** Broadcasting of `logits`, `one_hot` targets, and `log_probs` tensors.
*   **Formula:** Peak Memory $\approx 3 \times (\text{GroupSize} \times \text{BatchSize} \times \text{VocabSize} \times 4 \text{ bytes})$.
*   **Measurements (Batch=64, Vocab=10k):**
    *   **Group=1024:** Peak **7.87 GB** (Input 2.62 GB).
    *   **Group=128:** Peak **0.99 GB** (Input 0.33 GB).
*   **Optimization:** Reducing `group_size` to 128 eliminates the memory bottleneck without affecting the mathematical correctness of the large population training.

## 7. Performance Bottleneck Analysis (Comprehensive Report)
We investigated the slow training speed (~60s/step) observed with `group_size=128`.

### 7.1. Observation
*   **Target:** ~0.2s per step (for 10k population).
*   **Actual:** ~60s per step.
*   **Forward Pass Time:** ~0.8s - 1.6s per chunk (128 population).

### 7.2. Detailed Profiling Breakdown
We instrumented the code to measure the exact time spent in various operations.
*   **Total Time Loop (per chunk):** ~1.62s
*   **`int8_matmul` Kernel Execution:** **0.94s** (58% of total time).
    *   **Call Count:** 60 calls per chunk (16 time steps * layers).
    *   **Average Time per Kernel:** **15.7ms**.
*   **Overhead (Setup/Reshape):** < 0.005s (Negligible).
*   **Other Ops (Layernorm, Activations):** ~0.6s (Remaining 42%).

### 7.3. Root Cause
The bottleneck is the **pure execution latency of the `int8_matmul` kernel** on Metal when running with small batch sizes (`2048 x 256`).
*   **GPU Underutilization:** The grid size for `group_size=128` is too small to fully saturate the GPU, leading to execution being dominated by kernel launch latency and memory overhead rather than compute.
*   **Inefficiency:** Running 128 items takes ~1.6s. Running 512 items would likely take a similar amount of time (amortized), making it 4x more efficient.

### 7.4. Conclusion
*   **Memory vs. Speed Trade-off:**
    *   `group_size=1024`: Fast but crashes memory (14GB peak).
    *   `group_size=128`: Memory safe (1GB peak) but extremely slow due to kernel latency.
*   **Recommendation:** To achieve the target speed, we must increase `group_size` (e.g., to 512 or 1024) to amortize the kernel overhead. However, this requires careful memory management or device upgrades to handle the increased activation memory.

## 8. NAX exploration (MLX)
* NAX kernels in MLX are only instantiated for float/float16/bfloat16; quantized matmul dequantizes packed weights to float activations and returns float. No int8 activation/int32 accumulate path exists in NAX.
* Gates: requires arch gen ≥17 (M3/A17+), OS runtime ≥15.2, K % 64 == 0 for the transpose path; float32 needs `MLX_ENABLE_TF32=1`. Otherwise MLX falls back to non-NAX kernels.
* Quantized NAX kernels build for bits 2–8 with group sizes 32/64/128, but activations/output stay float.
* Benchmarks (bulk, single-sync): `quantized_matmul` float16, bits=8, group_size=64 showed ~0.05 ms / 42 TOPS for 1024×1024×1024 and ~0.5 ms / 68 TOPS for 4096×4096×1024 when NAX is eligible. Misaligned shapes still ran fast; backend may still pick optimized kernels even with TF32 off.
* Forcing TF32 off (float32 + `MLX_ENABLE_TF32=0`) lowers throughput on some shapes (e.g., 1024×1024×1024 → ~0.17 ms / 12.6 TOPS) but remains high on others; there’s no public flag to hard-disable NAX beyond shape/dtype gating.
* Heuristic device check: added a tiny extension (`extensions/nax_check`) exposing `nax_check.is_nax_available()` via Metal GPU family (≥9 → True). On Apple M4 Pro it returns True.
* Custom ops: you can reuse Steel/NAX headers in C++/Metal for float/bfloat matmul/attention; no integer fragments exist, so int8 activations on NAX are unsupported with current headers.
* Open investigations:
  - Build MLX from source with emitted metallibs and run `metal-disassemble` to enumerate kernels/symbols. Current evidence shows no integer NAX support.
  - Scaffolded a custom Metal GEMM extension (`extensions/nax_gemm`), but `xcrun metal` is missing in the current environment, so the metallib isn’t built and the host fails to load it. Need to install Xcode CLTs and rebuild; also replace the naive kernel with a NAX-tiled matmul using `steel/gemm/nax.h`.
