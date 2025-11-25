# Findings

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
