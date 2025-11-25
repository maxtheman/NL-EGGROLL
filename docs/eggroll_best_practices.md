# EGGROLL Implementation Best Practices

This document summarizes key learnings and best practices for implementing the EGGROLL (Evolutionary Gradient-Guided Representation Learning) algorithm, based on debugging and optimizing the `train_tinystories_gru_full.py` script.

## 1. Numerical Stability in Int8

The core challenge of EGGROLL is maintaining stability while training with `int8` weights and activations.

### L1 LayerNorm is Critical
Standard LayerNorm divides by the standard deviation ($\sigma$). In `int8` arithmetic, small variances can lead to division by zero or value explosion, causing saturation (`127` or `-127`).
*   **Best Practice:** Use **L1 LayerNorm** (normalize by mean absolute value).
*   **Formula:** $x_{norm} = \frac{x}{\text{mean}(|x|) + \epsilon} \times \text{scale}$
*   **Why:** It avoids square roots and squaring, which are risky in low-precision integer math, and provides a more stable scaling factor.

### Linear Scaling for Matrix Multiplication
Standard initialization assumes weights have unit variance and scales outputs by $1/\sqrt{d_{model}}$.
*   **Problem:** To avoid saturation with `int8` activations (range +/- 127), weights must be kept very small (~1).
*   **Consequence:** Integer updates of +/- 1 become destructive (100% relative change), preventing convergence.
*   **Best Practice:** Scale outputs by **$1/d_{model}$** (linear) instead of $1/\sqrt{d_{model}}$.
*   **Benefit:** This allows weights to be larger (e.g., ~16), so a +/- 1 update represents a fine-grained change (~6%), enabling stable learning.

### Initialization Scale
With linear scaling, you can and *should* initialize weights to a larger range.
*   **Best Practice:** Set `init_scale` such that weights are around **16-32**.
*   **Example:** `init_scale=16.0` with `fixed_point=4`.

## 2. Scaling to Large Populations

EGGROLL's efficiency comes from "pure batch inference," where the population dimension is treated as a batch dimension.

### Grid Size Limits
When `pop_size` is large (e.g., 262k), the total number of operations in a single kernel launch can exceed the 32-bit signed integer limit ($2^{31} - 1$).
*   **Symptom:** `TypeError` or kernel failures when `m * n > 2B`.
*   **Best Practice:** Implement **chunking** in your custom kernels (e.g., `int8_matmul`). Split the grid along the largest dimension to ensure each launch stays within limits.

### Memory Management
Generating noise for massive populations can cause OOM errors.
*   **Best Practice:** Cap the `param_span` (noise table size) to a safe limit (e.g., 256MB) and wrap around indices.
*   **Why:** The randomness quality doesn't degrade significantly for training, but it prevents crashing the GPU during initialization.

### Throughput Scaling
*   **Observation:** Throughput (tokens/sec) scales linearly with population size until the GPU is saturated.
*   **Result:** `pop_size=32k` can be *faster* (in tokens/sec) than `pop_size=1k` due to better GPU utilization.

## 3. Algorithm Tuning

### Update Threshold
The `update_threshold` gates small noise contributions.
*   **Best Practice:** Scale `update_threshold` with $\sqrt{\text{pop\_size}}$.
*   **Reason:** The sum of noise vectors grows with population size. A fixed threshold (e.g., 32) that works for `pop_size=64` will be useless (always exceeded) for `pop_size=1000`.

### Fast Fitness
*   **Best Practice:** Use **Sign-based Fitness** (Antithetical Sampling).
*   **Logic:** `sign(fitness_pos - fitness_neg)`. This simplifies the signal to "better" or "worse," which aligns well with the coarse +/- 1 integer updates of the weights.
