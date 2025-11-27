# ES Convergence Investigation Findings

## Objective
Verify that the EGGROLL Evolutionary Strategies (ES) implementation can solve a simple linear regression task ($Y = W X$) to rule out core algorithm bugs before debugging the full TinyStories training.

## Experimental Setup
- **Task**: Linear Regression ($d_{in}=64, d_{out}=16$).
- **Optimizer**: EGGROLL (Sign-based or CLT).
- **Metric**: MSE Loss, Distance to Target Weights ($||W - W_{target}||$).
- **Test Script**: `tests/test_es_convergence.py`.

## Key Findings

### 1. Rank is Critical
- **Observation**: With `rank=1`, convergence was extremely slow or non-existent, even with "perfect fitness" (injected gradient signal).
- **Result**: Increasing `rank` to 4 or 16 significantly improved the rate of descent in distance to target.
- **Conclusion**: `rank=1` is likely too restrictive for this high-dimensional optimization, or requires much larger population/steps.

### 2. Noise Scaling & Precision
- **Observation**: Initial tests with `float16` and `sigma_shift=4` showed noise magnitude ~1200x larger than signal in the update accumulator.
- **Observation**: `float16` caused gradient calculation overflows (`inf`) in the test script (fixed by casting to `float32`).
- **Result**: Switching to `float32` for the test logic was necessary for reliable debugging.
- **Recommendation**: Ensure `float32` is used for critical accumulators in the main training loop if possible, or carefully manage scaling.

### 3. Population Size
- **Observation**: Small populations (128) resulted in noisy gradient estimates (Cosine Similarity ~0).
- **Result**: Increasing `pop_size` to 2048 improved the consistency of the update direction.
- **Conclusion**: ES requires large populations to estimate the gradient effectively in high dimensions.

### 4. CLT vs. Sign-Based
- **Observation**: The "CLT" mode (`use_clt=True`, `fast_fitness=False`, `sigma_shift=0`) provided a stronger signal than the default sign-based mode.
- **Result**: This configuration showed the best decrease in distance to target (~32.6 -> ~26.9).

### 5. Fitness Correlation Anomaly
- **Observation**: We observed a **negative correlation** (~ -0.2) between "Perfect Fitness" (derived linearly from true gradient) and actual MSE fitness (`-mse`).
    - *Perfect Fitness* predicts descent direction.
    - *Negative Correlation* implies that steps predicted to improve loss actually worsened it (or vice versa).
- **Hypothesis**: This typically indicates **overshooting** (learning rate or noise magnitude too large) or **non-linearity** dominating the local linear approximation.
- **Action**: We tested `sigma_shift=2` to reduce noise, but correlation remained negative. This suggests `learning_rate` might still be too high for the effective noise scale.

## Best Configuration So Far
The configuration that showed the most promise (decreasing distance to target) was:
```python
cfg = NoiseConfig(
    rank=4,
    sigma_shift=0, # or 2
    fixed_point=4,
    learning_rate=0.01,
    weight_clip=3.0,
    fast_fitness=False,
    update_threshold=0,
    use_clt=True
)
pop_size = 2048
```

## Next Steps
1.  **Tune Learning Rate**: Reduce LR further to fix the negative fitness correlation (stop overshooting).
2.  **Apply to Training**: Port the "CLT" configuration (`rank=4`, `use_clt=True`, `pop_size` increase) to the main `train_tinystories_gru_full.py` script.
3.  **Verify FP16 Safety**: Ensure the main script handles `float16` without the overflows seen in the test (likely by keeping accumulators in `float32`).
