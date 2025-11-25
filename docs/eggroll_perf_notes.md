## Eggroll perf/quant notes

**What we tried**
- Benchmarked MLX int8 Metal kernel vs MLX `nn.QuantizedLinear`; QuantizedLinear is much faster (e.g., 4096x4096x512: int8_matmul ~62.6ms/0.27 TOPS vs QuantizedLinear ~9.16ms/1.88 TOPS).
- CoreML matmul (int8 inputs dequantized to fp16, target iOS18): similar or slightly better TOPS (e.g., 4096x4096x512: ~7.79ms/2.21 TOPS) but uses fp16 I/O and predict overhead.
- MPSGraph int8 matmul is unsupported (verification error: matmul requires float/complex).
- Updated int8 kernel back to simple version to keep eggroll tests green; tests now pass.

**Current best path**
- Use MLX `nn.QuantizedLinear` (int8 weights, float activations) for speed. Keep int8 weights for bandwidth/storage savings; activations remain float.

**Next steps (not yet implemented)**
- Refactor eggroll `do_mm`/trainer to:
  - Use QuantizedLinear for base matmul.
  - Add low-rank noise as a separate delta matmul (x@B, then @A^T) per pop member (batched), instead of in-kernel noise.
  - Give each pop member unique `thread_id`/noise slice and apply sign updates to all matrices (embeddings, Q/K/V/out, MLP, LM head).
  - Calibrate per-matrix scaling to keep logits in range; avoid double divides since QuantizedLinear handles weight dequant.
- Parallelize population by batching pop dimension into one forward pass.

**Tests**
- `NANO_EGG_NO_CLI=1 PYTHONPATH=. uv run python -m pytest tests -q` → 6 passed, 1 skipped (after reverting kernel to simple version).
