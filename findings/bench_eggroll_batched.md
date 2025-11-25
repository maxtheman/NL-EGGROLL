## Batched vs per-thread eggroll matmul (MLX)

Command examples (run from repo root):
```
PYTHONPATH=. uv run python scripts/bench_eggroll_batched.py --iters 5
PYTHONPATH=. uv run python scripts/bench_eggroll_batched.py --iters 5 --use-quant
```

### Results (B=64, N=256, K=256, pop=8, rank=1)
- `use_quant=False` (int8 kernel): per-thread 2.17 ms/iter; batched 6.70 ms/iter; outputs match; batching slower (extra reductions, no fused int8 stack).
- `use_quant=True` (QuantizedLinear-style base, weight-only int8, group_size=32): per-thread 11.35 ms/iter; batched 0.49 ms/iter; outputs match; ~23× speedup.

### Notes
- Batched path shares activations and applies low-rank deltas via broadcasted int32 reductions; correctness matches per-thread path.
- Quantized base uses weight-only affine quantization to int8 (bits=8, group_size=32) with float activations; int32 casts are accumulation only.
- Full `pytest` is blocked by third_party nested_learning torch deps; parity suite passes (`uv run pytest tests/test_eggroll_parity.py -q`).***
