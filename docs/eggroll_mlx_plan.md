## Eggroll MLX: Parity + Unified Memory Plan

Status:
- Parity tests vs nano-egg are green (`pytest tests/test_eggroll_parity.py`).
- Mini head-to-head benchmark runs:
  - `eggroll` (low-rank): Acc=0.7379, Loss=5.2508, Time=1.040s, RSS=254.6 MB (pop=32, steps=50, rank=1).
  - `full_es`: Acc=0.4757, Loss=0.6931, Time=0.746s, RSS=192.5 MB (same config).

Plan to simplify with unified memory and int matmul:

1) Vendor int8 matmul/dot kernels
   - Add a Metal kernel for int8 matmul/dot so forward/updates stay int-native without float casts.
   - Provide a small wrapper (e.g., `int8_matmul(a, b) -> int32`) and use it in `eggroll_mlx` for perturbation/recon where MLX matmul rejects ints.

2) Mixed-stream execution (no data moves)
   - Run dense forwards on `mx.gpu`, small reductions/updates on `mx.cpu` using streams; let unified memory handle dependencies.
   - Benchmark variants: gpu-only, cpu-only, mixed streams; log time/RSS.

3) Shared noise table
   - Keep one `BIG_RAND` in unified memory; slice on whichever device executes; drop any copy/placement code.

4) Cleanup float fallbacks
   - Remove float-cast matmuls once int8 kernel is in place for delta reconstruction and update accumulation.

5) Bench improvements
   - Extend `scripts/eggroll_mlx_head_to_head.py` to:
     - toggle int8 kernel vs float path,
     - choose stream placement (cpu/gpu/mixed),
     - report RSS/time/accuracy.

6) Integration into exp_3
   - Swap in `eggroll_mlx` primitives (int8 matmul, deterministic noise) in `scripts/exp_3_batched_eggroll.py`.
   - Add flags to enable int8 kernels and mixed-stream execution.
