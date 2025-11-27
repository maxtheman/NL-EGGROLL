## FP16 vs INT8 matmul on M-series (MLX Metal)

Benchmarked with MLX `mx.matmul` (fp16 weights) vs `mx.quantized_matmul` (int8 weights + scales), shapes matching the TinyStories GRU trainer (d_model=H=64, vocab=10k, batch=80). Timings are per call, averaged over a few iterations.

Group size controls population chunking; M = group_size * batch.

- Group=100 (M=8,000):
  - `gru_x` (192x64): fp16 0.10 ms, int8 0.11 ms → 0.94x
  - `gru_h` (192x64): fp16 0.07 ms, int8 0.07 ms → 1.01x
  - `mlp_in` (256x64): fp16 0.08 ms, int8 0.08 ms → 0.96x
  - `mlp_out` (64x256): fp16 0.12 ms, int8 0.15 ms → 0.76x
  - `head` (10000x64): fp16 0.62 ms, int8 1.50 ms → 0.42x

- Group=200 (M=16,000):
  - `gru_x`: fp16 0.05 ms, int8 0.05 ms → 1.02x
  - `gru_h`: fp16 0.05 ms, int8 0.05 ms → 0.95x
  - `mlp_in`: fp16 0.05 ms, int8 0.05 ms → 0.99x
  - `mlp_out`: fp16 0.16 ms, int8 0.06 ms → 2.72x
  - `head`: fp16 1.11 ms, int8 10.34 ms → 0.11x

- Group=400 (M=32,000):
  - `gru_x`: fp16 0.07 ms, int8 0.08 ms → 0.93x
  - `gru_h`: fp16 0.07 ms, int8 0.08 ms → 0.91x
  - `mlp_in`: fp16 0.17 ms, int8 0.09 ms → 1.89x
  - `mlp_out`: fp16 0.09 ms, int8 0.09 ms → 1.03x
  - `head`: fp16 1.48 ms, int8 11.91 ms → 0.12x

Takeaway: the large head matmul (K=64, N=10k) dominates and is far slower in int8; most other mats are a wash. Net speed favors fp16 for this workload. Hybrid quantization isn’t worth the complexity here.
