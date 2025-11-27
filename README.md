# NL-EGGROLL

Unofficial MLX/Metal exploration of the EGGROLL paper (“Evolution Guided General Optimization via Low-rank Learning”) using low-rank ES updates for int8-heavy RNN/GRU language models. This repo mirrors core ideas from the paper but adapts kernels and scaling to Apple Silicon; those changes appear to reduce numerical stability, and I have not been able to reproduce the headline results.

## Current status
- Training runs on M-series hardware with int8 weights/activations, but convergence is fragile and often stalls.
- Metal/MLX lacks an int8 activation + int32 accumulate path (no NAX integer fragments), forcing implementation differences from the reference that likely hurt stability.
- Threshold logic fixed to match the reference; large populations and higher rank improve signal, but reproducible wins remain elusive.

## Quick start (what’s been working best)
```bash
PYTHONPATH=. uv run python scripts/train_tinystories_gru_full.py \
  --data_dir data/count \
  --vocab_size 16 \
  --steps 500 \
  --batch_size 64 \
  --pop_size 512 \
  --group_size 512 \
  --rank 4 \
  --seq_len 32 \
  --layers 1 \
  --d_model 64 \
  --d_hidden 64 \
  --fast_fitness 0 \
  --fitness_alpha 0.1 \
  --update_threshold 1 \
  --learning_rate 0.005 \
  --weight_clip 3.0 \
  --sigma_shift 4 \
  --checkpoint ckpts/count.ckpt
```
Note: even with these settings, loss curves can bounce and sometimes fail to improve.

## Key observations
- **Population size & rank:** Rank=1 rarely converges; rank 4–16 with pop ≥2k gives a cleaner signal but still noisy.
- **Threshold scaling:** Must scale by √pop and fixed_point; high thresholds (e.g., 512) effectively freeze updates.
- **Performance:** Forward scales linearly with population; group_size too small underutilizes the GPU, too large blows memory (cross-entropy broadcast).
- **Matmul kernels:** int8 `quantized_matmul` on Metal dequantizes to float; the large vocab head is slower in int8 than fp16, making fp16 generally faster overall.
- **NAX:** Current MLX builds expose NAX only for float/bfloat; no integer NAX path, so true int8 activation pipelines aren’t available.

## What’s inside
- `eggroll_mlx.py`: Metal/MLX implementation of low-rank ES matmuls and noise.
- `scripts/train_tinystories_gru_full.py`: GRU language model trainer adapted to EGGROLL-style updates.
- `findings/`: Benchmarks, scale sweeps, and debugging notes (see `findings.md` for a narrative).
- `EGGROLL.pdf`: Paper for the original method.

## Open issues / future work
- Explore alternative fitness scaling (raw rewards vs sign) for stability.
- Try mixed-precision accumulators or higher-rank noise to mitigate Metal quantization quirks.
- Revisit custom Metal kernels once integer NAX fragments exist or a custom metallib can be built.

