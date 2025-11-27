# NL-EGGROLL

Minimal MLX/Metal exploration of the EGGROLL paper (“Evolution Guided General Optimization via Low-rank Learning”). This fork focuses on a few runnable scripts only. Because MLX on Metal lacks an int8 activation + int32 accumulate path, the current implementation forces fp16 activations/weights in the critical matmuls. That deviation likely reduces numerical stability, and I have not been able to reproduce the paper’s int8 results.

## Installation (uv)
```bash
# Create and enter a virtual environment
uv venv .venv
source .venv/bin/activate

# Install dependencies (MLX requires macOS + Apple Silicon)
uv pip install mlx optuna numpy
```

## Run instructions (reproductions)

### 1) ES trainer (fp16) on TinyStories-count
Uses low-rank ES with fp16 matmuls (no int8 path on MLX).
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
Status: runs end-to-end on M-series, but loss can stagnate; fp16 path differs from the paper’s int8 reference.

### 2) Baseline GRU with AdamW (count task)
Backprop baseline matching model geometry; requires count data.
```bash
# Generate toy count data (writes to data/count)
PYTHONPATH=. uv run python scripts/make_count_data.py --out_dir data/count

# Train baseline
PYTHONPATH=. uv run python scripts/train_count_gru_adamw.py \
  --data_dir data/count \
  --vocab_size 16 \
  --seq_len 32 \
  --d_model 64 \
  --d_hidden 64 \
  --layers 1 \
  --batch_size 64 \
  --steps 500 \
  --lr 1e-3
```

### 3) Hyperparameter tuning with Optuna (ES)
Lightweight sweep over a few ES hyperparameters on the count task.
```bash
PYTHONPATH=. uv run python scripts/tune_eggroll_optuna.py \
  --data_dir data/count \
  --vocab_size 16 \
  --seq_len 32 \
  --d_model 64 \
  --d_hidden 64 \
  --layers 1 \
  --batch_size 64 \
  --steps 100 \
  --pop_size 256 \
  --fixed_point 4 \
  --noise_reuse 1 \
  --weight_clip 3.0
```

## What’s inside
- `eggroll_mlx.py`: MLX implementation of low-rank ES primitives.
- `eggroll_api.py`: Convenience wrapper for noise/context handling.
- `scripts/train_tinystories_gru_full.py`: ES trainer (fp16 path on MLX).
- `scripts/train_count_gru_adamw.py`: AdamW baseline for the count task.
- `scripts/tune_eggroll_optuna.py`: Small Optuna tuner for ES hyperparams.
- `scripts/train_tinystories_gru_int8.py`: Support code used by the ES trainer (still executes with fp16 matmuls under MLX).

## Current status and caveats
- MLX/Metal has no int8 activation + int32 accumulate kernel; `mlx.quantized_matmul` dequantizes to float, so this code runs fp16 and cannot match the paper’s int8 path.
- Large populations and higher rank help signal quality, but convergence remains fragile on M-series.
- Memory can spike with large group sizes; adjust `group_size` if you hit OOM.
- Rank=1 struggled even more than rank=4 in my runs, and the update rate was hard to calibrate. Example log at pop=large:
  ```
  step 84: loss=4.5791, logits_max=2.5, reward_mean=-3.015, reward_std=0.199
    reward_range=[-4.197, -2.610], update_rate=100.0000% (62544/62544)
    update_threshold=0
    TIMING: Batch=0.000s, Forward=0.262s (1 chunks, 0.262s/chunk), Concat=0.000s, Update=0.002s, Log=0.007s, Total=0.271
  ```
  Large population runs fit in memory, but stable learning remained elusive with rank=1; rank=4 behaved better but still fragile.
