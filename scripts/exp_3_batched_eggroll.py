"""
Real Batched Eggroll Implementation.
Demonstrates the "Forward Pass De-duplication" which is the key to local LLM training efficiency.

Logic:
1. We process the input batch X against the shared base weights ONCE.
2. We process X against the P different low-rank adapters (A_i, B_i) efficiently using Einsum.
3. We sum them to get P different outputs with minimal compute.

Usage:
  uv run python scripts/exp_3_batched_eggroll.py --steps 200 --pop_size 128
"""

import argparse
import time
from dataclasses import asdict, dataclass
from typing import List, Tuple, Optional

import mlx.core as mx
import mlx.nn as nn
import mlx.utils
import numpy as np
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split


@dataclass
class ExperimentConfig:
    samples: int = 1024
    noise: float = 0.1
    test_size: float = 0.2
    input_dim: int = 2
    hidden_dim: int = 32
    output_dim: int = 2
    steps: int = 200
    pop_size: int = 32  # We can push this much higher now!
    sigma: float = 0.05
    lr_es: float = 0.05
    rank: int = 2
    seed: int = 42
    int_mode: bool = True
    scale: float = 16.0


class EggrollLinearBatched(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, bias: bool = True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        # Fix initialization: Use LeCun/He initialization scale to prevent explosion
        scale = np.sqrt(1.0 / input_dim)
        self.weight = mx.random.normal((output_dim, input_dim)) * scale
        self.bias = mx.zeros((output_dim,)) if bias else None
        self.norm_factor = np.sqrt(input_dim)

    def __call__(
        self, x: mx.array, adapters: Optional[Tuple[mx.array, mx.array]] = None
    ):
        """
        x: (Batch, InputDim)
        adapters: Tuple(A, B) where:
            A: (Pop, OutputDim, Rank)
            B: (Pop, InputDim, Rank)
        """
        # 1. Shared Dense Pass
        base_out = x @ self.weight.T

        # Add Low-Rank Perturbation if active
        if adapters is not None:
            A, B = adapters

            # Case 1: x is (Batch, In)
            if x.ndim == 2:
                input_projected = mx.einsum("bi, pir -> bpr", x, B)
            # Case 2: x is (Batch, Pop, In)
            elif x.ndim == 3:
                input_projected = mx.einsum("bpi, pir -> bpr", x, B)
            else:
                raise ValueError(f"Unexpected x shape {x.shape}")

            delta = mx.einsum("bpr, por -> bpo", input_projected, A)

            # Debug prints
            # if np.random.rand() < 0.01: # Sample occasionally
            #     print(f"Layer Debug: x_mean={x.mean()}, x_max={x.max()}")
            #     print(f"  base_out_mean={base_out.mean()}, base_out_max={base_out.max()}")
            #     print(f"  input_proj_mean={input_projected.mean()}, input_proj_max={input_projected.max()}")
            #     print(f"  delta_mean={delta.mean()}, delta_max={delta.max()}")

            # Combine
            if x.ndim == 2:
                base_out = mx.expand_dims(base_out, 1)  # (Batch, 1, Out)

            base_out = base_out + delta

        if self.bias is not None:
            base_out = base_out + self.bias

        return base_out / self.norm_factor


class BatchedMLP(nn.Module):
    def __init__(self, cfg: ExperimentConfig):
        super().__init__()
        self.fc1 = EggrollLinearBatched(cfg.input_dim, cfg.hidden_dim)
        self.fc2 = EggrollLinearBatched(cfg.hidden_dim, cfg.output_dim)
        self.scale = cfg.scale
        self.cfg = cfg

    def normalize(self, x):
        # Parameter-free LayerNorm to stabilize batched forward pass
        # x: (..., Hidden)
        mean = mx.mean(x, axis=-1, keepdims=True)
        var = mx.var(x, axis=-1, keepdims=True)
        return (x - mean) / mx.sqrt(var + 1e-6)

    def forward_train(self, x, adapter_stack):
        s = self.scale

        # Layer 1
        h1 = self.fc1(x, adapter_stack[0])
        h1 = self.normalize(h1)  # Stabilize
        h1 = nn.relu(h1)  # Activation

        # Layer 2
        out = self.fc2(h1, adapter_stack[1])
        # Output layer usually doesn't get LN before logits?
        # But if it explodes, we might need to scale it.
        # Let's just return it. fc2 normalizes by sqrt(d) internally.

        return out

    def forward_inference(self, x):
        s = self.scale
        h1 = self.fc1(x, None)
        h1 = self.normalize(h1)
        h1 = nn.relu(h1)

        out = self.fc2(h1, None)
        return out


def make_data(cfg: ExperimentConfig):
    X, y = make_moons(cfg.samples, noise=cfg.noise, random_state=cfg.seed)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg.test_size, random_state=cfg.seed
    )
    return (
        mx.array(X_train, dtype=mx.float32),
        mx.array(y_train, dtype=mx.int32),
        mx.array(X_test, dtype=mx.float32),
        mx.array(y_test, dtype=mx.int32),
    )


def cross_entropy_batched(logits, targets):
    # logits: (Batch, Pop, Out)
    # targets: (Batch,)

    B, P, C = logits.shape

    # Stable LogSoftmax
    # max_logits: (Batch, Pop, 1)
    max_logits = mx.max(logits, axis=-1, keepdims=True)
    shifted_logits = logits - max_logits

    # log(sum(exp(logits - max))) + max
    log_sum_exp = (
        mx.log(mx.sum(mx.exp(shifted_logits), axis=-1, keepdims=True)) + max_logits
    )

    # log_probs = logits - log_sum_exp
    log_probs = logits - log_sum_exp

    # Select target class
    # targets: (Batch,) -> need (Batch, 1, 1) for creating onehot or indexing?
    # We need to select along axis=-1 (C) using targets.
    # targets is same for all Pop members.

    # One-hot approach is easiest in MLX without advanced indexing
    one_hot = (targets[..., None] == mx.arange(C)).astype(mx.float32)  # (B, C)
    one_hot = mx.expand_dims(one_hot, 1)  # (B, 1, C)

    # sum(one_hot * log_probs) -> (B, P)
    loss_per_sample = -mx.sum(one_hot * log_probs, axis=-1)

    # Mean over batch -> (P,)
    return mx.mean(loss_per_sample, axis=0)


def accuracy(logits, targets):
    preds = mx.argmax(logits, axis=-1)
    return (preds == targets).astype(mx.float32).mean()


def flatten_params(tree):
    flat = mlx.utils.tree_flatten(tree)
    paths = [p for p, _ in flat]
    vals = [v for _, v in flat]
    return paths, vals


def unflatten(paths, vals):
    return mlx.utils.tree_unflatten(list(zip(paths, vals)))


def generate_batched_noise(cfg, model_shapes, key):
    # Return list of tuples [(A, B), (A, B)] for layers
    # Specifically tailored for the 2-layer MLP structure
    # shapes list likely: [fc1.weight, fc1.bias, fc2.weight, fc2.bias]

    # We need to identify which are weights (Matrices) and which are biases (Vectors)
    # And pair them correctly.

    noise_stack = []
    pop = cfg.pop_size
    k = key

    layer_struct = []

    # Helper to generate one param noise
    def get_noise(shape):
        nonlocal k
        if len(shape) < 2:
            # Bias -> Just dense noise (Pop, Dim)
            # Though for batched forward pass, we might ignore bias noise or treat it as (Pop, Out, 1) @ (Pop, 1, 1)?
            # The paper/nano-egg usually ignores bias or treats it as full param.
            # For this Batched Demo, let's ONLY perturb weights to simplify the "Low Rank" pipeline.
            return None

        rows, cols = shape
        k, subA = mx.random.split(k)
        k, subB = mx.random.split(k)

        # Generate stacked A, B directly
        # A: (Pop, Rows, Rank)
        # B: (Pop, Cols, Rank)

        # Antithetic trick inside the stack generation
        half = pop // 2 if cfg.pop_size % 2 == 0 else pop  # Simplifying

        A_half = mx.random.normal((half, rows, cfg.rank), key=subA)
        B_half = mx.random.normal((half, cols, cfg.rank), key=subB)

        scale = cfg.sigma / np.sqrt(cfg.rank)
        A_half = A_half * scale

        A = mx.concatenate([A_half, -A_half], axis=0)
        B = mx.concatenate([B_half, B_half], axis=0)

        return (A, B)

    for shape in model_shapes:
        res = get_noise(shape)
        if res is not None:
            layer_struct.append(res)

    return layer_struct


def run_batched_eggroll(X_train, y_train, X_test, y_test, cfg: ExperimentConfig):
    model = BatchedMLP(cfg)
    mx.eval(model.parameters())

    # Setup int mode initial state
    params_paths, params_vals_any = flatten_params(model.trainable_parameters())
    params_vals = [p for p in params_vals_any if isinstance(p, mx.array)]

    if cfg.int_mode:
        params_vals = [
            mx.clip((p * cfg.scale).round(), -127, 127).astype(mx.float32)
            for p in params_vals
        ]
        model.update(unflatten(params_paths, params_vals))

    key = mx.random.key(cfg.seed)

    losses = []
    accs = []

    start_time = time.perf_counter()

    for step in range(cfg.steps):
        k, key = mx.random.split(key)

        # 1. Generate Batched Noise (Stacked A, B)
        # Shapes: [fc1.w, fc1.b, fc2.w, fc2.b] -> we get [(A1, B1), (A2, B2)]
        shapes = [p.shape for p in params_vals]
        adapter_stack = generate_batched_noise(cfg, shapes, k)

        # 2. Batched Forward Pass (The Magic)
        # Runs P models in parallel
        logits_pop = model.forward_train(X_train, adapter_stack)  # (Batch, Pop, Out)

        if step == 0 or step % 10 == 0:
            print(
                f"Step {step}: Logits Mean: {mx.mean(logits_pop)}, Max: {mx.max(logits_pop)}, Min: {mx.min(logits_pop)}"
            )
            if mx.isnan(mx.mean(logits_pop)):
                print("NAN DETECTED IN LOGITS")
                # Break into layers to see where
                # We can't easily inspect inside model here without hooks, but we added prints inside EggrollLinearBatched

        # 3. Calculate Loss per Population Member
        # Returns (Pop,)
        pop_losses = cross_entropy_batched(logits_pop, y_train)

        # 4. ES Update (Sign based)
        # Calculate update for each layer
        # Update = Sum(Loss_i * Noise_i)

        # Antithetic pairing is implicit in the order of adapter_stack (0, N/2) are pairs
        # pop_losses: [L0, L1, ... L_half, L_half+1 ...]
        # Noise:      [N0, N1, ... N0,     N1 ...] (B is same, A is flipped)

        # Let's compute centered scores (ranking)
        # Fast fitness: sign(L_pos - L_neg)
        half = cfg.pop_size // 2
        l_pos = pop_losses[:half]
        l_neg = pop_losses[half:]

        # We want to MINIMIZE loss.
        # If l_pos < l_neg, perturbation +A improved loss. We want to step in +A.
        # l_pos - l_neg would be negative.
        # So we want sign(l_neg - l_pos).

        scores = mx.sign(l_neg - l_pos)  # (Half,)

        # Apply update to base weights
        # W_new = W_old + lr * Sum(Score * (A * B^T))
        # We can compute Sum(Score * A) first -> A_agg
        # Then W_new = W_old + lr * (A_agg @ B^T)

        # We need to iterate over layers
        current_param_idx = 0
        new_params = []

        adapter_idx = 0
        for p in params_vals:
            if len(p.shape) < 2:
                # Bias - we didn't perturb it
                new_params.append(p)
                continue

            # Weight Matrix
            A_stack, B_stack = adapter_stack[adapter_idx]
            adapter_idx += 1

            # A_stack: (Pop, Rows, Rank).
            # We only need first half A (since second half is -A)
            A_half = A_stack[:half]  # (Half, Rows, Rank)
            B_half = B_stack[:half]  # (Half, Cols, Rank)

            # Scores: (Half,) -> Broadcast to (Half, 1, 1)
            scores_expanded = scores.reshape(-1, 1, 1)

            # Weighted Sum of A
            # (Half, Rows, Rank) * (Half, 1, 1) -> Sum over Half
            A_agg = mx.sum(A_half * scores_expanded, axis=0)  # (Rows, Rank)

            # But wait, B is different for every member?
            # NO. In Standard ES, B is unique per member.
            # In my `generate_batched_noise`, I generated unique B per member.
            # So we can't just do A_agg @ B.T.
            # We must do Sum(Score_i * (A_i @ B_i.T)).

            # Opt 1: Reconstruct dense noise per member (Memory hungry?)
            # Opt 2: Einsum sum.

            # Weighted A: A_w = A_half * scores
            A_w = A_half * scores_expanded

            # Delta = Sum_i (A_w_i @ B_i.T)
            # Einsum: "nkr, ncr -> kc" (Sum over N, Matrix Mul KxR @ RxC)
            # n=Half, k=Rows, c=Cols, r=Rank
            delta = mx.einsum("nkr, ncr -> kc", A_w, B_half)

            # Normalize delta by population size to prevent explosion with large pop
            delta = delta / cfg.pop_size

            # In Antithetic, the negative sample had noise -A @ B.T.
            # Score = L+ - L-.
            # Gradient approx ~ (L+ - L-)/(2 sigma) * Noise.
            # We used sign(L+ - L-) as score.
            # So update is proportional to score * Noise.
            # Correct.

            # Apply update (Integer Step)
            # Delta is the accumulated gradient direction.
            step = mx.sign(delta)
            updated = p + step
            updated = mx.clip(updated, -127, 127)
            new_params.append(updated)

        params_vals = new_params
        model.update(unflatten(params_paths, params_vals))
        mx.eval(model.parameters())

        # Eval
        logits = model.forward_inference(X_train)

        # Re-define cross_entropy_logits locally or import
        def cross_entropy_logits_local(logits, targets):
            num_classes = logits.shape[-1]
            one_hot = (targets[..., None] == mx.arange(num_classes)).astype(mx.float32)
            log_probs = logits - mx.log(mx.sum(mx.exp(logits), axis=-1, keepdims=True))
            return -mx.mean(mx.sum(one_hot * log_probs, axis=-1))

        losses.append(cross_entropy_logits_local(logits, y_train).item())
        accs.append(accuracy(model.forward_inference(X_test), y_test).item())

    duration = time.perf_counter() - start_time
    print(
        f"Batched Eggroll (Pop {cfg.pop_size}): Time {duration:.3f}s, Final Loss {losses[-1]:.4f}, Acc {accs[-1]:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop_size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--sigma", type=float, default=0.05)
    parser.add_argument("--int_mode", action="store_true", default=False)
    parser.add_argument("--no_int_mode", action="store_false", dest="int_mode")
    parser.set_defaults(int_mode=False)
    args = parser.parse_args()

    cfg = ExperimentConfig(
        pop_size=args.pop_size,
        steps=args.steps,
        sigma=args.sigma,
        int_mode=args.int_mode,
    )
    if args.int_mode:
        cfg.scale = 16.0
    else:
        cfg.scale = 1.0

    X_train, y_train, X_test, y_test = make_data(cfg)
    run_batched_eggroll(X_train, y_train, X_test, y_test, cfg)
