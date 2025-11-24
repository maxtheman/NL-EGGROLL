"""
Final Benchmark: Batched Eggroll vs Sequential Eggroll vs Adam.
Includes Decision Boundary Visualization.

Usage:
  uv run python scripts/exp_final_benchmark.py --steps 1000 --pop_size 256 --no_int_mode
"""

import argparse
import time
import json
import mlx.core as mx
import mlx.optimizers as optim

# ...


def run_adam(X_train, y_train, X_test, y_test, cfg):
    model = BatchedMLP(cfg)
    mx.eval(model.parameters())

    def loss_fn(m):
        logits = m.forward_inference(X_train)
        return -mx.mean(
            mx.sum(
                (y_train[..., None] == mx.arange(cfg.output_dim)).astype(mx.float32)
                * (logits - mx.log(mx.sum(mx.exp(logits), axis=-1, keepdims=True))),
                axis=-1,
            )
        )

    loss_and_grad = nn.value_and_grad(model, loss_fn)
    optimizer = optim.Adam(learning_rate=cfg.lr_adam)
    losses = []
    times = []
    start_time = time.perf_counter()

    for i in range(cfg.steps):
        loss, grads = loss_and_grad(model)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        losses.append(loss.item())
        times.append(time.perf_counter() - start_time)

    acc = accuracy(model.forward_inference(X_test), y_test).item()
    return {
        "name": "Adam",
        "losses": losses,
        "times": times,
        "final_acc": acc,
        "model": model,
    }


def run_batched_eggroll(X_train, y_train, X_test, y_test, cfg):
    model = BatchedMLP(cfg)
    mx.eval(model.parameters())
    params_paths, params_vals = flatten_params(model.trainable_parameters())

    key = mx.random.key(cfg.seed)
    losses = []
    times = []
    start_time = time.perf_counter()

    for step in range(cfg.steps):
        k, key = mx.random.split(key)
        shapes = [p.shape for p in params_vals]
        adapter_stack = generate_batched_noise(cfg, shapes, k)

        # Batched Forward
        logits_pop = model.forward_train(X_train, adapter_stack)

        # Loss
        B, P, C = logits_pop.shape
        max_logits = mx.max(logits_pop, axis=-1, keepdims=True)
        shifted = logits_pop - max_logits
        log_probs = logits_pop - (
            mx.log(mx.sum(mx.exp(shifted), axis=-1, keepdims=True)) + max_logits
        )
        one_hot = mx.expand_dims(
            (y_train[..., None] == mx.arange(C)).astype(mx.float32), 1
        )
        pop_losses = mx.mean(-mx.sum(one_hot * log_probs, axis=-1), axis=0)

        # Update
        half = cfg.pop_size // 2
        l_pos = pop_losses[:half]
        l_neg = pop_losses[half:]
        scores = mx.sign(l_neg - l_pos)  # Minimize loss

        new_params = []
        adapter_idx = 0
        for p in params_vals:
            if len(p.shape) < 2:
                new_params.append(p)
                continue

            A_stack, B_stack = adapter_stack[adapter_idx]
            adapter_idx += 1

            A_half = A_stack[:half]
            B_half = B_stack[:half]

            A_w = A_half * scores.reshape(-1, 1, 1)
            delta = mx.einsum("nkr, ncr -> kc", A_w, B_half)
            delta = delta / cfg.pop_size

            # Float update
            grad_est = delta / (cfg.sigma + 1e-8)
            updated = p + cfg.lr_es * grad_est
            new_params.append(updated)

        params_vals = new_params
        model.update(unflatten(params_paths, params_vals))
        mx.eval(model.parameters())

        # Eval (using inference mode)
        loss_val = -mx.mean(
            mx.sum(
                (y_train[..., None] == mx.arange(C)).astype(mx.float32)
                * (
                    model.forward_inference(X_train)
                    - mx.log(
                        mx.sum(
                            mx.exp(model.forward_inference(X_train)),
                            axis=-1,
                            keepdims=True,
                        )
                    )
                ),
                axis=-1,
            )
        ).item()

        losses.append(loss_val)
        times.append(time.perf_counter() - start_time)

    acc = accuracy(model.forward_inference(X_test), y_test).item()
    return {
        "name": "Batched Eggroll",
        "losses": losses,
        "times": times,
        "final_acc": acc,
        "model": model,
    }


def run_sequential_eggroll(X_train, y_train, X_test, y_test, cfg):
    # Simulates the memory-efficient but slow way:
    # For each population member:
    # 1. Generate noise
    # 2. Add noise to weights (in-place update)
    # 3. Forward pass
    # 4. Subtract noise

    model = BatchedMLP(cfg)
    mx.eval(model.parameters())
    params_paths, params_vals = flatten_params(model.trainable_parameters())

    key = mx.random.key(cfg.seed)
    losses = []
    times = []
    start_time = time.perf_counter()

    # We simulate standard ES behavior: we just need the rewards to compute gradient
    # We will use the SAME gradient accumulation logic as batched to be fair on accuracy,
    # but we pay the time cost of sequential execution.

    for step in range(cfg.steps):
        k, key = mx.random.split(key)
        # Generate Batched Noise structure just to keep math identical for 'delta' calc
        shapes = [p.shape for p in params_vals]
        adapter_stack = generate_batched_noise(cfg, shapes, k)

        pop_losses_list = []

        # Sequential Loop
        for j in range(cfg.pop_size):
            # Apply perturbation j manually
            perturbed_params = []
            adapter_idx = 0
            for p in params_vals:
                if len(p.shape) < 2:
                    perturbed_params.append(p)
                    continue

                A_stack, B_stack = adapter_stack[adapter_idx]
                adapter_idx += 1

                A = A_stack[j]
                B = B_stack[j]
                delta = A @ B.T
                perturbed_params.append(p + delta)

            # Update model weights temporarily
            model.update(unflatten(params_paths, perturbed_params))

            # Forward Inference (Standard Pass)
            logits = model.forward_inference(X_train)

            # Loss
            loss_val = -mx.mean(
                mx.sum(
                    (y_train[..., None] == mx.arange(cfg.output_dim)).astype(mx.float32)
                    * (logits - mx.log(mx.sum(mx.exp(logits), axis=-1, keepdims=True))),
                    axis=-1,
                )
            ).item()
            pop_losses_list.append(loss_val)

        # Restore weights
        model.update(unflatten(params_paths, params_vals))

        pop_losses = mx.array(pop_losses_list)

        # SAME Update Logic as Batched
        half = cfg.pop_size // 2
        l_pos = pop_losses[:half]
        l_neg = pop_losses[half:]
        scores = mx.sign(l_neg - l_pos)

        new_params = []
        adapter_idx = 0
        for p in params_vals:
            if len(p.shape) < 2:
                new_params.append(p)
                continue

            A_stack, B_stack = adapter_stack[adapter_idx]
            adapter_idx += 1

            A_half = A_stack[:half]
            B_half = B_stack[:half]

            A_w = A_half * scores.reshape(-1, 1, 1)
            delta = mx.einsum("nkr, ncr -> kc", A_w, B_half)
            delta = delta / cfg.pop_size

            grad_est = delta / (cfg.sigma + 1e-8)
            updated = p + cfg.lr_es * grad_est
            new_params.append(updated)

        params_vals = new_params
        model.update(unflatten(params_paths, params_vals))
        mx.eval(model.parameters())

        # Tracking
        losses.append(np.mean(pop_losses_list))  # Approx loss
        times.append(time.perf_counter() - start_time)

    acc = accuracy(model.forward_inference(X_test), y_test).item()
    return {
        "name": "Sequential Eggroll",
        "losses": losses,
        "times": times,
        "final_acc": acc,
        "model": model,
    }


# -----------------------------------------------------------------------------
# Visualization & Main
# -----------------------------------------------------------------------------


def plot_decision_boundary(model, X, y, title="Decision Boundary"):
    # Define grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

    grid = np.c_[xx.ravel(), yy.ravel()]
    grid_mx = mx.array(grid, dtype=mx.float32)

    logits = model.forward_inference(grid_mx)
    preds = mx.argmax(logits, axis=1).astype(mx.int32)

    Z = np.array(preds).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap=plt.cm.coolwarm)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.coolwarm, edgecolors="k")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.savefig(f"decision_boundary_{title.replace(' ', '_')}.png")
    print(f"Saved decision_boundary_{title.replace(' ', '_')}.png")


def plot_training_curves(results_list):
    plt.figure(figsize=(10, 6))
    for res in results_list:
        plt.plot(res["times"], res["losses"], label=res["name"])
    plt.xlabel("Time (s)")
    plt.ylabel("Loss")
    plt.title(
        f"Training Efficiency (Wall Clock) - {results_list[0]['final_acc']} vs others"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig("training_efficiency.png")
    print("Saved training_efficiency.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pop_size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--no_int_mode", action="store_true")
    args = parser.parse_args()

    cfg = ExperimentConfig(pop_size=args.pop_size, steps=args.steps, int_mode=False)

    print(f"Running Benchmark with Pop={cfg.pop_size}, Steps={cfg.steps}")
    X_train, y_train, X_test, y_test = make_data(cfg)

    results = []

    # 1. Batched Eggroll
    print("\n--- Batched Eggroll ---")
    res_batched = run_batched_eggroll(X_train, y_train, X_test, y_test, cfg)
    results.append(res_batched)
    print(f"Final Acc: {res_batched['final_acc']:.4f}")

    # 2. Adam (Baseline)
    print("\n--- Adam ---")
    res_adam = run_adam(X_train, y_train, X_test, y_test, cfg)
    results.append(res_adam)
    print(f"Final Acc: {res_adam['final_acc']:.4f}")

    # 3. Sequential Eggroll (Time Baseline)
    # Only run if pop is manageable, otherwise it takes forever. 256 is okay-ish for 200 steps, maybe slow for 1000.
    # Let's run it.
    print("\n--- Sequential Eggroll ---")
    res_seq = run_sequential_eggroll(X_train, y_train, X_test, y_test, cfg)
    results.append(res_seq)
    print(f"Final Acc: {res_seq['final_acc']:.4f}")

    # Plotting
    plot_training_curves(results)
    plot_decision_boundary(
        res_batched["model"], np.array(X_test), np.array(y_test), "Batched Eggroll"
    )
    plot_decision_boundary(
        res_adam["model"], np.array(X_test), np.array(y_test), "Adam"
    )
