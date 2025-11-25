import argparse
import time

import mlx.core as mx
import numpy as np

from eggroll_mlx import NoiseConfig, do_mm, do_mm_batched, generate_big_rand


def bench(cfg: NoiseConfig, x, w, big_rand, epoch, base_seed, thread_ids, iters, batched: bool):
    mx.eval(x, w, big_rand)  # materialize
    start = time.perf_counter()
    out = None
    for _ in range(iters):
        if batched:
            out = do_mm_batched(cfg, x, w, big_rand, epoch, thread_ids, base_seed)
        else:
            outs = []
            for tid in thread_ids:
                outs.append(do_mm(cfg, x, w, big_rand, epoch, tid, base_seed))
            out = mx.stack(outs, axis=0)
    mx.eval(out)
    elapsed = time.perf_counter() - start
    return elapsed / iters, out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--m", type=int, default=64, help="batch dimension")
    p.add_argument("--n", type=int, default=256, help="output dimension")
    p.add_argument("--k", type=int, default=256, help="input dimension")
    p.add_argument("--rank", type=int, default=1)
    p.add_argument("--pop", type=int, default=8)
    p.add_argument("--iters", type=int, default=20)
    p.add_argument("--use-quant", action="store_true", help="enable quantized base matmul path")
    args = p.parse_args()

    cfg = NoiseConfig(rank=args.rank, use_quantized_base=args.use_quant)
    batch, out_dim, in_dim = args.m, args.n, args.k
    x = mx.array(np.random.randint(-128, 127, size=(batch, in_dim), dtype=np.int8))
    w = mx.array(np.random.randint(-128, 127, size=(out_dim, in_dim), dtype=np.int8))
    big_rand = generate_big_rand((out_dim + in_dim) * cfg.rank * args.pop * 2 + 1024, seed=0, fixed_point=cfg.fixed_point)
    epoch = 0
    base_seed = 0
    thread_ids = list(range(args.pop))

    loop_time, loop_out = bench(cfg, x, w, big_rand, epoch, base_seed, thread_ids, args.iters, batched=False)
    batch_time, batch_out = bench(cfg, x, w, big_rand, epoch, base_seed, thread_ids, args.iters, batched=True)

    try:
        np.testing.assert_array_equal(np.array(loop_out), np.array(batch_out))
        correctness = "outputs match"
    except AssertionError:
        correctness = "outputs differ"

    print(f"Shape (B={batch}, N={out_dim}, K={in_dim}), pop={args.pop}, rank={args.rank}, use_quant={args.use_quant}")
    print(f" looped per-thread: {loop_time*1e3:.3f} ms/iter")
    print(f" batched delta   : {batch_time*1e3:.3f} ms/iter")
    print(f" speedup         : {loop_time / batch_time:.2f}x")
    print(f" correctness     : {correctness}")


if __name__ == "__main__":
    main()
