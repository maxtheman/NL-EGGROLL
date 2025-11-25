import numpy as np
import mlx.core as mx

from eggroll_api import EggrollContext, make_context
from eggroll_mlx import NoiseConfig, do_mm


def test_forward_matches_do_mm():
    cfg = NoiseConfig(rank=1, noise_reuse=1)
    rows, cols = 4, 3
    param_span = (rows + cols) * cfg.rank * 4  # enough for a few thread ids
    ctx = make_context(cfg, param_span=param_span, seed=0)
    x = mx.array(np.random.randint(-5, 5, size=(2, cols)), dtype=mx.int8)
    w = mx.array(np.random.randint(-5, 5, size=(rows, cols)), dtype=mx.int8)
    tid = 2

    y_ctx = ctx.forward(x, w, thread_id=tid)
    y_ref = do_mm(cfg, x, w, ctx.big_rand, ctx.epoch, tid, ctx.base_seed)
    np.testing.assert_array_equal(np.array(y_ctx), np.array(y_ref))


def test_forward_batched_matches_stack():
    cfg = NoiseConfig(rank=1, noise_reuse=1)
    rows, cols = 4, 3
    param_span = (rows + cols) * cfg.rank * 8
    ctx = make_context(cfg, param_span=param_span, seed=1)
    x = mx.array(np.random.randint(-5, 5, size=(2, cols)), dtype=mx.int8)
    w = mx.array(np.random.randint(-5, 5, size=(rows, cols)), dtype=mx.int8)
    thread_ids = [0, 1, 2]

    y_batch = ctx.forward_batched(x, w, thread_ids)
    y_loop = mx.stack([ctx.forward(x, w, thread_id=tid) for tid in thread_ids], axis=0)
    np.testing.assert_array_equal(np.array(y_batch), np.array(y_loop))
