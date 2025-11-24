import mlx.core as mx

from eggroll_mlx import calibrate_divisor, int8_matmul


def test_calibrate_divisor_large_dot():
    # base_out = 128, target_max=32 -> divisor 4, scaled ~32
    x = mx.array([[8, 8]], dtype=mx.int8)
    w = mx.array([[8, 8]], dtype=mx.int8)
    div = calibrate_divisor(x, w, target_max=32)
    assert div == 4
    scaled = int8_matmul(x, w) // div
    assert int(mx.max(scaled).item()) == 32


def test_calibrate_divisor_small_dot():
    # base_out = 2, target_max=32 -> divisor clamps to 1, preserves amplitude
    x = mx.array([[1, 1]], dtype=mx.int8)
    w = mx.array([[1, 1]], dtype=mx.int8)
    div = calibrate_divisor(x, w, target_max=32)
    assert div == 1
    scaled = int8_matmul(x, w) // div
    assert int(mx.max(scaled).item()) == 2
