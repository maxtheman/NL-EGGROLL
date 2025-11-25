import os
import sys
from pathlib import Path

import numpy as np
import mlx.core as mx
import pytest

# Add nano-egg to path
ROOT = Path(__file__).resolve().parents[1]
NANO_PATH = ROOT / "third_party_references" / "repos" / "nano-egg"
sys.path.append(str(NANO_PATH))
sys.path.append(str(ROOT))

import jax
import jax.numpy as jnp
_argv_backup = sys.argv
sys.argv = ["nano-egg"]  # prevent tyro CLI from parsing pytest args
os.environ["NANO_EGG_NO_CLI"] = "1"
import run as nano  # type: ignore
sys.argv = _argv_backup

from eggroll_mlx import (
    NoiseConfig,
    get_common_start_idx,
    get_lora_update_params,
    convert_fitnesses,
    apply_sign_update,
    do_mm,
    generate_big_rand,
)


def _base_seed(seed: int):
    return seed


def test_common_start_idx_matches_nano():
    cfg = NoiseConfig(rank=1, noise_reuse=1)
    epoch, thread_id = 3, 4
    base_seed = _base_seed(0)
    # nano reference
    dummy_param = jnp.zeros((2, 2))
    nano_start, nano_sign = nano.get_common_start_idx(
        {"noise_reuse": 1}, (epoch, thread_id), dummy_param, jax.random.key(0)
    )
    mx_start, mx_sign = get_common_start_idx(cfg, epoch, thread_id, base_seed)
    assert int(mx_start) == int(nano_start)
    assert int(mx_sign) == int(nano_sign)


def test_lora_slice_matches_nano():
    cfg = NoiseConfig(rank=1, noise_reuse=1)
    epoch, thread_id = 0, 1
    rows, cols = 3, 2
    base_seed = _base_seed(0)
    span = (rows + cols) * cfg.rank
    param = jnp.zeros((rows, cols), dtype=jnp.int8)
    start_ref, _ = nano.get_common_start_idx({"noise_reuse": 1}, (epoch, thread_id), param, jax.random.key(0))
    start_ref = int(start_ref)
    if start_ref + span > 2_000_000:
        pytest.skip("Reference start_idx too large to allocate test buffer safely")
    size = start_ref + span + 1
    big_rand_np = np.arange(size, dtype=np.int32)  # deterministic
    big_rand_mx = mx.array(big_rand_np.astype(np.int8))
    big_rand_jax = jnp.array(big_rand_np.astype(np.int8))

    A_ref, B_ref = nano.get_lora_update_params(
        big_rand_jax, {"rank": cfg.rank, "noise_reuse": 1}, (epoch, thread_id), param, jax.random.key(0)
    )
    A_ref = np.array(A_ref)
    B_ref = np.array(B_ref)

    A_mx, B_mx = get_lora_update_params(big_rand_mx, cfg, epoch, thread_id, (rows, cols), base_seed)
    np.testing.assert_array_equal(np.array(A_mx), A_ref)
    np.testing.assert_array_equal(np.array(B_mx), B_ref)


def test_convert_fitnesses_signs_match():
    cfg = NoiseConfig(rank=1)
    raw = mx.array([1.0, 2.0, -3.0, -1.0])  # pairs: (1,2)->-, (-3,-1)->-
    out = convert_fitnesses(cfg, raw)
    assert np.all(np.array(out) == np.array([-1, -1], dtype=np.int8))


def test_do_mm_matches_nano_on_small_case():
    cfg = NoiseConfig(rank=1, sigma_shift=0, noise_reuse=1)
    epoch, thread_id = 0, 0
    base_seed = _base_seed(0)
    x_np = np.array([[1, 2]], dtype=np.int8)
    w_np = np.array([[3, 4]], dtype=np.int8)
    big_rand = generate_big_rand(64, seed=0, fixed_point=cfg.fixed_point, dtype=mx.int8)

    # nano forward
    x_jax = jnp.array(x_np)
    w_jax = jnp.array(w_np)
    y_ref = nano.QEggRoll.do_mm(
        {"noise_reuse": 1, "rank": cfg.rank, "sigma_shift": cfg.sigma_shift, "use_clt": True, "fast_fitness": True},
        {"BIG_RAND_MATRIX": jnp.array(np.array(big_rand)), "sigma_shift": cfg.sigma_shift},
        w_jax,
        jax.random.key(0),
        (epoch, thread_id),
        x_jax,
    )
    # ours
    y_mx = do_mm(cfg, mx.array(x_np), mx.array(w_np), big_rand, epoch, thread_id, base_seed)
    np.testing.assert_array_equal(np.array(y_mx), np.array(y_ref))


def test_apply_sign_update_direction_matches():
    cfg = NoiseConfig(rank=1, sigma_shift=0, noise_reuse=1)
    epoch = 0
    base_seed = _base_seed(0)
    param = mx.zeros((2, 2), dtype=mx.int8)
    fitnesses = mx.array([1, -1], dtype=mx.int8)  # two pairs -> four threads
    # generate shared BIG_RAND via jax for exact parity
    scale = 2 ** cfg.fixed_point
    big_rand_jax = (jax.random.normal(jax.random.key(1), (2_000_000,)) * scale).astype(jnp.int8)
    big_rand = mx.array(np.array(big_rand_jax))

    # nano reference using do_updates with es_map=1 for this param
    p_jax = jnp.array(np.zeros((2, 2), dtype=jnp.int8))
    f_jax = jnp.array(np.array(fitnesses))
    nr = {"noise_reuse": 1, "rank": cfg.rank, "use_clt": cfg.use_clt, "fast_fitness": cfg.fast_fitness}
    np_params = {
        "BIG_RAND_MATRIX": big_rand_jax,
        "sigma_shift": cfg.sigma_shift,
        "update_threshold": 0,
    }
    base_key = jax.random.key(0)
    iterinfos = (jnp.zeros(fitnesses.shape[0], dtype=jnp.int32), jnp.arange(fitnesses.shape[0], dtype=jnp.int32))
    ref = nano.QEggRoll._do_update(nr, np_params, p_jax, base_key, f_jax, iterinfos, 1)
    ref = np.array(ref)

    updated = apply_sign_update(cfg, param, fitnesses, big_rand, epoch, base_seed)
    np.testing.assert_array_equal(np.array(updated), np.array(ref))
