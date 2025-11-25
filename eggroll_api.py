"""
Lightweight API helpers to make eggroll primitives easy to reuse across models.

Use EggrollContext to bundle NoiseConfig, BIG_RAND, epoch, and base_seed, and to
run forwards (single or batched) and sign updates without repeatedly threading
those arguments around.
"""

from dataclasses import dataclass
from typing import List, Optional

import mlx.core as mx

from eggroll_mlx import (
    NoiseConfig,
    apply_sign_update,
    do_mm,
    do_mm_batched,
    generate_big_rand,
)


@dataclass
class EggrollContext:
    cfg: NoiseConfig
    big_rand: mx.array
    base_seed: int = 0
    epoch: int = 0

    def forward(self, x: mx.array, w: mx.array, thread_id: Optional[int] = None) -> mx.array:
        """
        Single-perturbation forward. If thread_id is None, runs noiseless base matmul.
        """
        if thread_id is None:
            return do_mm(self.cfg, x, w, None, None, None, None)
        return do_mm(self.cfg, x, w, self.big_rand, self.epoch, thread_id, self.base_seed)

    def forward_batched(self, x: mx.array, w: mx.array, thread_ids: List[int]) -> mx.array:
        """
        Batched forward with shared activations for multiple perturbations.
        """
        return do_mm_batched(self.cfg, x, w, self.big_rand, self.epoch, thread_ids, self.base_seed)

    def update(self, param: mx.array, fitnesses: mx.array, thread_ids: Optional[List[int]] = None, seed_offset: int = 0) -> mx.array:
        """
        Apply sign update for a matrix parameter using paired fitnesses.
        """
        return apply_sign_update(
            self.cfg,
            param,
            fitnesses,
            self.big_rand,
            self.epoch,
            self.base_seed,
            thread_ids=thread_ids,
            seed_offset=seed_offset,
        )

    def next_epoch(self, delta: int = 1) -> None:
        """Increment epoch in place."""
        self.epoch += delta


def make_context(
    cfg: NoiseConfig,
    param_span: int,
    seed: int = 0,
    safety_margin: int = 1024,
) -> EggrollContext:
    """
    Convenience to create an EggrollContext with a generated BIG_RAND buffer.
    param_span should be at least the maximum span needed: (rows+cols)*rank or
    prod(shape)*2 for full-noise params. safety_margin adds extra space for slicing.
    """
    big_rand = generate_big_rand(param_span + safety_margin, seed=seed, fixed_point=cfg.fixed_point)
    return EggrollContext(cfg=cfg, big_rand=big_rand, base_seed=seed, epoch=0)
