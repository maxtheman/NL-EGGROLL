
import time
import mlx.core as mx
import numpy as np
from scripts.train_tinystories_gru_full import forward_block, init_model, EggrollContext, NoiseConfig

def benchmark_compile():
    print("Benchmarking mx.compile on forward_block...")
    
    # Setup
    pop_size = 100
    batch_size = 4
    d_model = 256
    d_hidden = 256
    
    cfg = NoiseConfig()
    # Mock context
    class MockContext:
        def __init__(self):
            self.cfg = cfg
            self.big_rand = mx.zeros((1000000,), dtype=mx.int8)
            self.epoch = 0
            self.base_seed = 0
            
    ctx = MockContext()
    
    # Init weights
    rng = np.random.default_rng(0)
    w = init_model(cfg, 100, 10, d_model, d_hidden, 1, rng, 1.0)
    blk = w.blocks[0]
    
    # Inputs
    x_in = mx.zeros((pop_size, batch_size, d_model), dtype=mx.int8)
    h_state = mx.zeros((pop_size, batch_size, d_hidden), dtype=mx.int8)
    tid_base = list(range(pop_size))
    seed_offset = 0
    
    # Wrapper to unpack context
    def forward_wrapper(ln1, ln2, gru_x_w, gru_x_s, gru_h_w, gru_h_s, mlp_in_w, mlp_in_s, mlp_out_w, mlp_out_s, 
                       gru_b1, gru_b2, mlp_b1, mlp_b2,
                       big_rand, epoch, base_seed,
                       x_in, h_state, tid_base, seed_offset):
        
        # Reconstruct objects
        cfg = NoiseConfig() # Assuming default config
        ctx = EggrollContext(cfg, big_rand, base_seed, epoch)
        
        # Reconstruct BlockWeights
        # We need to reconstruct Int8Mat objects
        from scripts.train_tinystories_gru_full import Int8Mat, BlockWeights
        
        gru_x = Int8Mat(gru_x_w, gru_x_s, None)
        gru_h = Int8Mat(gru_h_w, gru_h_s, None)
        mlp_in = Int8Mat(mlp_in_w, mlp_in_s, None)
        mlp_out = Int8Mat(mlp_out_w, mlp_out_s, None)
        
        blk = BlockWeights(ln1, ln2, gru_x, gru_h, mlp_in, mlp_out, gru_b1, gru_b2, mlp_b1, mlp_b2)
        
        return forward_block(ctx, blk, x_in, h_state, tid_base, seed_offset)

    # Compile
    compiled_forward = mx.compile(forward_wrapper)
    
    # Arguments
    args = (
        blk.ln1, blk.ln2, 
        blk.gru_x.weight, blk.gru_x.scale,
        blk.gru_h.weight, blk.gru_h.scale,
        blk.mlp_in.weight, blk.mlp_in.scale,
        blk.mlp_out.weight, blk.mlp_out.scale,
        blk.gru_b1, blk.gru_b2, blk.mlp_b1, blk.mlp_b2,
        ctx.big_rand, ctx.epoch, ctx.base_seed,
        x_in, h_state, tid_base, seed_offset
    )
    
    # Warmup
    print("Warmup...")
    for _ in range(5):
        # Run raw
        out = forward_block(ctx, blk, x_in, h_state, tid_base, seed_offset)
        mx.eval(out)
        # Run compiled
        out_c = compiled_forward(*args)
        mx.eval(out_c)
        
    # Benchmark Raw
    print("Running Raw...")
    start = time.time()
    for _ in range(20):
        out = forward_block(ctx, blk, x_in, h_state, tid_base, seed_offset)
        mx.eval(out)
    end = time.time()
    raw_time = (end - start) / 20
    print(f"Raw time: {raw_time*1000:.3f} ms")
    
    # Benchmark Compiled
    print("Running Compiled...")
    start = time.time()
    for _ in range(20):
        out_c = compiled_forward(*args)
        mx.eval(out_c)
    end = time.time()
    compiled_time = (end - start) / 20
    print(f"Compiled time: {compiled_time*1000:.3f} ms")
    
    print(f"Speedup: {raw_time / compiled_time:.2f}x")

if __name__ == "__main__":
    benchmark_compile()
