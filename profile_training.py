
import cProfile
import pstats
import mlx.core as mx
import time
from scripts.train_tinystories_gru_full import main
import sys

# Mock sys.argv to pass arguments to main
sys.argv = [
    "scripts/train_tinystories_gru_full.py",
    "--pop_size", "20",
    "--steps", "3",  # Few steps for profiling
    "--batch_size", "4",
    "--seq_len", "64", # Shorter seq len for faster profiling cycle
    "--group_size", "20"
]

def profile_run():
    print("Starting profiling run...")
    profiler = cProfile.Profile()
    profiler.enable()
    
    try:
        main()
    except KeyboardInterrupt:
        pass
        
    profiler.disable()
    print("Profiling finished. Dumping stats...")
    
    stats = pstats.Stats(profiler).sort_stats('cumtime')
    stats.print_stats(30)
    
    # Also print per-call stats for key functions
    print("\nPer-call stats for eggroll functions:")
    stats.sort_stats('tottime').print_stats('eggroll', 20)

if __name__ == "__main__":
    profile_run()
