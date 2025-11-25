import sys
print("Starting imports...")
try:
    from scripts.train_tinystories_gru_int8 import load_tokens
    print("Imported load_tokens")
except ImportError as e:
    print(f"Failed to import load_tokens: {e}")

try:
    import mlx.core as mx
    print("Imported mlx")
except ImportError as e:
    print(f"Failed to import mlx: {e}")

print("Done.")
