Prototype scaffold for a direct NAX-tile GEMM
============================================

Purpose
-------
Skeleton to experiment with a custom Metal kernel that would use MLX’s Steel/NAX fragments. This scaffold does **not** implement a working GEMM yet; it just sets up the build plumbing (Metal shader compilation + ObjC++ extension). You will need to fill in the Metal kernel with a real NAX-tiled matmul and wire the host to dispatch it.

Layout
------
- `nax_gemm.metal`: placeholder Metal kernel that includes `steel/gemm/nax.h`. Currently writes zeros; replace with a real NAX tile matmul (e.g., 64×64×64, float16 inputs).
- `nax_gemm.mm`: minimal Python extension stub (no host dispatch yet).
- `setup.py`: custom `build_ext` that attempts to compile the Metal shader to a `.metallib` via `xcrun metal/metallib` using the MLX include path.

Building
--------
```
cd extensions/nax_gemm
uv pip install -e .
```
Notes:
- Requires Xcode command-line tools (`xcrun metal`/`metallib`).
- `setup.py` tries to locate the MLX include dir via `python -c "import mlx, os, inspect; print(os.path.join(os.path.dirname(inspect.getfile(mx)),'include'))"`; adjust if needed.
- If Metal compilation fails, the extension still builds but the `.metallib` will be missing; you’ll need to fix paths or kernel code.

Next steps to make it real
--------------------------
1) Implement the kernel in `nax_gemm.metal` using `NAXTile`/`NAXSubTile` from `steel/gemm/nax.h` and a K-loop over 64-chunks; assume row-contiguous float16 inputs, M/N/K multiples of 64.
2) Update `nax_gemm.mm` to load `nax_gemm.metallib`, create MTLBuffers from NumPy arrays, dispatch the kernel with grid=(ceil(N/64), ceil(M/64)), and return a NumPy float32 output.
3) Rebuild (`uv pip install -e .`) and validate with a small known matmul vs. NumPy.
