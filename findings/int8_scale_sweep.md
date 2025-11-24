## Int8 scale sweep (fixed_point × sigma_shift × input magnitude)

- Script: `findings/int8_scale_sweep.py` (run with `PYTHONPATH=. NANO_EGG_NO_CLI=1 uv run python findings/int8_scale_sweep.py`)
- Outputs: `findings/int8_scale_sweep.csv`, `findings/int8_scale_sweep.png`

### Setup
- Inputs/weights set to constant magnitude in int8: `[[mag, mag]]`.
- Forward uses `do_mm` with the Metal int8 kernel, fixed-point division `// (2^fixed_point * sqrt(d))`, and noise enabled (epoch=0, thread_id=0).
- Swept `mag ∈ {1, 2, 4, 8, 16, 32}`, `fixed_point ∈ {2, 3, 4}`, `sigma_shift ∈ {0, 2, 4}`.
- Recorded `max_output` after divide/clamp to int8.

### Visuals
- Heatmap with annotated values: `findings/int8_scale_sweep.png` (0–127 scale; 127 means clamp).
- Grouped bars by sigma_shift per fixed_point: `findings/int8_scale_bars.png`.

### Key observations
- `mag=1` underflows to 0 across all shifts/FP settings.
- At `fixed_point=4`, nonzero outputs appear at `mag≈4`; `mag≥8–16` yields healthy range, larger mags saturate at 127.
- Lowering `fixed_point` reduces the divisor; e.g., `mag=2` survives at `fp=2` but not at `fp=4`.
- `sigma_shift` mostly affects the noise term; the dominant underflow driver is the `(2^fixed_point * sqrt(d))` divisor vs input magnitude.
- Measured RSS stayed flat (~process baseline) across the sweep; memory usage does not meaningfully vary with these small configs.

See the annotated heatmap and bars plus the full table (`int8_scale_sweep.csv`) for the grid. Adjust fixed_point or input scaling upward to avoid underflow; too high input causes saturation at 127. Shift choice balances underflow vs saturation for the target activation scale.
