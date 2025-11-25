# Parity TODO (TinyStories GRU/EGGROLL)

1) Biases and LN params updated  
   - Add biases to GRU/MLP/head and include them in perturb/update.  
   - Update layernorm scales (ln1/ln2/ln_out) via apply_sign_update.  
   - Acceptance: test shows biases/LN params change after a step; no shape/dtype errors; parity tests still pass.

2) Per-matmul shift constants (fan-in aware)  
   - Use per-layer shift = fixed_point + floor(log2(sqrt(fan_in))) + sigma_shift for each matmul (in→out, expand/proj, head).  
   - Acceptance: do_mm/do_mm_batched use the correct shift per matmul; delta scaling matches reference; parity tests pass.

3) Fitness normalization (CLT path)  
   - In convert_fitnesses when fast_fitness=False, normalize paired diffs by RMS (CLT-style) to reduce variance sensitivity.  
   - Acceptance: convert_fitnesses returns zero-mean, unit-ish scaled values on synthetic input; tests updated/passing.

4) Stateful data/noise reuse semantics  
   - Honor noise_reuse/group_size fully: reuse noise slices across epochs per noise_reuse; share batches/states across group_size antithetic pairs.  
   - Acceptance: unit test showing identical perturbations across noise_reuse epochs; group_size reuses data per pair; states carried consistently when enabled.

5) Alpha/threshold schedule mirroring reference  
   - Implement alpha/threshold decay (e.g., linear or reference schedule) so vote gating and step size reduce over time.  
   - Acceptance: CLI flags to set initial/final alpha/threshold; schedule applied each step; tests confirm values change over steps.
