## Fitness scaling and ES update notes

- The current code uses `convert_fitnesses` with optional fast_fitness/sign reduction. This can distort the ES gradient estimate, especially on simple tasks where the raw reward has a clean signal.
- For a more accurate gradient estimate on toy tasks, you can bypass `convert_fitnesses` and feed raw rewards directly into `apply_sign_update` (i.e., no sign/normalization). That is a code change and deviates from the reference.
- In the Eggroll reference (see `run.py` and the C impl in `third_party`), the update uses paired antithetic samples and fast_fitness/sign logic designed for the integer/bit-shift path. Keeping that matches the reference; swapping to raw rewards is a deliberate deviation for stability/signal clarity on simple FP16 experiments.
