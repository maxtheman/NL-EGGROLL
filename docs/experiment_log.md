## Experiment Log

Date: 2024-04-08 (local run)

- **scripts/exp_1_first_repro.py**
  - Command: `python scripts/exp_1_first_repro.py`
  - Config (printed): task=linear, input_dim=256, output_dim=16, samples=2048, steps=200, eval_budget=6400, pop_size=32, sigma=0.05, lr_es=0.05, lr_sgd=0.1, rank=2, antithetic=True, seed=42, fast_fitness=True, update_threshold=0.0, noise_reuse=1, group_size=2, dtype=float32, mode_map=full, use_sign_updates=True, param_clip=1.0, int_mode=False, z_threshold=0.0, stride=4096
  - Results: sgd loss=3.4251 (200 evals, 0.242s); es_full loss=268.0047 (6400 evals, 1.327s); eggroll_low_rank loss=268.8207 (6400 evals, 1.439s); bandwidth ratio=0.133.

- **scripts/exp_2_mlp_eggroll_vs_adam.py**
  - Command: `python scripts/exp_2_mlp_eggroll_vs_adam.py`
  - Config: samples=1024, noise=0.1, test_size=0.2, input_dim=2, hidden_dim=16, output_dim=2, steps=200, eval_budget=6400, pop_size=32, sigma=0.05, lr_es=0.05, lr_adam=0.01, rank=2, antithetic=True, seed=42, fast_fitness=True, update_threshold=0.0, dtype=float32, use_sign_updates=True, param_clip=1.0, int_mode=False, z_threshold=0.0
  - Results: adam loss=0.2420 / acc=0.8683 (200 evals, 0.547s); es_full loss=0.4967 / acc=0.7707 (6400 evals, 3.160s); eggroll_low_rank loss=0.2568 / acc=0.8732 (6400 evals, 3.132s); bandwidth ratio=0.562.

- **scripts/exp_3_batched_eggroll.py**
  - Command: `python scripts/exp_3_batched_eggroll.py`
  - Config: defaults (pop_size=32, steps=200, sigma=0.05, int_mode=False)
  - Results: final loss=0.4449, test acc=0.8537, wall time=0.430s; logits remained finite (periodic stats printed every 10 steps).

- **scripts/eggroll_benchmark.py**
  - Command: `python scripts/eggroll_benchmark.py`
  - Config: task=linear, input_dim=256, output_dim=16, samples=2048, steps=200, eval_budget=6400, pop_size=32, sigma=0.05, lr_es=0.05, lr_sgd=0.1, rank=2, antithetic=True, seed=42, fast_fitness=True, update_threshold=0.0, noise_reuse=1, group_size=2, dtype=float32, mode_map=full, use_sign_updates=True, param_clip=1.0, int_mode=False, z_threshold=0.0, stride=4096
  - Results: sgd loss=3.4331 (200 evals, 0.090s); es_full loss=288.1144 (6400 evals, 1.382s); eggroll_low_rank loss=300.7085 (6400 evals, 1.515s); bandwidth ratio=0.133.

- **scripts/eggroll_benchmark_v2.py**
  - Command: `python scripts/eggroll_benchmark_v2.py`
  - Config: task=linear, input_dim=256, output_dim=16, samples=2048, steps=200, eval_budget=6400, pop_size=32, sigma=0.05, lr_es=0.05, lr_sgd=0.1, rank=2, antithetic=True, seed=42, fast_fitness=True, update_threshold=0.0, noise_reuse=1, group_size=2, dtype=float32, mode_map=full, use_sign_updates=True, param_clip=1.0, int_mode=False, z_threshold=0.0, stride=4096
  - Results: sgd loss=3.4272 (200 evals, 0.091s); es_full loss=292.7211 (6400 evals, 1.365s); eggroll_low_rank loss=308.5578 (6400 evals, 1.493s); bandwidth ratio=0.133.

- **scripts/exp_final_benchmark.py**
  - Command: `python scripts/exp_final_benchmark.py`
  - Status: failed to start. Traceback: `NameError: name 'ExperimentConfig' is not defined` at line 313 when constructing cfg. Needs ExperimentConfig definition or import before argparse block.
