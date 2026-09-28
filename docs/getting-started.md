# Getting started

## Installation

BenchMFG requires Python 3.10 or newer. The default installation uses CPU-compatible JAX:

```bash
pip install bench-mfg-suite
```

If you use uv, run `uv add bench-mfg-suite` in your project instead.
For Linux with an NVIDIA GPU and pip-managed CUDA runtime components:

```bash
pip install "bench-mfg-suite[cuda12]"
```

## Explore the available configurations

```bash
benchmfg hello
benchmfg env list
benchmfg algo list
benchmfg algo-parameters
```

The list commands show the configurations available in your installed version.
`algo-parameters` explains the algorithm settings and how to override them.

## Run and plot an experiment

Run Online Mirror Descent for 20 iterations on the four-rooms environment:

```bash
benchmfg train algorithm=omd environment=four_rooms_obstacles device=cpu algorithm.omd.num_iterations=20
```

Results are saved under:

```text
outputs/<Env>/<Algorithm>/seed_<seed>/<Experiment>/<run_id>/
```

Each run includes its configuration, metrics, final policy, and final mean field.
Replace `<run_dir>` below with the directory created by your run:

```bash
benchmfg plot single-run <run_dir>
```

## Run a parameter sweep

Comma-separated values select multiple seeds or parameter values:

```bash
benchmfg sweep \
  algorithm=omd environment=lasry_lions_chain device=cpu \
  experiment.random_seed=42,10 \
  algorithm.omd.learning_rate=0.05,0.005
```

For comparisons across random game instances, see [MF-Garnet](MFG_GARNET.md).
To implement your own environment or algorithm, see [Extending BenchMFG](EXTENDING.md).
