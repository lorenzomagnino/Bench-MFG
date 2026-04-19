<div align="center">

<img src="favicon.svg" width="90" alt="Bench-MFG icon"/>

# Bench-MFG

**A benchmark suite for Mean Field Game algorithms**

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://github.com/astral-sh/uv)
[![ruff](https://img.shields.io/badge/ruff-⚡-gold.svg)](https://github.com/astral-sh/ruff)
[![pre-commit](https://img.shields.io/badge/pre--commit-enabled-brightgreen.svg)](https://github.com/pre-commit/pre-commit)
[![arXiv](https://img.shields.io/badge/arXiv-2602.12517-b31b1b.svg)](https://arxiv.org/pdf/2602.12517)

</div>

---

## Overview

Bench-MFG is a unified benchmarking suite for **Mean Field Game (MFG)** algorithms. It covers a diverse set of environments — including the novel **MF Garnet** — and implements both classical and state-of-the-art solvers. Experiments are configured with [Hydra](https://hydra.cc) and hot paths are accelerated with [JAX](https://github.com/google/jax).

### Environments

| Class | Variants |
|---|---|
| No Interaction | Move Forward |
| Contractive Game | Coordination Game |
| Lasry-Lions Game | Beach Bar Problem · *(anti)* Two Beach Bars |
| Potential Game | Four Room Exploration · *(anti)* RockPaperScissor |
| Dynamics-Coupled Game | SIS Epidemic · Kinetic Congestion |
| **MF Garnet** *(novel)* | Random Instances |

### Algorithms

| Category | Algorithms |
|---|---|
| BR-based Fixed Point | Fixed Point · Damped Fixed Point · Fictitious Play |
| Policy-Eval. Based | Policy Iteration · Smoothed PI · Boltzmann PI · Online Mirror Descent |
| Exploitability Min. | **MF-PSO** *(novel)* |

### Features

- **Hydra** config — compose and sweep experiments from the CLI
- **JAX & Python** solvers — run on CPU, GPU, or TPU with minimal changes
- **Logging & plots** — WandB integration, per-run `.npz` results, and publication-ready figures

---

## Quick Start

```bash
# Create and activate environment
uv venv --python 3.11
source .venv/bin/activate

# Install dependencies
uv pip install -e .
```

### Run an experiment

Select your algorithm and environment in `conf/defaults.yaml`, then:

```bash
python main.py
```

General way to run an experiment (after selecting algo and env in `conf/defaults.yaml`):
```bash
python main.py -m \
  experiment.name="omd_sweep" \
  experiment.random_seed=42,10,111,1032 \
  algorithm.omd.learning_rate=0.5,0.05,0.005 \
  algorithm.omd.temperature=0.2,0.5,0.8
```

For detailed instructions on batch runs see [EXPERIMENTS.md](EXPERIMENTS.md).

---

## Configuration

All configuration lives under `conf/` and is managed by Hydra:

| File / Folder | Purpose |
|---|---|
| `conf/defaults.yaml` | Top-level defaults |
| `conf/algorithm/` | Per-algorithm settings (pso, omd, pi, …) |
| `conf/environment/` | Environment configurations |
| `conf/experiment/` | Experiment overrides |
| `conf/logging/` | WandB logging settings |
| `conf/visualization/` | Plot settings |

---

## Repository Structure

```
Bench-MFG/
├── conf/                    # Hydra configuration files
│   ├── defaults.yaml
│   ├── algorithm/
│   ├── environment/
│   ├── experiment/
│   ├── logging/
│   └── visualization/
├── envs/                    # MFG environments
│   ├── mf_garnet/           # MF Garnet (novel)
│   ├── four_rooms_obstacles/
│   ├── lasry_lions_chain/
│   ├── contraction_game/
│   ├── kinetic_congestion/
│   ├── sis_epidemic/
│   └── ...
├── learner/                 # Solver implementations
│   ├── jax/                 # JAX-accelerated solvers
│   └── python/              # Pure-Python solvers
├── utility/                 # Shared utilities
│   ├── create_environment.py
│   ├── create_solver.py
│   ├── run_training.py
│   ├── save_results.py
│   ├── wandb_logger.py
│   └── MFGPlots.py
├── outputs/                 # Experiment results
├── scripts/                 # Helper shell scripts
├── main.py                  # Entry point
└── pyproject.toml           # Project dependencies
```

---

## Outputs

Results are written to `outputs/YYYY-MM-DD/<Env>/<Algorithm>/<Experiment>/`:

| File | Contents |
|---|---|
| `*_results.npz` | Policy, mean field, exploitabilities |
| `mfg_experiment.log` | Full execution log |
