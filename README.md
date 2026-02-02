# Bench-MFG

[![Hydra](https://img.shields.io/badge/Hydra-config-red.svg)](https://hydra.cc/)
[![uv](https://img.shields.io/badge/uv-package%20manager-purple.svg)](https://github.com/astral-sh/uv)
[![JAX](https://img.shields.io/badge/JAX-framework-orange.svg)](https://github.com/google/jax)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

Benchmark methods for Mean Field Games (MFG).

### Environments

<table>
<thead>
<tr style="background-color: #4A90E2;">
<th style="padding: 8px; text-align: left; color: white;"><strong>Environment</strong></th>
<th style="padding: 8px; text-align: left; color: white;"><strong>Variants</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 8px;"><strong>No Interaction</strong></td>
<td style="padding: 8px;">• Move Forward</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Contractive Game</strong></td>
<td style="padding: 8px;">• Coordination Game</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Lasry-Lions Game</strong></td>
<td style="padding: 8px;">• Beach Bar Problem<br>• <em>(anti)</em> Two Beach Bars</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Potential Game</strong></td>
<td style="padding: 8px;">• Four Room Exploration<br>• <em>(anti)</em> RockPaperScissor</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Dynamics-Coupled Game</strong></td>
<td style="padding: 8px;">• SIS Epidemic<br>• Kinetic Congestion</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>MF Garnet</strong> <em>(novel!)</em></td>
<td style="padding: 8px;">• Random Instances</td>
</tr>
</tbody>
</table>

### Algorithms

<table>
<thead>
<tr style="background-color: #FF8C42;">
<th style="padding: 8px; text-align: left; color: white;"><strong>Category</strong></th>
<th style="padding: 8px; text-align: left; color: white;"><strong>Algorithms</strong></th>
</tr>
</thead>
<tbody>
<tr>
<td style="padding: 8px;"><strong>BR-based Fixed Point</strong></td>
<td style="padding: 8px;">• Fixed Point<br>• Damped Fixed Point<br>• Fictitious Play</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Policy-Eval. Based</strong></td>
<td style="padding: 8px;">• Policy Iteration (PI)<br>• Smoothed PI<br>• Boltzmann PI<br>• Online Mirror Descent</td>
</tr>
<tr>
<td style="padding: 8px;"><strong>Exploitability Min.</strong></td>
<td style="padding: 8px;">• MF-PSO <em>(novel!)</em></td>
</tr>
</tbody>
</table>

**Framework Features:** ✓ Hydra • ✓ JAX & Python • ✓ Log, Save and Plot

## Easy Start

```bash
# Setup environment
uv venv --python 3.11
source .venv/bin/activate
uv pip install -e .
```
### Run experiment
Modify the `config/defaults.yaml` selecting algorithm and environment. Then run:
```bash
python main.py
```

## Running Experiments
See [EXPERIMENTS.md](EXPERIMENTS.md) for detailed instructions on how to run experiments in batch.

### Configuration

Edit configuration files in `conf/`:
- `conf/defaults.yaml`: Main configuration
- `conf/algorithm/`: Algorithm-specific settings
- `conf/environment/`: Environment configurations
- `conf/logging/logging.yaml`: WandB logging settings

## Repository Structure

```
Bench-MFG/
├── conf/                    # Configuration files (Hydra)
│   ├── defaults.yaml        # Main configuration
│   ├── algorithm/           # Algorithm configs (pso, omd, pi, etc.)
│   ├── environment/         # Environment configs
│   ├── experiment/          # Experiment settings
│   ├── logging/             # Logging configuration
│   └── visualization/       # Plotting settings
├── envs/                    # MFG environments
│   ├── mf_garnet/          # MF Garnet (novel)
│   ├── four_rooms_obstacles/
│   ├── lasry_lions_chain/
│   ├── contraction_game/
│   ├── kinetic_congestion/
│   ├── sis_epidemic/
│   └── ...                  # Other environments
├── learner/                 # Algorithm implementations
│   ├── jax/                 # JAX implementations
│   └── python/              # Python implementations
├── utility/                 # Utilities
│   ├── create_environment.py
│   ├── create_solver.py
│   ├── run_training.py
│   ├── save_results.py
│   ├── wandb_logger.py
│   └── MFGPlots.py
├── outputs/                 # Experiment results
├── scripts/                 # Shell scripts
├── main.py                  # Entry point
└── pyproject.toml           # Project dependencies
```

## Outputs

Results are saved to `outputs/YYYY-MM-DD/EnvironmentName/AlgorithmName/ExperimentName/`:
- `*_results.npz`: Numerical results (policy, mean field, exploitabilities)
- `mfg_experiment.log`: Execution logs
