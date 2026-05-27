# MF-Garnet

MF-Garnet is a random Mean Field Game generator. It is useful for benchmarking
algorithms across many controlled, reproducible game instances instead of a
single hand-designed environment.

## What It Generates

Each instance is controlled by `environment.reward.mfgarnet.seed`. For a fixed
seed, BenchMFG builds:

- a sparse base transition kernel `P0`;
- a transition coupling tensor `C`;
- a base reward table `R0`;
- a mean-field reward interaction matrix `M`.

Transitions can be `additive` or `multiplicative`:

```text
additive:       intensity = cp * P0(s,a) + rho_p * (C[s,a] @ mu)
multiplicative: intensity = P0(s,a) * (cp + rho_p * (C[s,a] @ mu))
```

Rewards can also be `additive` or `multiplicative`:

```text
additive:       r = cr * R0[s,a] + rho_r * (M[s] @ mu)
multiplicative: r = R0[s,a] * (cr + rho_r * (M[s] @ mu))
```

`game_type=potential` symmetrizes `M`; `game_type=cyclic` anti-symmetrizes it.

## Quick Run

```bash
benchmfg train \
  environment=mf_garnet \
  algorithm=omd \
  device=cpu \
  environment.num_states=5 \
  environment.num_actions=5 \
  environment.reward.mfgarnet.seed=0 \
  environment.reward.mfgarnet.branching_factor=5 \
  environment.reward.mfgarnet.dynamics_structure=additive \
  environment.reward.mfgarnet.reward_structure=multiplicative \
  algorithm.omd.num_iterations=20
```

## Benchmark Protocol

Use paired seeds when comparing algorithms:

- vary `environment.reward.mfgarnet.seed` to generate different games;
- vary `experiment.random_seed` for algorithm initialization;
- keep model size and algorithm hyperparameters fixed inside one comparison.

Example pattern:

```bash
benchmfg train environment=mf_garnet algorithm=pso \
  environment.reward.mfgarnet.seed=0 \
  experiment.random_seed=42

benchmfg train environment=mf_garnet algorithm=pso \
  environment.reward.mfgarnet.seed=1 \
  experiment.random_seed=10
```

The helper scripts in `scripts/garnet/` automate this protocol.

## Key Config Fields

```yaml
environment:
  num_states: 5
  num_actions: 5
  reward:
    mfgarnet:
      seed: 0
      branching_factor: 5
      dynamics_structure: additive       # additive | multiplicative
      reward_structure: multiplicative   # additive | multiplicative
      game_type: potential               # potential | cyclic
      cp: 0.5
      rho_p: 0.5
      cr: 0.5
      rho_r: 0.5
      reward_scale: 1.0
```

## Outputs

Runs are saved under the normal BenchMFG output root, with a Garnet-specific
directory that records the model class:

```text
outputs/Garnet_<states>_<actions>_<branching>_<dyn>_<rew>/Garnet_<seed>/...
```

Use the standard plotting commands after a run:

```bash
benchmfg plot single-run <run_dir>
benchmfg plot sweep <environment> <algorithm>
benchmfg plot compare <environment>
```

`plot sweep` uses `outputs/` unless `--outputs-dir` is provided. Within each
Garnet seed/version directory it selects the latest timestamped run containing
`exploitabilities.npz`; `plot compare` then consumes the best-model YAML files
created by the sweep step.
