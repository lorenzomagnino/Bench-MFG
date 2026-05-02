"""Filesystem discovery and label utilities for experiment results."""

import contextlib
from pathlib import Path
from typing import Any

import numpy as np
from utility.plot_loaders import load_exploitabilities, load_runtime
import yaml

ALGORITHMS = [
    "PSO",
    "OMD",
    "DampedFP_damped",
    "DampedFP_fictitious_play",
    "DampedFP_pure",
    "PI_policy_iteration",
    "PI_smooth_policy_iteration",
    "PI_boltzmann_policy_iteration",
]

DEFAULT_COMPARISON_ALGORITHMS = dict.fromkeys(ALGORITHMS, True)


def _get_environment_results_dir(
    environment: str, results_dir: str | Path | None = None
) -> Path:
    """Return the results directory for one environment."""
    if results_dir is None:
        project_root = Path(__file__).parent.parent
        return project_root / "results" / environment
    return Path(results_dir) / environment


def _get_algorithm_results_dir(
    environment: str, algorithm: str, results_dir: str | Path | None = None
) -> Path:
    """Return the results directory for one environment/algorithm."""
    return _get_environment_results_dir(environment, results_dir) / algorithm


def _latest_timestamp_dir(version_dir: Path) -> Path | None:
    """Return the latest timestamped run directory that contains exploitability data."""
    candidates = [
        path
        for path in version_dir.iterdir()
        if path.is_dir() and (path / "exploitabilities.npz").exists()
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda path: (path.stat().st_mtime, path.name))


def _write_yaml(path: Path, data: dict[str, Any]) -> None:
    """Write YAML data to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def build_results_yaml_for_algorithm(
    environment: str,
    algorithm: str,
    outputs_dir: str | Path = "outputs",
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Build `results.yaml` for one algorithm using the latest run per seed/config."""
    outputs_dir = Path(outputs_dir)
    algorithm_dir = outputs_dir / environment / algorithm

    if not algorithm_dir.exists():
        raise FileNotFoundError(f"Algorithm directory not found: {algorithm_dir}")

    configurations: list[dict[str, Any]] = []
    version_names: set[str] = set()
    for seed_dir in algorithm_dir.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        for version_dir in seed_dir.iterdir():
            if version_dir.is_dir():
                version_names.add(version_dir.name)

    for version_name in sorted(version_names):
        seeds: list[dict[str, Any]] = []
        for seed_dir in sorted(algorithm_dir.iterdir()):
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            version_dir = seed_dir / version_name
            if not version_dir.exists():
                continue
            latest_run_dir = _latest_timestamp_dir(version_dir)
            if latest_run_dir is None:
                continue
            exploitabilities = load_exploitabilities(
                latest_run_dir / "exploitabilities.npz"
            )
            metrics_path = latest_run_dir / "metrics.npz"
            runtime_s = load_runtime(metrics_path) if metrics_path.exists() else None
            seeds.append(
                {
                    "seed": seed_dir.name,
                    "run_id": latest_run_dir.name,
                    "runtime_s": float(runtime_s) if runtime_s is not None else None,
                    "exploitabilities": [float(value) for value in exploitabilities],
                    "final_exploitability": float(exploitabilities[-1]),
                }
            )

        if not seeds:
            continue

        final_values = np.array([seed["final_exploitability"] for seed in seeds])
        configurations.append(
            {
                "version": version_name,
                "seeds": seeds,
                "num_seeds": len(seeds),
                "final_mean_exploitability": float(np.mean(final_values)),
                "final_std_exploitability": float(np.std(final_values)),
            }
        )

    configurations.sort(key=lambda config: config["final_mean_exploitability"])

    results_data = {
        "environment": environment,
        "algorithm": algorithm,
        "selection_policy": "latest_run",
        "configurations": configurations,
    }
    results_path = (
        _get_algorithm_results_dir(environment, algorithm, results_dir) / "results.yaml"
    )
    _write_yaml(results_path, results_data)
    return results_data


def write_best_model_yaml(
    results_data: dict[str, Any],
    environment: str,
    algorithm: str,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Write `best_model.yaml` from algorithm `results.yaml` data."""
    configurations = results_data.get("configurations", [])
    if not configurations:
        raise ValueError(
            f"No configuration data available for algorithm '{algorithm}' in environment '{environment}'."
        )

    best_configuration = configurations[0]
    best_model_data = {
        "environment": environment,
        "algorithm": algorithm,
        "selection_policy": "latest_run",
        "best_version": best_configuration["version"],
        "num_seeds": best_configuration["num_seeds"],
        "final_mean_exploitability": best_configuration["final_mean_exploitability"],
        "final_std_exploitability": best_configuration["final_std_exploitability"],
        "seeds": best_configuration["seeds"],
        "all_versions": [
            {
                "version": configuration["version"],
                "final_mean_exploitability": configuration["final_mean_exploitability"],
            }
            for configuration in configurations
        ],
    }
    best_model_path = (
        _get_algorithm_results_dir(environment, algorithm, results_dir)
        / "best_model.yaml"
    )
    _write_yaml(best_model_path, best_model_data)
    return best_model_data


def load_algorithm_results_yaml(
    environment: str,
    algorithm: str,
    results_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Load `results.yaml` for one algorithm."""
    yaml_path = (
        _get_algorithm_results_dir(environment, algorithm, results_dir) / "results.yaml"
    )
    if not yaml_path.exists():
        raise FileNotFoundError(f"Algorithm results YAML not found: {yaml_path}")
    with open(yaml_path) as f:
        data = yaml.safe_load(f)
    if not data or "configurations" not in data:
        raise ValueError(f"Invalid results YAML: {yaml_path}")
    return data


def version_to_algorithm_dir(version_withhyper: str) -> str:
    """Map version_withhyper string to its algorithm directory name.

    Example: "pso_sweep_temp0p20" → "PSO"
    """
    v = version_withhyper.lower()
    if "pso_sweep" in v:
        return "PSO"
    elif "omd_sweep" in v:
        return "OMD"
    elif "smooth_pi_sweep" in v or "smooth_policy_iteration" in v:
        return "PI_smooth_policy_iteration"
    elif "boltzmann_pi_sweep" in v or "boltzmann_policy_iteration" in v:
        return "PI_boltzmann_policy_iteration"
    elif "policy_iteration_sweep" in v:
        return "PI_policy_iteration"
    elif "fplay_sweep" in v or "fictitious" in v:
        return "DampedFP_fictitious_play"
    elif "pure_fp_sweep" in v or v.startswith("pure"):
        return "DampedFP_pure"
    elif "damped_sweep" in v:
        return "DampedFP_damped"
    else:
        parts = version_withhyper.split("_")
        return parts[0].capitalize() if parts else version_withhyper


def version_to_algorithm_name(version_withhyper: str) -> str:
    """Convert version_withhyper to a short human-readable algorithm name.

    Example: "damped_sweep_damped0p10" → "Damped FP"
    """
    v = version_withhyper.lower()
    if "pure_fp" in v or v.startswith("pure"):
        return "Fixed Point (FP)"
    elif "fplay" in v or "fictitious" in v:
        return "Fictitious Play"
    elif "smooth_policy_iteration" in v or "smooth_pi_sweep" in v:
        return "Smooth PI"
    elif "boltzmann_policy_iteration" in v or "boltzmann_pi_sweep" in v:
        return "Boltzmann PI"
    elif "omd_sweep" in v:
        return "OMD"
    elif "pso_sweep" in v:
        return "PSO"
    elif "damped_sweep_damped" in v:
        return "Damped FP"
    elif "policy_iteration_sweep" in v:
        return "PI"
    else:
        return version_withhyper


def extract_hyperparameters(version_withhyper: str) -> str:
    """Extract hyperparameters from a version name as a compact LaTeX-ready string.

    Example: "pso_sweep_temp0p20_w0p30_c10p30_c21p20" → "$\\tau$, w, c1, c2 = 0.2, 0.30, 0.30, 1.20"
    """
    parts = version_withhyper.split("_")
    temp_value = damped_value = lr_value = w_value = c1_value = c2_value = None

    for part in parts:
        if part.startswith("damped") and "damped" in part:
            with contextlib.suppress(ValueError):
                damped_value = float(part.replace("damped", "").replace("p", "."))
        elif part.startswith("temp"):
            with contextlib.suppress(ValueError):
                temp_value = float(part.replace("temp", "").replace("p", "."))
        elif part.startswith("lr"):
            with contextlib.suppress(ValueError):
                lr_value = float(part.replace("lr", "").replace("p", "."))
        elif part.startswith("w") and len(part) > 1:
            with contextlib.suppress(ValueError):
                w_value = float(part.replace("w", "").replace("p", "."))
        elif part.startswith("c1"):
            with contextlib.suppress(ValueError):
                c1_value = float(part.replace("c1", "").replace("p", "."))
        elif part.startswith("c2"):
            with contextlib.suppress(ValueError):
                c2_value = float(part.replace("c2", "").replace("p", "."))

    param_names: list[str] = []
    param_values: list[str] = []

    if temp_value is not None:
        param_names.append("$\\tau$")
        param_values.append(f"{temp_value:.1f}")
    if damped_value is not None:
        param_names.append("$\\lambda$")
        param_values.append(f"{damped_value:.1f}")
    if lr_value is not None:
        param_names.append("lr")
        param_values.append(f"{lr_value:.4f}")
    if w_value is not None:
        param_names.append("w")
        param_values.append(f"{w_value:.2f}")
    if c1_value is not None:
        param_names.append("c1")
        param_values.append(f"{c1_value:.2f}")
    if c2_value is not None:
        param_names.append("c2")
        param_values.append(f"{c2_value:.2f}")

    if param_names and param_values:
        return f"{', '.join(param_names)} = {', '.join(param_values)}"
    return ""


def group_exploitabilities_by_seed(
    environment: str,
    outputs_dir: str | Path = "outputs",
    results_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Group exploitabilities by version and seed from `results.yaml` files."""
    env_results_dir = _get_environment_results_dir(environment, results_dir)
    if not env_results_dir.exists():
        raise FileNotFoundError(
            f"Environment results directory not found: {env_results_dir}"
        )

    result: dict[str, dict[str, Any]] = {}
    for yaml_path in sorted(env_results_dir.glob("*/results.yaml")):
        with open(yaml_path) as f:
            yaml_data = yaml.safe_load(f)
        if not yaml_data:
            continue
        algorithm = yaml_data.get("algorithm")
        for configuration in yaml_data.get("configurations", []):
            version_name = configuration["version"]
            seed_records = list(configuration.get("seeds", []))
            seed_records.sort(key=lambda record: record["seed"])
            result[version_name] = {
                "groups": [
                    [np.array(seed_record["exploitabilities"], dtype=float)]
                    for seed_record in seed_records
                ],
                "seed_names": [seed_record["seed"] for seed_record in seed_records],
                "seed_records": seed_records,
                "final_mean_exploitability": configuration.get(
                    "final_mean_exploitability"
                ),
                "algorithm": algorithm,
            }
    return result


def group_runtimes_by_seed(
    environment: str,
    outputs_dir: str | Path = "outputs",
    results_dir: str | Path | None = None,
) -> dict[str, dict[str, Any]]:
    """Group wall-clock runtimes by version and seed from `results.yaml` files."""
    exploitability_groups = group_exploitabilities_by_seed(
        environment, outputs_dir=outputs_dir, results_dir=results_dir
    )
    result: dict[str, dict[str, Any]] = {}
    for version_name, version_data in exploitability_groups.items():
        runtimes = [
            seed_record["runtime_s"]
            for seed_record in version_data.get("seed_records", [])
            if seed_record.get("runtime_s") is not None
        ]
        result[version_name] = {
            "runtimes": runtimes,
            "seed_names": version_data.get("seed_names", []),
        }
    return result


def get_versions_for_algorithm(
    environment: str,
    algorithm: str,
    outputs_dir: str | Path = "outputs",
) -> list[str]:
    """Return all version names listed in `results.yaml` for one algorithm."""
    results_data = load_algorithm_results_yaml(environment, algorithm)
    return [
        configuration["version"]
        for configuration in results_data.get("configurations", [])
    ]


def get_versions_for_comparison(
    environment: str,
    fixed_versions: list[str] | None = None,
    results_dir: str | Path | None = None,
) -> list[str]:
    """Combine fixed versions with best-rank-1 versions from *_best_models.yaml files.

    Run plot_exploitability_by_algorithm (in plot_sweep.py) for each algorithm first
    to generate the YAML files this function reads.
    """
    if fixed_versions is None:
        fixed_versions = [
            "pure_fp_sweep",
            "fplay_sweep",
            "policy_iteration_sweep_temp0p20",
        ]

    if results_dir is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
    else:
        results_dir = Path(results_dir) / environment

    best_from_yaml: list[str] = []
    if results_dir.exists():
        yaml_files = list(results_dir.glob("*/best_models.yaml")) + list(
            results_dir.glob("*_best_models.yaml")
        )
        for yaml_file in yaml_files:
            try:
                with open(yaml_file) as f:
                    yaml_data = yaml.safe_load(f)
                if (
                    yaml_data
                    and "best_models" in yaml_data
                    and yaml_data["best_models"]
                ):
                    rank1 = yaml_data["best_models"][0]
                    if rank1.get("rank") == 1:
                        best_from_yaml.append(rank1["version"])
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")

    return list(dict.fromkeys(fixed_versions + best_from_yaml))


def get_best_versions_by_algorithm(
    environment: str,
    algorithms: list[str] | None = None,
    results_dir: str | Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    """Return best version per algorithm and a list of missing algorithms."""
    if algorithms is None:
        algorithms = list(DEFAULT_COMPARISON_ALGORITHMS.keys())

    if results_dir is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
    else:
        results_dir = Path(results_dir) / environment

    best_versions: dict[str, str] = {}
    missing_algorithms: list[str] = []

    for algorithm in algorithms:
        yaml_path = results_dir / algorithm / "best_model.yaml"

        if not yaml_path.exists():
            missing_algorithms.append(algorithm)
            continue

        try:
            with open(yaml_path) as f:
                yaml_data = yaml.safe_load(f)
        except Exception:
            missing_algorithms.append(algorithm)
            continue

        version = yaml_data.get("best_version") if yaml_data else None
        if version is None:
            missing_algorithms.append(algorithm)
            continue

        best_versions[algorithm] = version

    return best_versions, missing_algorithms


def get_four_rooms_walls(grid_dim: tuple) -> np.ndarray:
    """Generate a walls mask for FourRoomsAversion2D (0 = wall, 1 = free)."""
    n_rows, n_cols = grid_dim
    N_flat = n_rows * n_cols
    walls = np.ones(N_flat, dtype=int)
    mid_row, mid_col = n_rows // 2, n_cols // 2
    doors = {(2, 5), (8, 5), (5, 8), (5, 2)}

    for row in range(n_rows):
        if (row, mid_col) not in doors:
            idx = row * n_cols + mid_col
            if 0 <= idx < N_flat:
                walls[idx] = 0

    for col in range(n_cols):
        if (mid_row, col) not in doors:
            idx = mid_row * n_cols + col
            if 0 <= idx < N_flat:
                walls[idx] = 0

    return walls
