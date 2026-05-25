"""Data loading utilities for NPZ/YAML result files."""

from pathlib import Path

import numpy as np
import yaml


def load_exploitabilities(npz_path: str | Path) -> np.ndarray:
    """Load exploitabilities from an NPZ file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if "exploitabilities" in data:
            return np.array(data["exploitabilities"])
        raise ValueError(
            f"NPZ file {npz_path} does not contain 'exploitabilities' key. "
            f"Available keys: {list(data.keys())}"
        )


def load_mean_field(npz_path: str | Path) -> np.ndarray:
    """Load final mean field from an NPZ file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if "mean_field" in data:
            return np.array(data["mean_field"])
        raise ValueError(
            f"NPZ file {npz_path} does not contain 'mean_field' key. "
            f"Available keys: {list(data.keys())}"
        )


def load_policy(npz_path: str | Path) -> np.ndarray:
    """Load final policy from an NPZ file (shape: N_states x N_actions)."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if "policy" in data:
            return np.array(data["policy"])
        raise ValueError(
            f"NPZ file {npz_path} does not contain 'policy' key. "
            f"Available keys: {list(data.keys())}"
        )


def load_runtime(npz_path: str | Path) -> float:
    """Load wall-clock runtime in seconds from a metrics.npz file."""
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")
    with np.load(npz_path, allow_pickle=True) as data:
        if "runtime_s" not in data:
            raise ValueError(
                f"{npz_path} does not contain 'runtime_s'. Available: {list(data.keys())}"
            )
        return float(data["runtime_s"])


def find_best_model_npz(
    environment: str,
    npz_filename: str,
    outputs_dir: str | Path = "outputs",
    seed: int = 42,
) -> Path:
    """Find the NPZ file for the best model from results/{environment}/best/best_model.yaml.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        npz_filename: Name of the NPZ file to find (e.g., "final_mean_field.npz").
        outputs_dir: Root directory containing outputs.
        seed: Seed number to use.

    Returns:
        Path to the NPZ file.
    """
    project_root = Path.cwd()

    # Fast path: npz already collected into results/ by collect_best_artifacts
    cached = project_root / "results" / environment / "best" / npz_filename
    if cached.exists():
        return cached

    yaml_path = project_root / "results" / environment / "best" / "best_model.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Best model YAML not found: {yaml_path}. "
            "Please run plot_exploitability_multiple_versions first to generate it."
        )

    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    best_version = yaml_data.get("best_version")
    if not best_version:
        raise ValueError(f"No 'best_version' found in {yaml_path}")

    from benchmfg.utility.plot_discovery import version_to_algorithm_dir

    algorithm_dir = version_to_algorithm_dir(best_version)
    outputs_dir = Path(outputs_dir)
    seed_dir = outputs_dir / environment / algorithm_dir / f"seed_{seed}" / best_version

    if not seed_dir.exists():
        raise FileNotFoundError(
            f"Directory not found: {seed_dir}. "
            f"Make sure seed_{seed} exists for version '{best_version}'."
        )

    timestamp_dirs = [d for d in seed_dir.iterdir() if d.is_dir()]
    if not timestamp_dirs:
        raise FileNotFoundError(f"No timestamp directories found in {seed_dir}")

    latest = max(timestamp_dirs, key=lambda d: d.stat().st_mtime)
    npz_path = latest / npz_filename
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    return npz_path
