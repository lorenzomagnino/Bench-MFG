"""Utility to retrieve and display final exploitability results from Garnet experiments."""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from utility.plot_npz_results import load_exploitabilities

# Garnet version display names
GARNET_DISPLAY_NAMES = {
    "Garnet_5_5_5_add_mult": "5x5x5 (A/M)",
    "Garnet_5_5_5_mult_mult": "5x5x5 (M/M)",
    "Garnet_25_10_10_add_add": "25x10x10 (A/A)",
    "Garnet_25_10_10_mult_add": "25x10x10 (M/A)",
}

# All available algorithms
ALL_ALGORITHMS = [
    "DampedFP_damped",
    "DampedFP_fictitious_play",
    "DampedFP_pure",
    "PI_boltzmann_policy_iteration",
    "PI_smooth_policy_iteration",
    "PI_policy_iteration",
    "OMD",
    "PSO",
]

# Default algorithm selection (all enabled)
DEFAULT_ALGORITHMS = dict.fromkeys(ALL_ALGORITHMS, True)


def find_latest_exploitability_npz(
    instance_dir: Path,
    algorithm: str,
) -> Optional[Path]:
    """Find the latest exploitabilities.npz file for a given instance and algorithm.

    Args:
        instance_dir: Path to the Garnet instance directory (e.g., outputs/Garnet_5_5_5_add_mult/Garnet_1).
        algorithm: Algorithm name (e.g., "OMD", "PSO").

    Returns:
        Path to the latest exploitabilities.npz file, or None if not found.
    """
    algorithm_dir = instance_dir / algorithm
    if not algorithm_dir.exists():
        return None

    # Structure: algorithm/seed_*/config/timestamp/exploitabilities.npz
    npz_files = list(algorithm_dir.glob("seed_*/*/*/exploitabilities.npz"))
    if not npz_files:
        return None

    # Return the most recently modified file
    return max(npz_files, key=lambda p: p.stat().st_mtime)


def collect_final_exploitabilities(
    garnet_version_dir: Path,
    algorithm: str,
    num_instances: int = 10,
) -> List[float]:
    """Collect final exploitability values across all instances for a given algorithm.

    Args:
        garnet_version_dir: Path to the Garnet version directory (e.g., outputs/Garnet_5_5_5_add_mult).
        algorithm: Algorithm name (e.g., "OMD", "PSO").
        num_instances: Number of Garnet instances to check (default: 10).

    Returns:
        List of final exploitability values (one per instance that has data).
    """
    final_values = []

    for i in range(1, num_instances + 1):
        instance_dir = garnet_version_dir / f"Garnet_{i}"
        if not instance_dir.exists():
            continue

        npz_path = find_latest_exploitability_npz(instance_dir, algorithm)
        if npz_path is None:
            continue

        try:
            exploitabilities = load_exploitabilities(npz_path)
            final_value = exploitabilities[-1]  # Get final iteration value
            final_values.append(float(final_value))
        except Exception as e:
            print(f"Warning: Failed to load {npz_path}: {e}")
            continue

    return final_values


def compute_mean_std(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    """Compute mean and standard deviation of a list of values.

    Args:
        values: List of float values.

    Returns:
        Tuple of (mean, std), or (None, None) if list is empty.
    """
    if not values:
        return None, None

    mean = np.mean(values)
    std = np.std(values)
    return float(mean), float(std)


def format_cell(mean: Optional[float], std: Optional[float]) -> str:
    """Format mean and std as a table cell string.

    Args:
        mean: Mean value (or None).
        std: Standard deviation (or None).

    Returns:
        Formatted string like "1.23e-04 +/- 3.00e-05" or "N/A".
    """
    if mean is None or std is None:
        return "N/A"

    # Use scientific notation for small values (< 0.001)
    mean_str = f"{mean:.2e}" if abs(mean) < 0.001 else f"{mean:.4f}"
    std_str = f"{std:.2e}" if abs(std) < 0.001 else f"{std:.4f}"

    return f"{mean_str} +/- {std_str}"


def generate_garnet_results_table(
    algorithms: Optional[Dict[str, bool]] = None,
    outputs_dir: Union[str, Path] = "outputs",
    num_instances: int = 10,
    save_csv: Optional[Union[str, Path]] = None,
    save_latex: Optional[Union[str, Path]] = None,
) -> pd.DataFrame:
    """Generate a table of final exploitability results for Garnet experiments.

    Args:
        algorithms: Dictionary mapping algorithm names to boolean (True to include).
            If None, includes all algorithms.
        outputs_dir: Root directory containing outputs. Defaults to "outputs".
        num_instances: Number of Garnet instances per version (default: 10).
        save_csv: Optional path to save results as CSV.
        save_latex: Optional path to save results as LaTeX table.

    Returns:
        pandas DataFrame with algorithms as rows and Garnet versions as columns.
    """
    if algorithms is None:
        algorithms = DEFAULT_ALGORITHMS

    outputs_dir = Path(outputs_dir)

    # Filter to only selected algorithms
    selected_algorithms = [alg for alg, include in algorithms.items() if include]

    # Garnet versions to process (in order)
    garnet_versions = list(GARNET_DISPLAY_NAMES.keys())

    # Build the results table
    results_data = {}

    for garnet_version in garnet_versions:
        display_name = GARNET_DISPLAY_NAMES[garnet_version]
        garnet_version_dir = outputs_dir / garnet_version

        column_data = []
        for algorithm in selected_algorithms:
            if not garnet_version_dir.exists():
                column_data.append("N/A")
                continue

            final_values = collect_final_exploitabilities(
                garnet_version_dir, algorithm, num_instances
            )
            mean, std = compute_mean_std(final_values)
            cell = format_cell(mean, std)
            column_data.append(cell)

        results_data[display_name] = column_data

    # Create DataFrame
    df = pd.DataFrame(results_data, index=pd.Index(selected_algorithms))
    df.index.name = "Algorithm"

    # Print to console
    print("=" * 100)
    print("Garnet Experiment Results: Final Exploitability (mean +/- std)")
    print("=" * 100)
    print(df.to_string())
    print("=" * 100)

    # Save to CSV if requested
    if save_csv is not None:
        save_csv = Path(save_csv)
        save_csv.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_csv)
        print(f"Results saved to CSV: {save_csv}")

    # Save to LaTeX if requested
    if save_latex is not None:
        save_latex = Path(save_latex)
        save_latex.parent.mkdir(parents=True, exist_ok=True)
        latex_str = df.to_latex(escape=False)
        with open(save_latex, "w") as f:
            f.write(latex_str)
        print(f"Results saved to LaTeX: {save_latex}")

    return df


if __name__ == "__main__":
    # Example usage: generate table with all algorithms
    df = generate_garnet_results_table()

    # Example: select specific algorithms only
    # df = generate_garnet_results_table(
    #     algorithms={"PSO": True, "OMD": True, "DampedFP_damped": True}
    # )
