"""Plot utilities for loading and visualizing saved NPZ results."""

import contextlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import matplotlib
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import yaml

from conf.visualization.visualization_schema import ColorsConfig
from utility.MFGPlots import plot_mean_field, plot_policy

matplotlib.use("Agg")  # Use non-interactive backend to prevent display


class HandlerMarker(HandlerLine2D):
    """Custom legend handler that shows only the marker, no line."""

    def create_artists(
        self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans
    ):
        try:
            if isinstance(orig_handle, Line2D):
                color = orig_handle.get_color()
                marker = orig_handle.get_marker()
                if marker is None or marker == "None" or marker == "":
                    marker = "o"
            else:
                color = getattr(orig_handle, "_color", "black")
                marker = "o"
        except (AttributeError, TypeError):
            color = "black"
            marker = "o"
        return [
            Line2D(
                [width / 2],
                [height / 2],
                marker=marker,
                markersize=fontsize * 0.8,
                markerfacecolor=color,
                markeredgecolor=color,
                linestyle="None",
            )
        ]


HandlerCircle = HandlerMarker


def load_exploitabilities(npz_path: Union[str, Path]) -> np.ndarray:
    """Load exploitabilities from an NPZ file.

    Args:
        npz_path: Path to the exploitabilities.npz file.

    Returns:
        Array of exploitability values (1D array).
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "exploitabilities" in data:
            return np.array(data["exploitabilities"])
        else:
            raise ValueError(
                f"NPZ file {npz_path} does not contain 'exploitabilities' key. "
                f"Available keys: {list(data.keys())}"
            )


def load_mean_field(npz_path: Union[str, Path]) -> np.ndarray:
    """Load final mean field from an NPZ file.

    Args:
        npz_path: Path to the final_mean_field.npz file.

    Returns:
        Array of mean field values (1D or 2D array).
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "mean_field" in data:
            return np.array(data["mean_field"])
        else:
            raise ValueError(
                f"NPZ file {npz_path} does not contain 'mean_field' key. "
                f"Available keys: {list(data.keys())}"
            )


def load_policy(npz_path: Union[str, Path]) -> np.ndarray:
    """Load final policy from an NPZ file.

    Args:
        npz_path: Path to the final_policy.npz file.

    Returns:
        Array of policy values (2D array: N_states x N_actions).
    """
    npz_path = Path(npz_path)
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    with np.load(npz_path, allow_pickle=True) as data:
        if "policy" in data:
            return np.array(data["policy"])
        else:
            raise ValueError(
                f"NPZ file {npz_path} does not contain 'policy' key. "
                f"Available keys: {list(data.keys())}"
            )


def plot_exploitability(
    exploitabilities: np.ndarray,
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    log_scale: bool = False,
    colors: Optional[ColorsConfig] = None,
) -> Optional[Figure]:
    """Plot a single exploitability vector.

    Args:
        exploitabilities: 1D array of exploitability values.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure.
        log_scale: If True, use log scale for y-axis.
        colors: Optional color configuration.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    exploitabilities = np.array(exploitabilities)
    if exploitabilities.ndim != 1:
        raise ValueError(
            f"exploitabilities must be 1D, got {exploitabilities.ndim}D array"
        )

    if colors is None:
        colors = ColorsConfig()

    iterations = np.arange(len(exploitabilities))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(colors.figure_background)

    ax.plot(iterations, exploitabilities, linewidth=2, color="blue", alpha=0.8)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if log_scale:
        ax.set_yscale("log")

    plt.tight_layout()

    if fn is not None:
        fn = Path(fn)
        fn.parent.mkdir(parents=True, exist_ok=True)
        if str(fn).lower().endswith(".pdf"):
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
        else:
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig

    plt.close(fig)
    return None


def plot_exploitability_from_npz(
    npz_path: Union[str, Path],
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    log_scale: bool = False,
    colors: Optional[ColorsConfig] = None,
) -> Optional[Figure]:
    """Plot exploitability loaded from an NPZ file.

    Saves the plot in the 'plots' subfolder of the directory containing the NPZ file.

    Args:
        npz_path: Path to the exploitabilities.npz file.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure. If None, saves as 'exploitability.pdf' in plots folder.
        log_scale: If True, use log scale for y-axis.
        colors: Optional color configuration.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    exploitabilities = load_exploitabilities(npz_path)
    npz_path = Path(npz_path)

    if fn is None:
        plots_dir = npz_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fn = plots_dir / "exploitability.pdf"

    fn_str = str(fn) if fn is not None else None

    return plot_exploitability(
        exploitabilities=exploitabilities,
        xlabel=xlabel,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn_str,
        log_scale=log_scale,
        colors=colors,
    )


def version_to_algorithm_dir(version_withhyper: str) -> str:
    """Extract algorithm directory name from version_withhyper string.

    Args:
        version_withhyper: Version name with hyperparameters (e.g., "pso_sweep_temp0p20", "damped_sweep_damped0p10").

    Returns:
        Algorithm directory name (e.g., "PSO", "DampedFP_damped", "OMD").
    """
    version_lower = version_withhyper.lower()

    if "pso_sweep" in version_lower:
        return "PSO"
    elif "omd_sweep" in version_lower:
        return "OMD"
    elif (
        "smooth_pi_sweep" in version_lower or "smooth_policy_iteration" in version_lower
    ):
        return "PI_smooth_policy_iteration"
    elif (
        "boltzmann_pi_sweep" in version_lower
        or "boltzmann_policy_iteration" in version_lower
    ):
        return "PI_boltzmann_policy_iteration"
    elif "policy_iteration_sweep" in version_lower:
        return "PI_policy_iteration"
    elif "fplay_sweep" in version_lower or "fictitious" in version_lower:
        return "DampedFP_fictitious_play"
    elif "pure_fp_sweep" in version_lower or version_lower.startswith("pure"):
        return "DampedFP_pure"
    elif "damped_sweep" in version_lower:
        return "DampedFP_damped"
    else:
        parts = version_withhyper.split("_")
        if len(parts) > 0:
            return parts[0].capitalize()
        return version_withhyper


def find_best_model_npz(
    environment: str,
    npz_filename: str,
    outputs_dir: Union[str, Path] = "outputs",
    seed: int = 42,
) -> Path:
    """Find the NPZ file for the best model from results/{environment}/best/best_model.yaml.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        npz_filename: Name of the NPZ file to find (e.g., "final_mean_field.npz", "final_policy.npz").
        outputs_dir: Root directory containing outputs. Defaults to "outputs".
        seed: Seed number to use (default: 42).

    Returns:
        Path to the NPZ file.

    Raises:
        FileNotFoundError: If YAML file or NPZ file not found.
    """
    project_root = Path(__file__).parent.parent
    yaml_path = project_root / "results" / environment / "best" / "best_model.yaml"

    if not yaml_path.exists():
        raise FileNotFoundError(
            f"Best model YAML not found: {yaml_path}. "
            f"Please run plot_exploitability_multiple_versions first to generate it."
        )

    with open(yaml_path) as f:
        yaml_data = yaml.safe_load(f)

    best_version = yaml_data.get("best_version")
    if not best_version:
        raise ValueError(f"No 'best_version' found in {yaml_path}")

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

    latest_timestamp_dir = max(timestamp_dirs, key=lambda d: d.stat().st_mtime)

    npz_path = latest_timestamp_dir / npz_filename
    if not npz_path.exists():
        raise FileNotFoundError(f"NPZ file not found: {npz_path}")

    return npz_path


def plot_mean_field_from_npz(
    npz_path: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None,
    is_grid: bool = False,
    grid_dim: Optional[tuple] = None,
    walls: Optional[np.ndarray] = None,
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    colors: Optional[ColorsConfig] = None,
    outputs_dir: Union[str, Path] = "outputs",
    seed: int = 42,
    background_color: Optional[str] = None,
    bar_color: Optional[str] = None,
    grid_color: Optional[str] = None,
    cmap_2d: Optional[str] = None,
) -> Optional[Union[Figure, str]]:
    """Plot final mean field loaded from an NPZ file.

    If environment is provided and npz_path is None, automatically finds the best model
    from results/{environment}/best/best_model.yaml and uses seed 42.

    Saves the plot in the 'plots' subfolder of the directory containing the NPZ file.

    Args:
        npz_path: Path to the final_mean_field.npz file. If None and environment is provided,
            automatically finds the best model.
        environment: Name of the environment (e.g., "LasryLionsChain"). Required if npz_path is None.
        is_grid: Whether the environment is a grid.
        grid_dim: Tuple (n_rows, n_cols) for grid environments.
        walls: Optional array for wall positions in grid environments.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure. If None, saves as 'mean_field.pdf' in plots folder.
        colors: Optional color configuration.
        outputs_dir: Root directory containing outputs. Defaults to "outputs". Only used if environment is provided.
        seed: Seed number to use when finding best model. Defaults to 42. Only used if environment is provided.
        background_color: Optional background color for the plot (e.g., "#D0D8E0"). If None, uses default.
        bar_color: Optional color for the bars (e.g., "#0F3E66"). If None, uses default.
        grid_color: Optional color for the grid lines (e.g., "gray", "#808080"). If None, uses default.
        cmap_2d: Optional colormap name for 2D grid plots (e.g., "viridis", "plasma"). If None, uses default.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    if npz_path is None:
        if environment is None:
            raise ValueError("Either npz_path or environment must be provided")
        npz_path = find_best_model_npz(
            environment, "final_mean_field.npz", outputs_dir, seed
        )
        print(f"Using best model from: {npz_path}")

    mean_field = load_mean_field(npz_path)
    npz_path = Path(npz_path)

    if fn is None:
        if environment is not None:
            project_root = Path(__file__).parent.parent
            env_dir = project_root / "results" / environment
            env_dir.mkdir(parents=True, exist_ok=True)
            fn = env_dir / "mean_field.pdf"
        else:
            plots_dir = npz_path.parent / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fn = plots_dir / "mean_field.pdf"

    fn_str = str(fn) if fn is not None else None

    return plot_mean_field(
        mean_field=mean_field,
        is_grid=is_grid,
        grid_dim=grid_dim,
        walls=walls,
        return_fig=return_fig,
        fn=fn_str,
        colors=colors,
        background_color=background_color,
        bar_color=bar_color,
        grid_color=grid_color,
        cmap_2d=cmap_2d,
    )


def plot_policy_from_npz(
    npz_path: Optional[Union[str, Path]] = None,
    environment: Optional[str] = None,
    is_grid: bool = False,
    grid_dim: Optional[tuple] = None,
    walls: Optional[np.ndarray] = None,
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    colors: Optional[ColorsConfig] = None,
    action_labels: Optional[List[str]] = None,
    action_cmaps: Optional[List[str]] = None,
    outputs_dir: Union[str, Path] = "outputs",
    seed: int = 42,
    cmap: Optional[str] = None,
    cmap_2d: Optional[str] = None,
    show_interval_in_labels: bool = True,
    tick_step: Optional[int] = None,
) -> Optional[Union[Figure, str]]:
    """Plot final policy loaded from an NPZ file.

    If environment is provided and npz_path is None, automatically finds the best model
    from results/{environment}/best/best_model.yaml and uses seed 42.

    Saves the plot in the 'plots' subfolder of the directory containing the NPZ file.

    Args:
        npz_path: Path to the final_policy.npz file. If None and environment is provided,
            automatically finds the best model.
        environment: Name of the environment (e.g., "LasryLionsChain"). Required if npz_path is None.
        is_grid: Whether the environment is a grid.
        grid_dim: Tuple (n_rows, n_cols) for grid environments.
        walls: Optional array for wall positions in grid environments.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure. If None, saves as 'policy.pdf' in plots folder.
        colors: Optional color configuration.
        action_labels: Optional list of action labels.
        action_cmaps: Optional list of colormap names for 2D plots.
        outputs_dir: Root directory containing outputs. Defaults to "outputs". Only used if environment is provided.
        seed: Seed number to use when finding best model. Defaults to 42. Only used if environment is provided.
        cmap: Optional colormap name for 1D policy plots (e.g., "viridis", "plasma"). If None, uses default.
        cmap_2d: Optional colormap name for 2D grid plots (e.g., "viridis", "plasma"). If None, uses default.
        show_interval_in_labels: If True, show probability intervals in legend labels. Default True.
        tick_step: Step size for axis ticks (e.g., 2 for 0, 2, 4, ...). If None, auto-detects for 11x11 grids.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    if npz_path is None:
        if environment is None:
            raise ValueError("Either npz_path or environment must be provided")
        npz_path = find_best_model_npz(
            environment, "final_policy.npz", outputs_dir, seed
        )
        print(f"Using best model from: {npz_path}")

    policy = load_policy(npz_path)
    npz_path = Path(npz_path)

    # Default save location
    if fn is None:
        if environment is not None:
            project_root = Path(__file__).parent.parent
            env_dir = project_root / "results" / environment
            env_dir.mkdir(parents=True, exist_ok=True)
            fn = env_dir / "policy.pdf"
        else:
            plots_dir = npz_path.parent / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            fn = plots_dir / "policy.pdf"

    fn_str = str(fn) if fn is not None else None

    return plot_policy(
        policy_array=policy,
        is_grid=is_grid,
        grid_dim=grid_dim,
        walls=walls,
        return_fig=return_fig,
        fn=fn_str,
        colors=colors,
        action_labels=action_labels,
        action_cmaps=action_cmaps,
        cmap=cmap,
        cmap_2d=cmap_2d,
        show_interval_in_labels=show_interval_in_labels,
        tick_step=tick_step,
    )


def plot_exploitability_mean_variance(
    exploitabilities_list: List[np.ndarray],
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    log_scale: bool = False,
    colors: Optional[ColorsConfig] = None,
    label: Optional[str] = None,
) -> Optional[Figure]:
    """Plot mean and variance of multiple exploitability vectors.

    Saves the plot in the 'final_results' folder at the project root.

    Args:
        exploitabilities_list: List of 1D arrays, each representing an exploitability trajectory.
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure. If None, saves as 'exploitability_mean_variance.pdf' in final_results folder.
        log_scale: If True, use log scale for y-axis.
        colors: Optional color configuration.
        label: Optional label for the curve.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    if len(exploitabilities_list) == 0:
        raise ValueError("exploitabilities_list cannot be empty")

    exploitabilities_list = [np.array(exp) for exp in exploitabilities_list]
    max_len = max(len(exp) for exp in exploitabilities_list)
    padded = np.full((len(exploitabilities_list), max_len), np.nan)
    for i, exp in enumerate(exploitabilities_list):
        padded[i, : len(exp)] = exp

    # Compute mean and std
    mean_exp = np.nanmean(padded, axis=0)
    std_exp = np.nanstd(padded, axis=0)

    if colors is None:
        colors = ColorsConfig()

    iterations = np.arange(len(mean_exp))
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(colors.figure_background)

    # Plot mean with shaded variance
    ax.plot(iterations, mean_exp, linewidth=2, color="blue", alpha=0.8, label=label)
    ax.fill_between(
        iterations,
        mean_exp - std_exp,
        mean_exp + std_exp,
        alpha=0.3,
        color="blue",
    )

    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    if log_scale:
        ax.set_yscale("log")

    if label is not None:
        ax.legend()

    plt.tight_layout()

    # Default save location: final_results folder at project root
    if fn is None:
        # Get project root (assuming we're in utility/ directory, go up one level)
        project_root = Path(__file__).parent.parent
        final_results_dir = project_root / "final_results"
        final_results_dir.mkdir(parents=True, exist_ok=True)
        fn = final_results_dir / "exploitability_mean_variance.pdf"

    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)
    if str(fn).lower().endswith(".pdf"):
        plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
    else:
        plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig

    plt.close(fig)
    return None


def plot_exploitability_groups(
    exploitabilities_groups: List[List[np.ndarray]],
    labels: Optional[List[str]] = None,
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    log_scale: bool = False,
    colors: Optional[ColorsConfig] = None,
    color_list: Optional[List[str]] = None,
    legend_loc: Optional[str] = None,
    show_legend: bool = True,
    plot_every_n: int = 1,
    marker: Optional[str] = None,
    marker_list: Optional[List[str]] = None,
    ax: Optional[Any] = None,
) -> Optional[Figure]:
    """Plot mean and variance for multiple groups of exploitability vectors.

    Each group is plotted as a separate curve with mean and variance shading.
    Saves the plot in the 'final_results' folder at the project root.

    Args:
        exploitabilities_groups: List of groups, where each group is a list of 1D exploitability arrays.
        labels: Optional list of labels for each group (must match number of groups).
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure. If None, saves as 'exploitability_groups.pdf' in final_results folder.
        log_scale: If True, use log scale for y-axis.
        colors: Optional color configuration.
        color_list: Optional list of colors for each group. If None, uses default colors.
        legend_loc: Optional legend location. If None, places legend below plot.
            Can be matplotlib legend location strings like "upper left", "upper right", etc.
        show_legend: If True, display the legend. If False, hide the legend.
        plot_every_n: Plot every Nth iteration (default: 1, plots all iterations).
        marker: Optional marker style (e.g., "o", ".", "s"). If None and plot_every_n > 1,
            uses "o" (dot). If None and plot_every_n == 1, no markers shown on plot.
        marker_list: Optional list of marker styles for each group (e.g., ["o", "s", "D", "^"]).
            If provided, each group gets a different marker. Takes precedence over marker parameter.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    if len(exploitabilities_groups) == 0:
        raise ValueError("exploitabilities_groups cannot be empty")

    if labels is not None and len(labels) != len(exploitabilities_groups):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of groups "
            f"({len(exploitabilities_groups)})"
        )

    if colors is None:
        colors = ColorsConfig()

    # Default colors if not provided
    if color_list is None:
        default_colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        color_list = [
            default_colors[i % len(default_colors)]
            for i in range(len(exploitabilities_groups))
        ]
    elif len(color_list) < len(exploitabilities_groups):
        # Extend color list if needed
        default_colors = [
            "blue",
            "red",
            "green",
            "orange",
            "purple",
            "brown",
            "pink",
            "gray",
            "olive",
            "cyan",
        ]
        color_list = list(color_list) + [
            default_colors[i % len(default_colors)]
            for i in range(len(color_list), len(exploitabilities_groups))
        ]

    # Adjust figure size based on number of groups
    num_groups = len(exploitabilities_groups)
    external_ax = ax is not None  # Track if ax was provided externally
    if ax is None:
        # Create new figure and axes if not provided
        if num_groups > 20:
            # Large figure for many groups, taller to accommodate legend below
            figsize = (18, 13)
        elif num_groups > 10:
            # Medium-large figure
            figsize = (18, 18)
        elif num_groups > 5:
            # Medium figure for moderate number of groups
            figsize = (13, 9)
        else:
            # Standard figure for few groups
            figsize = (12, 8)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor(colors.figure_background)
    else:
        # Use provided axes, get figure from it
        fig = ax.figure

    # Process each group
    for group_idx, group in enumerate(exploitabilities_groups):
        if len(group) == 0:
            continue

        # Convert to numpy arrays and pad to same length
        group_arrays = [np.array(exp) for exp in group]
        max_len = max(len(exp) for exp in group_arrays)
        padded = np.full((len(group_arrays), max_len), np.nan)
        for i, exp in enumerate(group_arrays):
            padded[i, : len(exp)] = exp

        # Compute mean and std
        mean_exp = np.nanmean(padded, axis=0)
        std_exp = np.nanstd(padded, axis=0)

        iterations = np.arange(len(mean_exp))
        label = labels[group_idx] if labels is not None else f"Group {group_idx + 1}"
        color = color_list[group_idx]

        # Subsample if plot_every_n > 1
        if plot_every_n > 1:
            indices = np.arange(0, len(iterations), plot_every_n)
            if len(indices) == 0 or indices[-1] != len(iterations) - 1:
                indices = np.append(indices, len(iterations) - 1)
            iterations_plot = iterations[indices]
            mean_exp_plot = mean_exp[indices]
            std_exp_plot = std_exp[indices]
            if marker_list is not None and group_idx < len(marker_list):
                plot_marker = marker_list[group_idx]
            elif marker is not None:
                plot_marker = marker
            else:
                plot_marker = "o"
            marker_size = 6 if plot_marker == "o" else 5
        else:
            iterations_plot = iterations
            mean_exp_plot = mean_exp
            std_exp_plot = std_exp
            if marker_list is not None and group_idx < len(marker_list):
                plot_marker = marker_list[group_idx]
            elif marker is not None:
                plot_marker = marker
            else:
                plot_marker = "o"
            marker_size = 0

        ax.plot(
            iterations_plot,
            mean_exp_plot,
            linewidth=2,
            color=color,
            alpha=0.8,
            label=label,
            marker=plot_marker,
            markersize=marker_size,
        )
        ax.fill_between(
            iterations_plot,
            mean_exp_plot - std_exp_plot,
            mean_exp_plot + std_exp_plot,
            alpha=0.3,
            color=color,
        )

    axis_label_size = 34 if num_groups > 10 else 22
    ax.set_xlabel(xlabel, fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    tick_labelsize = 26 if num_groups > 10 else 20
    ax.tick_params(axis="both", which="major", labelsize=tick_labelsize)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    max_iteration = 0
    for group in exploitabilities_groups:
        if len(group) > 0:
            group_arrays = [np.array(exp) for exp in group]
            max_len = max(len(exp) for exp in group_arrays)
            max_iteration = max(max_iteration, max_len - 1)

    if max_iteration == 149:
        max_iteration = 150

    x_ticks = np.arange(0, max_iteration + 1, 30)
    if len(x_ticks) == 0 or x_ticks[-1] != max_iteration:
        x_ticks = np.append(x_ticks, max_iteration)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(i)) for i in x_ticks])
    margin = max_iteration * 0.03
    ax.set_xlim(left=-margin, right=max_iteration + margin)

    if log_scale:
        ax.set_yscale("log")

    if legend_loc == "upper right" and show_legend:
        ymin, ymax = ax.get_ylim()
        if log_scale:
            ax.set_ylim(ymin, ymax * 5)
        else:
            y_range = ymax - ymin
            ax.set_ylim(ymin, ymax + 0.3 * y_range)

    if show_legend and (labels is not None or len(exploitabilities_groups) > 1):
        if legend_loc is not None:
            if legend_loc == "right":
                if num_groups > 20:
                    fontsize = 12
                elif num_groups > 10:
                    fontsize = 13
                elif num_groups > 5:
                    fontsize = 14
                else:
                    fontsize = 15
                ax.legend(
                    loc="center left",
                    bbox_to_anchor=(1.02, 0.5),
                    ncol=1,
                    fontsize=fontsize,
                    frameon=True,
                    handler_map={Line2D: HandlerCircle()},
                )
            else:
                if num_groups > 20:
                    ncol = min(6, (num_groups + 4) // 5)
                    fontsize = 14
                elif num_groups > 10:
                    ncol = min(5, (num_groups + 3) // 4)
                    fontsize = 15
                elif num_groups > 5:
                    ncol = min(3, (num_groups + 2) // 3)
                    fontsize = 16
                else:
                    ncol = 2
                    fontsize = 18
                ax.legend(
                    loc=legend_loc,
                    ncol=ncol,
                    fontsize=fontsize,
                    frameon=True,
                    handler_map={Line2D: HandlerCircle()},
                )
        else:
            ncol = 2
            nrows = (num_groups + ncol - 1) // ncol  # Ceiling division

            if nrows > 6:
                y_offset = -0.22
                fontsize = 24
            elif nrows > 4:
                y_offset = -0.19
                fontsize = 25
            elif nrows > 2:
                y_offset = -0.16
                fontsize = 26
            else:
                y_offset = -0.13
                fontsize = 27

            ax.legend(
                loc="upper center",
                bbox_to_anchor=(0.5, y_offset),
                ncol=ncol,
                fontsize=fontsize,
                frameon=True,
                handler_map={Line2D: HandlerCircle()},
                columnspacing=1.0,
            )

    if not show_legend:
        plt.tight_layout()
    elif legend_loc == "right":
        plt.tight_layout(rect=(0, 0, 0.75, 1))
    elif legend_loc is not None:
        plt.tight_layout()
    else:
        nrows = (num_groups + 1) // 2  # Ceiling division for 2 columns
        if nrows > 6:
            plt.tight_layout(rect=(0, 0.25, 1, 1))
        elif nrows > 4:
            plt.tight_layout(rect=(0, 0.22, 1, 1))
        elif nrows > 2:
            plt.tight_layout(rect=(0, 0.19, 1, 1))
        else:
            plt.tight_layout(rect=(0, 0.15, 1, 1))

    if external_ax:
        if return_fig:
            return fig
        return None

    if fn is None:
        project_root = Path(__file__).parent.parent
        final_results_dir = project_root / "final_results"
        final_results_dir.mkdir(parents=True, exist_ok=True)
        fn = final_results_dir / "exploitability_groups.pdf"

    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)
    if str(fn).lower().endswith(".pdf"):
        plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
    else:
        plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig

    plt.close(fig)
    return None


def group_exploitabilities_by_seed(
    environment: str,
    outputs_dir: Union[str, Path] = "outputs",
) -> Dict[str, Dict[str, Any]]:
    """Group exploitabilities by seed for each algorithm version with hyperparameters.

    Given an environment, scans the outputs directory and groups exploitabilities
    by algorithm_version_withhyper, then further groups by seed within each version.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        outputs_dir: Root directory containing outputs. Defaults to "outputs".

    Returns:
        Dictionary mapping version_withhyper to a dict with:
        - "groups": List of groups, where each group contains exploitability arrays for a specific seed
        - "seed_names": List of seed names corresponding to each group
        Structure: {
            "version_withhyper": {
                "groups": [
                    [exploitabilities from seed_10 run1, seed_10 run2, ...],  # group 0
                    [exploitabilities from seed_42 run1, seed_42 run2, ...],  # group 1
                ],
                "seed_names": ["seed_10", "seed_42"]
            }
        }

    Example:
        >>> result = group_exploitabilities_by_seed("LasryLionsChain")
        >>> version_data = result["damped_sweep_damped0p10"]
        >>> groups = version_data["groups"]  # List of seed groups
        >>> seed_names = version_data["seed_names"]  # ["seed_10", "seed_40"]
    """
    outputs_dir = Path(outputs_dir)
    env_dir = outputs_dir / environment

    if not env_dir.exists():
        raise FileNotFoundError(f"Environment directory not found: {env_dir}")

    version_groups: Dict[str, Dict[str, List[np.ndarray]]] = {}

    for algorithm_dir in env_dir.iterdir():
        if not algorithm_dir.is_dir():
            continue

        for seed_dir in algorithm_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue

            seed_name = seed_dir.name

            for version_dir in seed_dir.iterdir():
                if not version_dir.is_dir():
                    continue

                version_name = version_dir.name

                if version_name not in version_groups:
                    version_groups[version_name] = {}

                if seed_name not in version_groups[version_name]:
                    version_groups[version_name][seed_name] = []

                for timestamp_dir in version_dir.iterdir():
                    if not timestamp_dir.is_dir():
                        continue

                    exploitabilities_path = timestamp_dir / "exploitabilities.npz"
                    if exploitabilities_path.exists():
                        try:
                            exploitabilities = load_exploitabilities(
                                exploitabilities_path
                            )
                            version_groups[version_name][seed_name].append(
                                exploitabilities
                            )
                        except Exception as e:
                            print(
                                f"Warning: Failed to load {exploitabilities_path}: {e}"
                            )
                            continue

    result: Dict[str, Dict[str, Any]] = {}
    for version_name, seed_dict in version_groups.items():
        sorted_seeds = sorted(seed_dict.keys())
        groups: List[List[np.ndarray]] = [seed_dict[seed] for seed in sorted_seeds]
        result[version_name] = {
            "groups": groups,
            "seed_names": sorted_seeds,
        }

    return result


def plot_exploitability_by_version_and_seed(
    environment: str,
    version_withhyper: Optional[str] = None,
    outputs_dir: Union[str, Path] = "outputs",
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    log_scale: bool = False,
    colors: Optional[ColorsConfig] = None,
    color_list: Optional[List[str]] = None,
) -> Optional[Figure]:
    """Plot exploitabilities grouped by seed for a specific algorithm version.

    Given an environment and version_withhyper, loads all exploitabilities
    grouped by seed and plots them using plot_exploitability_groups.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        version_withhyper: Specific version to plot. If None, raises error.
        outputs_dir: Root directory containing outputs. Defaults to "outputs".
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure.
        log_scale: If True, use log scale for y-axis.
        colors: Optional color configuration.
        color_list: Optional list of colors for each seed group.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    if version_withhyper is None:
        raise ValueError("version_withhyper must be specified")

    all_groups = group_exploitabilities_by_seed(environment, outputs_dir)

    if version_withhyper not in all_groups:
        available = list(all_groups.keys())
        raise ValueError(
            f"Version '{version_withhyper}' not found. Available versions: {available}"
        )

    version_data = all_groups[version_withhyper]
    seed_groups: List[List[np.ndarray]] = version_data["groups"]

    if len(seed_groups) == 0:
        raise ValueError(
            f"No exploitability data found for version '{version_withhyper}'"
        )

    all_exploitabilities: List[np.ndarray] = []
    for seed_group in seed_groups:
        all_exploitabilities.extend(seed_group)

    if len(all_exploitabilities) == 0:
        raise ValueError(
            f"No exploitability data found for version '{version_withhyper}'"
        )

    if fn is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_version = version_withhyper.replace("/", "_").replace("\\", "_")
        fn = results_dir / f"{safe_version}_by_seed.pdf"

    return plot_exploitability_mean_variance(
        exploitabilities_list=all_exploitabilities,
        xlabel=xlabel,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn,
        log_scale=log_scale,
        colors=colors,
        label=None,
    )


def plot_exploitability_multiple_versions(
    environment: str,
    versions_withhyper: List[str],
    outputs_dir: Union[str, Path] = "outputs",
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    log_scale: bool = False,
    colors: Optional[ColorsConfig] = None,
    color_list: Optional[List[str]] = None,
    cmap: Optional[Union[str, Any]] = None,
    legend_loc: Optional[str] = None,
    show_legend: bool = True,
    label_format: str = "algorithm",
    best_version: Optional[str] = None,
    plot_every_n: int = 1,
    marker: Optional[str] = None,
    marker_list: Optional[List[str]] = None,
) -> Optional[Figure]:
    """Plot exploitabilities for multiple algorithm versions on the same plot.

    Each version is plotted as a separate curve with mean and variance shading,
    combining all seeds for each version.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        versions_withhyper: List of version_withhyper names to plot.
        outputs_dir: Root directory containing outputs. Defaults to "outputs".
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure.
        log_scale: If True, use log scale for y-axis.
        colors: Optional color configuration.
        color_list: Optional list of colors for each version. If None, uses default colors.
        cmap: Optional colormap name or colormap object. If provided and color_list is None,
            colors will be generated from the colormap. Takes precedence over default colors.
        legend_loc: Optional legend location. If None, places legend below plot.
            Can be matplotlib legend location strings like "upper left", "upper right", etc.
        show_legend: If True, display the legend. If False, hide the legend.
        label_format: Format for labels. "algorithm" for short algorithm names,
            "hyperparameters" for hyperparameter strings, "full" for full version names.
        best_version: Optional version_withhyper name that achieved the best final mean.
            If provided, its label will be bolded in the legend.
        plot_every_n: Plot every Nth iteration (default: 1, plots all iterations).
        marker: Optional marker style (e.g., "o", ".", "s"). If None and plot_every_n > 1,
            uses "o" (dot). If None and plot_every_n == 1, no markers shown on plot.
        marker_list: Optional list of marker styles for each version (e.g., ["o", "s", "D", "^"]).
            If provided, each version gets a different marker. Takes precedence over marker parameter.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.
    """
    if len(versions_withhyper) == 0:
        raise ValueError("versions_withhyper cannot be empty")

    all_groups = group_exploitabilities_by_seed(environment, outputs_dir)

    final_means: Dict[str, float] = {}

    exploitabilities_groups: List[List[np.ndarray]] = []
    labels: List[str] = []

    for version_withhyper in versions_withhyper:
        if version_withhyper not in all_groups:
            available = list(all_groups.keys())
            raise ValueError(
                f"Version '{version_withhyper}' not found. Available versions: {available}"
            )

        version_data = all_groups[version_withhyper]
        seed_groups: List[List[np.ndarray]] = version_data["groups"]

        if len(seed_groups) == 0:
            print(
                f"Warning: No exploitability data found for version '{version_withhyper}', skipping."
            )
            continue

        combined_exploitabilities: List[np.ndarray] = []
        for seed_group in seed_groups:
            combined_exploitabilities.extend(seed_group)

        if len(combined_exploitabilities) > 0:
            exp_arrays = [np.array(exp) for exp in combined_exploitabilities]
            max_len = max(len(exp) for exp in exp_arrays)
            padded = np.full((len(exp_arrays), max_len), np.nan)
            for i, exp in enumerate(exp_arrays):
                padded[i, : len(exp)] = exp

            mean_exp = np.nanmean(padded, axis=0)
            final_mean = mean_exp[-1]
            final_means[version_withhyper] = final_mean

            exploitabilities_groups.append(combined_exploitabilities)
            if label_format == "algorithm":
                label = version_to_algorithm_name(version_withhyper)
            elif label_format == "hyperparameters":
                hyperparams = extract_hyperparameters(version_withhyper)
                label = hyperparams if hyperparams else version_withhyper
            else:  # "full" or default
                label = version_withhyper

            if best_version is not None and version_withhyper == best_version:
                if "$" in label:
                    parts = label.split("$")
                    math_parts = []
                    for i, part in enumerate(parts):
                        if i % 2 == 1:
                            math_parts.append(f"\\mathbf{{{part}}}")
                        else:
                            if part.strip():
                                part_with_spaces = part.replace(" ", "\\ ")
                                math_parts.append(f"\\mathbf{{{part_with_spaces}}}")
                    label = f"${''.join(math_parts)}$"
                else:
                    label = f"$\\mathbf{{{label}}}$"

            labels.append(label)

    if len(exploitabilities_groups) == 0:
        raise ValueError("No valid exploitability data found for any version")

    best_version_name = None
    best_final_mean = None
    if final_means:
        sorted_versions = sorted(final_means.items(), key=lambda x: x[1])
        best_version_name = sorted_versions[0][0]
        best_final_mean = sorted_versions[0][1]
        print(
            f"\nBest model: {best_version_name} with final mean exploitability: {best_final_mean:.6f}"
        )

        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment / "best"
        results_dir.mkdir(parents=True, exist_ok=True)
        yaml_path = results_dir / "best_model.yaml"

        yaml_data = {
            "environment": environment,
            "best_version": best_version_name,
            "final_mean_exploitability": float(best_final_mean),
            "all_versions": [
                {
                    "version": version,
                    "final_mean_exploitability": float(exploitability),
                }
                for version, exploitability in sorted_versions
            ],
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        print(f"Best model saved to: {yaml_path}")

    if cmap is not None and color_list is None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        colormap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        num_versions = len(exploitabilities_groups)
        color_list = [
            mcolors.to_hex(colormap(i / (num_versions - 1) if num_versions > 1 else 0))
            for i in range(num_versions)
        ]

    if fn is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_versions = "_".join(
            [v.replace("/", "_").replace("\\", "_") for v in versions_withhyper[:3]]
        )
        if len(versions_withhyper) > 3:
            safe_versions += f"_and_{len(versions_withhyper) - 3}_more"
        fn = results_dir / f"multiple_versions_{safe_versions}.pdf"

    return plot_exploitability_groups(
        exploitabilities_groups=exploitabilities_groups,
        labels=labels,
        xlabel=xlabel,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn,
        log_scale=log_scale,
        colors=colors,
        color_list=color_list,
        legend_loc=legend_loc,
        show_legend=show_legend,
        plot_every_n=plot_every_n,
        marker=marker,
        marker_list=marker_list,
    )


def get_versions_for_algorithm(
    environment: str,
    algorithm: str,
    outputs_dir: Union[str, Path] = "outputs",
) -> List[str]:
    """Get all versions_withhyper for a specific algorithm and environment.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        algorithm: Name of the algorithm (e.g., "PSO", "DampedFP_damped").
        outputs_dir: Root directory containing outputs. Defaults to "outputs".

    Returns:
        List of unique version_withhyper names, sorted alphabetically.
    """
    outputs_dir = Path(outputs_dir)
    algorithm_dir = outputs_dir / environment / algorithm

    if not algorithm_dir.exists():
        raise FileNotFoundError(f"Algorithm directory not found: {algorithm_dir}")

    versions_set = set()

    # Scan directory structure: outputs/{environment}/{algorithm}/{seed}/{version_withhyper}/
    for seed_dir in algorithm_dir.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue

        for version_dir in seed_dir.iterdir():
            if not version_dir.is_dir():
                continue

            version_name = version_dir.name
            # Check if this version directory contains exploitabilities.npz files
            has_exploitabilities = any(
                (timestamp_dir / "exploitabilities.npz").exists()
                for timestamp_dir in version_dir.iterdir()
                if timestamp_dir.is_dir()
            )

            if has_exploitabilities:
                versions_set.add(version_name)

    return sorted(versions_set)


def plot_exploitability_by_algorithm(
    environment: str,
    algorithm: str,
    outputs_dir: Union[str, Path] = "outputs",
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: Optional[Union[str, Path]] = None,
    log_scale: bool = False,
    colors: Optional[ColorsConfig] = None,
    color_list: Optional[List[str]] = None,
    legend_loc: Optional[str] = None,
    show_legend: bool = True,
    plot_every_n: int = 1,
    marker: Optional[str] = None,
) -> Optional[Figure]:
    """Plot all versions_withhyper for a specific algorithm and environment.

    For each version, combines all seeds and plots mean ± std. All versions are
    plotted on the same figure.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        algorithm: Name of the algorithm (e.g., "PSO", "DampedFP_damped").
        outputs_dir: Root directory containing outputs. Defaults to "outputs".
        xlabel: Label for x-axis.
        ylabel: Label for y-axis.
        return_fig: If True, returns the figure object.
        fn: Optional filename to save the figure.
        log_scale: If True, use log scale for y-axis.
        colors: Optional color configuration.
        color_list: Optional list of colors for each version. If None, uses default colors.
        legend_loc: Optional legend location. If None, places legend below plot.
            Can be matplotlib legend location strings like "upper left", "upper right", etc.
        show_legend: If True, display the legend. If False, hide the legend.
        plot_every_n: Plot every Nth iteration (default: 1, plots all iterations).
        marker: Optional marker style (e.g., "o", ".", "s"). If None and plot_every_n > 1,
            uses "o" (dot). If None and plot_every_n == 1, no markers shown on plot.

    Returns:
        The matplotlib Figure if return_fig is True; otherwise None.

    Example:
        >>> plot_exploitability_by_algorithm(
        ...     environment="LasryLionsChain",
        ...     algorithm="PSO",
        ...     log_scale=True,
        ... )
        # Plots all 36 versions for PSO with mean ± std computed across seeds
    """
    # Get all versions for this algorithm
    versions_withhyper = get_versions_for_algorithm(environment, algorithm, outputs_dir)

    if len(versions_withhyper) == 0:
        raise ValueError(
            f"No versions found for algorithm '{algorithm}' in environment '{environment}'"
        )

    print(
        f"Found {len(versions_withhyper)} versions for {algorithm}: {versions_withhyper}"
    )

    # Get grouped exploitabilities to compute final means
    all_groups = group_exploitabilities_by_seed(environment, outputs_dir)

    # Compute final mean exploitability for each version
    final_means = {}
    for version_withhyper in versions_withhyper:
        if version_withhyper not in all_groups:
            continue

        version_data = all_groups[version_withhyper]
        seed_groups: List[List[np.ndarray]] = version_data["groups"]

        if len(seed_groups) == 0:
            continue

        combined_exploitabilities: List[np.ndarray] = []
        for seed_group in seed_groups:
            combined_exploitabilities.extend(seed_group)

        if len(combined_exploitabilities) == 0:
            continue

        group_arrays = [np.array(exp) for exp in combined_exploitabilities]
        max_len = max(len(exp) for exp in group_arrays)
        padded = np.full((len(group_arrays), max_len), np.nan)
        for i, exp in enumerate(group_arrays):
            padded[i, : len(exp)] = exp

        mean_exp = np.nanmean(padded, axis=0)
        final_mean = mean_exp[-1]
        final_means[version_withhyper] = final_mean

    if final_means:
        sorted_versions = sorted(final_means.items(), key=lambda x: x[1])
        best_version = sorted_versions[0][0]
        best_final_mean = sorted_versions[0][1]
        print(
            f"\nBest final mean exploitability: {best_version} = {best_final_mean:.6f}"
        )

        print(f"\n{'Rank':<6} {'Version':<50} {'Final Exploitability':<20}")
        print("-" * 76)
        for rank, (version, exploitability) in enumerate(sorted_versions, 1):
            print(f"{rank:<6} {version:<50} {exploitability:.6f}")

        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_algorithm = algorithm.replace("/", "_").replace("\\", "_")
        yaml_path = results_dir / f"{safe_algorithm}_best_models.yaml"

        yaml_data = {
            "algorithm": algorithm,
            "environment": environment,
            "best_models": [
                {
                    "rank": rank,
                    "version": version,
                    "final_exploitability": float(exploitability),
                }
                for rank, (version, exploitability) in enumerate(sorted_versions, 1)
            ],
        }

        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

        print(f"\nBest models saved to: {yaml_path}")

    if fn is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_algorithm = algorithm.replace("/", "_").replace("\\", "_")
        fn = results_dir / f"{safe_algorithm}_all_versions.pdf"

    best_version_for_legend = None
    if final_means:
        sorted_versions = sorted(final_means.items(), key=lambda x: x[1])
        best_version_for_legend = sorted_versions[0][0]

    max_versions_per_plot = 20
    if len(versions_withhyper) > max_versions_per_plot:
        mid_point = len(versions_withhyper) // 2
        versions_part1 = versions_withhyper[:mid_point]
        versions_part2 = versions_withhyper[mid_point:]

        best_version_part1 = (
            best_version_for_legend
            if best_version_for_legend in versions_part1
            else None
        )
        best_version_part2 = (
            best_version_for_legend
            if best_version_for_legend in versions_part2
            else None
        )

        print(
            f"\nSplitting {len(versions_withhyper)} versions into two figures: "
            f"Part 1 ({len(versions_part1)} versions), Part 2 ({len(versions_part2)} versions)"
        )

        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        safe_algorithm = algorithm.replace("/", "_").replace("\\", "_")

        if fn is None:
            fn_part1 = results_dir / f"{safe_algorithm}_all_versions_part1.pdf"
            fn_part2 = results_dir / f"{safe_algorithm}_all_versions_part2.pdf"
        else:
            fn = Path(fn)
            fn_part1 = fn.parent / f"{fn.stem}_part1{fn.suffix}"
            fn_part2 = fn.parent / f"{fn.stem}_part2{fn.suffix}"

        plot_exploitability_multiple_versions(
            environment=environment,
            versions_withhyper=versions_part1,
            outputs_dir=outputs_dir,
            xlabel=xlabel,
            ylabel=ylabel,
            return_fig=False,
            fn=fn_part1,
            log_scale=log_scale,
            colors=colors,
            color_list=color_list,
            legend_loc=legend_loc,
            show_legend=show_legend,
            label_format="hyperparameters",
            best_version=best_version_part1,
            plot_every_n=plot_every_n,
            marker=marker,
        )
        print(f"Saved part 1 to: {fn_part1}")

        plot_exploitability_multiple_versions(
            environment=environment,
            versions_withhyper=versions_part2,
            outputs_dir=outputs_dir,
            xlabel=xlabel,
            ylabel=ylabel,
            return_fig=return_fig,
            fn=fn_part2,
            log_scale=log_scale,
            colors=colors,
            color_list=color_list,
            legend_loc=legend_loc,
            show_legend=show_legend,
            label_format="hyperparameters",
            best_version=best_version_part2,
            plot_every_n=plot_every_n,
            marker=marker,
        )
        print(f"Saved part 2 to: {fn_part2}")

        return None
    else:
        return plot_exploitability_multiple_versions(
            environment=environment,
            versions_withhyper=versions_withhyper,
            outputs_dir=outputs_dir,
            xlabel=xlabel,
            ylabel=ylabel,
            return_fig=return_fig,
            fn=fn,
            log_scale=log_scale,
            colors=colors,
            color_list=color_list,
            legend_loc=legend_loc,
            show_legend=show_legend,
            label_format="hyperparameters",
            best_version=best_version_for_legend,
            plot_every_n=plot_every_n,
            marker=marker,
        )


def version_to_algorithm_name(version_withhyper: str) -> str:
    """Convert version_withhyper name to short algorithm name.

    Args:
        version_withhyper: Version name with hyperparameters (e.g., "damped_sweep_damped0p10").

    Returns:
        Short algorithm name (e.g., "Damped FP").
    """
    version_lower = version_withhyper.lower()

    if "pure_fp" in version_lower or version_lower.startswith("pure"):
        return "Fixed Point (FP)"
    elif "fplay" in version_lower or "fictitious" in version_lower:
        return "Fictitious Play"
    elif (
        "smooth_policy_iteration" in version_lower or "smooth_pi_sweep" in version_lower
    ):
        return "Smooth PI"
    elif (
        "boltzmann_policy_iteration" in version_lower
        or "boltzmann_pi_sweep" in version_lower
    ):
        return "Boltzmann PI"
    elif "omd_sweep" in version_lower:
        return "OMD"
    elif "pso_sweep" in version_lower:
        return "PSO"
    elif "damped_sweep_damped" in version_lower:
        return "Damped FP"
    elif "policy_iteration_sweep" in version_lower:
        return "PI"
    else:
        return version_withhyper


def extract_hyperparameters(version_withhyper: str) -> str:
    """Extract hyperparameters from version name in compact format with LaTeX notation.

    Args:
        version_withhyper: Version name with hyperparameters (e.g., "pso_sweep_temp0p20_w0p30_c10p30_c21p20").

    Returns:
        Compact hyperparameter string with LaTeX notation, grouping names together then values.
        Example: "$\\tau$, lr = 0.2, 0.0050" or "$\\tau, \\lambda$, lr = 0.2, 0.5, 0.0050".
    """
    parts = version_withhyper.split("_")
    temp_value = None
    damped_value = None
    lr_value = None
    w_value = None
    c1_value = None
    c2_value = None

    for part in parts:
        if part.startswith("damped") and "damped" in part:
            value_str = part.replace("damped", "").replace("p", ".")
            with contextlib.suppress(ValueError):
                damped_value = float(value_str)
        elif part.startswith("temp"):
            value_str = part.replace("temp", "").replace("p", ".")
            with contextlib.suppress(ValueError):
                temp_value = float(value_str)
        elif part.startswith("lr"):
            value_str = part.replace("lr", "").replace("p", ".")
            with contextlib.suppress(ValueError):
                lr_value = float(value_str)
        elif part.startswith("w") and len(part) > 1:
            value_str = part.replace("w", "").replace("p", ".")
            with contextlib.suppress(ValueError):
                w_value = float(value_str)
        elif part.startswith("c1"):
            value_str = part.replace("c1", "").replace("p", ".")
            with contextlib.suppress(ValueError):
                c1_value = float(value_str)
        elif part.startswith("c2"):
            value_str = part.replace("c2", "").replace("p", ".")
            with contextlib.suppress(ValueError):
                c2_value = float(value_str)

    param_names = []
    param_values = []

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
        names_str = ", ".join(param_names)
        values_str = ", ".join(param_values)
        return f"{names_str} = {values_str}"
    else:
        return ""


def get_four_rooms_walls(grid_dim: tuple) -> np.ndarray:
    """Generate walls array for FourRoomsAversion2D environment.

    Args:
        grid_dim: Tuple (n_rows, n_cols) specifying the grid dimensions.

    Returns:
        Array of shape (n_rows * n_cols,) where 0 = wall, 1 = free.
    """
    n_rows, n_cols = grid_dim
    N_flat = n_rows * n_cols
    walls = np.ones(N_flat, dtype=int)

    mid_row, mid_col = n_rows // 2, n_cols // 2  # 5,5 for 11x11

    doors = {(2, 5), (8, 5), (5, 8), (5, 2)}

    for row in range(n_rows):
        if (row, mid_col) not in doors:
            state_idx = row * n_cols + mid_col
            if 0 <= state_idx < N_flat:
                walls[state_idx] = 0

    for col in range(n_cols):
        if (mid_row, col) not in doors:
            state_idx = mid_row * n_cols + col
            if 0 <= state_idx < N_flat:
                walls[state_idx] = 0

    return walls


def get_versions_for_comparison(
    environment: str,
    fixed_versions: Optional[List[str]] = None,
    results_dir: Optional[Union[str, Path]] = None,
) -> List[str]:
    """Get versions for comparison by combining fixed versions with best models from YAML files.

    Args:
        environment: Name of the environment (e.g., "LasryLionsChain").
        fixed_versions: List of fixed versions to always include. If None, uses default fixed versions.
        results_dir: Root directory containing results. If None, uses project_root/results.

    Returns:
        List of version names combining fixed versions and best models (rank 1) from YAML files.
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

    best_versions_from_yaml = []
    if results_dir.exists():
        for yaml_file in results_dir.glob("*_best_models.yaml"):
            try:
                with open(yaml_file) as f:
                    yaml_data = yaml.safe_load(f)
                    if (
                        yaml_data
                        and "best_models" in yaml_data
                        and len(yaml_data["best_models"]) > 0
                    ):
                        rank_1_model = yaml_data["best_models"][0]
                        if rank_1_model.get("rank") == 1:
                            best_versions_from_yaml.append(rank_1_model["version"])
            except Exception as e:
                print(f"Warning: Failed to load {yaml_file}: {e}")
                continue

    versions_withhyper = list(dict.fromkeys(fixed_versions + best_versions_from_yaml))

    return versions_withhyper


ALGORITHMS = [
    "PSO",
    "OMD",
    "PI_smooth_policy_iteration",
    "PI_boltzmann_policy_iteration",
    "DampedFP_damped",
]


if __name__ == "__main__":
    environment = "MultipleEquilibriaGame"
    for algorithm in ALGORITHMS:
        legend_location = None if algorithm == "PSO" else "upper right"
        show_legend = True
        plot_exploitability_by_algorithm(
            environment=environment,
            algorithm=algorithm,
            outputs_dir="outputs",
            log_scale=True,
            legend_loc=legend_location,
            show_legend=show_legend,
        )

    versions_withhyper = get_versions_for_comparison(environment=environment)

    print(f"Using {len(versions_withhyper)} versions:")
    print(f"Versions: {versions_withhyper}")

    plot_exploitability_multiple_versions(
        environment=environment,
        versions_withhyper=versions_withhyper,
        outputs_dir="outputs",
        log_scale=True,
        color_list=[
            "#F86262",
            "#F0816A",
            "#7F11F5",
            "#0936C8",
            "#63B0F8",
            "#FA8FBF",
            "#703F62",
            "#97B9C3",
        ],
        legend_loc="upper right",
        plot_every_n=3,
        marker_list=[
            "o",
            "s",
            "D",
            "^",
            "v",
            "p",
            "h",
            "*",
        ],
    )
    # walls = get_four_rooms_walls(grid_dim=(11, 11))

    plot_mean_field_from_npz(
        environment=environment,
        is_grid=False,
        # grid_dim=(11, 11),
        # walls=walls,
        outputs_dir="outputs",
        seed=42,
        background_color="white",
        grid_color="gray",
        # cmap_2d="RdPu_r",
    )
    plot_policy_from_npz(
        environment=environment,
        is_grid=False,
        # grid_dim=(11, 11),
        # walls=walls,  # activating for 2D environments
        outputs_dir="outputs",
        seed=42,
        cmap="berlin",  # activating for 1D environments
    )
