"""Core matplotlib plot functions and high-level orchestrators shared across scripts."""

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

from conf.visualization.visualization_schema import ColorsConfig
from matplotlib.figure import Figure
from matplotlib.legend_handler import HandlerLine2D
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, NullLocator
import numpy as np
from utility.MFGPlots import plot_mean_field, plot_policy
from utility.plot_discovery import (
    extract_hyperparameters,
    group_exploitabilities_by_seed,
    group_runtimes_by_seed,
    version_to_algorithm_name,
)
from utility.plot_loaders import (
    find_best_model_npz,
    load_exploitabilities,
    load_mean_field,
    load_policy,
)
import yaml

_DARK_TEAL = "#006D6D"


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


def _save_fig(fig: Figure, fn: str | Path | None) -> None:
    """Save figure to PDF or PNG depending on extension."""
    if fn is None:
        return
    fn = Path(fn)
    fn.parent.mkdir(parents=True, exist_ok=True)
    if str(fn).lower().endswith(".pdf"):
        plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
    else:
        plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)


def _format_log_y_axis(ax) -> None:
    """Reduce log-scale y-axis clutter for exploitability plots."""
    ax.set_yscale("log")
    ax.yaxis.set_major_locator(LogLocator(base=10.0, numticks=3))
    ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10.0))
    ax.yaxis.set_minor_locator(NullLocator())
    ax.tick_params(axis="y", which="minor", left=False, right=False)


# ---------------------------------------------------------------------------
# Single-run plot functions
# ---------------------------------------------------------------------------


def plot_exploitability(
    exploitabilities: np.ndarray,
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: str | Path | None = None,
    log_scale: bool = False,
    colors: ColorsConfig | None = None,
    plot_every_n: int = 10,
    marker: str = "o",
) -> Figure | None:
    """Plot a single exploitability trajectory."""
    exploitabilities = np.array(exploitabilities)
    if exploitabilities.ndim != 1:
        raise ValueError(
            f"exploitabilities must be 1D, got {exploitabilities.ndim}D array"
        )

    if colors is None:
        colors = ColorsConfig()

    iterations = np.arange(len(exploitabilities))

    if plot_every_n > 1:
        indices = np.arange(0, len(iterations), plot_every_n)
        if len(indices) == 0 or indices[-1] != len(iterations) - 1:
            indices = np.append(indices, len(iterations) - 1)
        iterations_plot = iterations[indices]
        exp_plot = exploitabilities[indices]
        marker_size = 5
    else:
        iterations_plot = iterations
        exp_plot = exploitabilities
        marker_size = 0

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.patch.set_facecolor(colors.figure_background)

    ax.plot(iterations_plot, exp_plot, linewidth=2, color=_DARK_TEAL, zorder=2)
    if marker_size > 0:
        ax.scatter(
            iterations_plot,
            exp_plot,
            s=marker_size**2,
            color=_DARK_TEAL,
            alpha=0.5,
            zorder=3,
        )
    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    max_iter = len(iterations) - 1
    step = 50 if max_iter >= 100 else (25 if max_iter >= 50 else 10)
    x_ticks = np.arange(0, max_iter + 1, step)
    if len(x_ticks) == 0 or x_ticks[-1] != max_iter:
        x_ticks = np.append(x_ticks, max_iter)
    ax.set_xticks(x_ticks)

    if log_scale:
        _format_log_y_axis(ax)

    plt.tight_layout()
    _save_fig(fig, fn)

    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_exploitability_from_npz(
    npz_path: str | Path,
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: str | Path | None = None,
    log_scale: bool = False,
    colors: ColorsConfig | None = None,
    plot_every_n: int = 10,
    marker: str = "o",
) -> Figure | None:
    """Load exploitabilities from NPZ and plot them.

    Saves to <npz_dir>/plots/exploitability.pdf by default.
    """
    exploitabilities = load_exploitabilities(npz_path)
    npz_path = Path(npz_path)

    if fn is None:
        plots_dir = npz_path.parent / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)
        fn = plots_dir / "exploitability.pdf"

    return plot_exploitability(
        exploitabilities=exploitabilities,
        xlabel=xlabel,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn,
        log_scale=log_scale,
        colors=colors,
        plot_every_n=plot_every_n,
        marker=marker,
    )


def plot_mean_field_from_npz(
    npz_path: str | Path | None = None,
    environment: str | None = None,
    is_grid: bool = False,
    grid_dim: tuple | None = None,
    walls: np.ndarray | None = None,
    return_fig: bool = False,
    fn: str | Path | None = None,
    colors: ColorsConfig | None = None,
    outputs_dir: str | Path = "outputs",
    seed: int = 42,
    background_color: str | None = None,
    bar_color: str | None = None,
    grid_color: str | None = None,
    cmap_2d: str | None = None,
) -> Figure | None:
    """Load mean field from NPZ and plot it.

    If environment is given and npz_path is None, finds the best model from
    results/{environment}/best/best_model.yaml.
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

    return plot_mean_field(
        mean_field=mean_field,
        is_grid=is_grid,
        grid_dim=grid_dim,
        walls=walls,
        return_fig=return_fig,
        fn=str(fn),
        colors=colors,
        background_color=background_color,
        bar_color=bar_color,
        grid_color=grid_color,
        cmap_2d=cmap_2d,
    )


def plot_policy_from_npz(
    npz_path: str | Path | None = None,
    environment: str | None = None,
    is_grid: bool = False,
    grid_dim: tuple | None = None,
    walls: np.ndarray | None = None,
    return_fig: bool = False,
    fn: str | Path | None = None,
    colors: ColorsConfig | None = None,
    action_labels: list[str] | None = None,
    action_cmaps: list[str] | None = None,
    outputs_dir: str | Path = "outputs",
    seed: int = 42,
    cmap: str | None = None,
    cmap_2d: str | None = None,
    show_interval_in_labels: bool = True,
    tick_step: int | None = None,
) -> Figure | None:
    """Load policy from NPZ and plot it.

    If environment is given and npz_path is None, finds the best model from
    results/{environment}/best/best_model.yaml.
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

    return plot_policy(
        policy_array=policy,
        is_grid=is_grid,
        grid_dim=grid_dim,
        walls=walls,
        return_fig=return_fig,
        fn=str(fn),
        colors=colors,
        action_labels=action_labels,
        action_cmaps=action_cmaps,
        cmap=cmap,
        cmap_2d=cmap_2d,
        show_interval_in_labels=show_interval_in_labels,
        tick_step=tick_step,
    )


# ---------------------------------------------------------------------------
# Multi-run / multi-version plot functions
# ---------------------------------------------------------------------------


def plot_exploitability_mean_variance(
    exploitabilities_list: list[np.ndarray],
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: str | Path | None = None,
    log_scale: bool = False,
    colors: ColorsConfig | None = None,
    label: str | None = None,
    plot_every_n: int = 10,
    marker: str = "o",
) -> Figure | None:
    """Plot mean ± std over multiple exploitability trajectories."""
    if len(exploitabilities_list) == 0:
        raise ValueError("exploitabilities_list cannot be empty")

    exploitabilities_list = [np.array(exp) for exp in exploitabilities_list]
    max_len = max(len(exp) for exp in exploitabilities_list)
    padded = np.full((len(exploitabilities_list), max_len), np.nan)
    for i, exp in enumerate(exploitabilities_list):
        padded[i, : len(exp)] = exp

    mean_exp = np.nanmean(padded, axis=0)
    std_exp = np.nanstd(padded, axis=0)

    if colors is None:
        colors = ColorsConfig()

    iterations = np.arange(len(mean_exp))

    if plot_every_n > 1:
        indices = np.arange(0, len(iterations), plot_every_n)
        if len(indices) == 0 or indices[-1] != len(iterations) - 1:
            indices = np.append(indices, len(iterations) - 1)
        iterations_plot = iterations[indices]
        mean_plot = mean_exp[indices]
        std_plot = std_exp[indices]
        marker_size = 5
    else:
        iterations_plot = iterations
        mean_plot = mean_exp
        std_plot = std_exp
        marker_size = 0

    fig, ax = plt.subplots(1, 1, figsize=(5, 4))
    fig.patch.set_facecolor(colors.figure_background)

    ax.plot(
        iterations_plot, mean_plot, linewidth=2, color=_DARK_TEAL, label=label, zorder=2
    )
    if marker_size > 0:
        ax.scatter(
            iterations_plot,
            mean_plot,
            s=marker_size**2,
            color=_DARK_TEAL,
            alpha=0.5,
            zorder=3,
        )
    ax.fill_between(
        iterations_plot,
        mean_plot - std_plot,
        mean_plot + std_plot,
        alpha=0.25,
        color=_DARK_TEAL,
    )

    ax.set_xlabel(xlabel, fontsize=28)
    ax.set_ylabel(ylabel, fontsize=28)
    ax.tick_params(axis="both", labelsize=22)
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    max_iter = len(iterations) - 1
    step = 50 if max_iter >= 100 else (25 if max_iter >= 50 else 10)
    x_ticks = np.arange(0, max_iter + 1, step)
    if len(x_ticks) == 0 or x_ticks[-1] != max_iter:
        x_ticks = np.append(x_ticks, max_iter)
    ax.set_xticks(x_ticks)

    if log_scale:
        _format_log_y_axis(ax)

    if label is not None:
        ax.legend()

    plt.tight_layout()

    if fn is None:
        project_root = Path(__file__).parent.parent
        final_results_dir = project_root / "final_results"
        final_results_dir.mkdir(parents=True, exist_ok=True)
        fn = final_results_dir / "exploitability_mean_variance.pdf"

    _save_fig(fig, fn)

    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_exploitability_groups(
    exploitabilities_groups: list[list[np.ndarray]],
    labels: list[str] | None = None,
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: str | Path | None = None,
    log_scale: bool = False,
    colors: ColorsConfig | None = None,
    color_list: list[str] | None = None,
    legend_loc: str | None = None,
    show_legend: bool = True,
    plot_every_n: int = 1,
    marker: str | None = None,
    marker_list: list[str] | None = None,
    ax: Any | None = None,
) -> Figure | None:
    """Plot mean ± std for multiple groups of exploitability trajectories."""
    if len(exploitabilities_groups) == 0:
        raise ValueError("exploitabilities_groups cannot be empty")

    if labels is not None and len(labels) != len(exploitabilities_groups):
        raise ValueError(
            f"Number of labels ({len(labels)}) must match number of groups "
            f"({len(exploitabilities_groups)})"
        )

    if colors is None:
        colors = ColorsConfig()

    _default_colors = [
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

    if color_list is None:
        color_list = [
            _default_colors[i % len(_default_colors)]
            for i in range(len(exploitabilities_groups))
        ]
    elif len(color_list) < len(exploitabilities_groups):
        color_list = list(color_list) + [
            _default_colors[i % len(_default_colors)]
            for i in range(len(color_list), len(exploitabilities_groups))
        ]

    num_groups = len(exploitabilities_groups)
    external_ax = ax is not None
    if ax is None:
        if num_groups > 20:
            figsize = (14, 10)
        elif num_groups > 10:
            figsize = (14, 12)
        elif num_groups > 5:
            figsize = (10, 7)
        else:
            figsize = (9, 6)
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        fig.patch.set_facecolor(colors.figure_background)
    else:
        fig = ax.figure

    for group_idx, group in enumerate(exploitabilities_groups):
        if len(group) == 0:
            continue

        group_arrays = [np.array(exp) for exp in group]
        max_len = max(len(exp) for exp in group_arrays)
        padded = np.full((len(group_arrays), max_len), np.nan)
        for i, exp in enumerate(group_arrays):
            padded[i, : len(exp)] = exp

        mean_exp = np.nanmean(padded, axis=0)
        std_exp = np.nanstd(padded, axis=0)

        iterations = np.arange(len(mean_exp))
        label = labels[group_idx] if labels is not None else f"Group {group_idx + 1}"
        color = color_list[group_idx]

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

    axis_label_size = 34 if num_groups > 10 else 28
    ax.set_xlabel(xlabel, fontsize=axis_label_size)
    ax.set_ylabel(ylabel, fontsize=axis_label_size)
    tick_labelsize = 26 if num_groups > 10 else 22
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

    tick_step = 50 if max_iteration >= 100 else (25 if max_iteration >= 50 else 10)
    x_ticks = np.arange(0, max_iteration + 1, tick_step)
    if len(x_ticks) == 0 or x_ticks[-1] != max_iteration:
        x_ticks = np.append(x_ticks, max_iteration)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(i)) for i in x_ticks])
    margin = max_iteration * 0.03
    ax.set_xlim(left=-margin, right=max_iteration + margin)

    if log_scale:
        _format_log_y_axis(ax)

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
                fontsize = (
                    12
                    if num_groups > 20
                    else (13 if num_groups > 10 else (14 if num_groups > 5 else 15))
                )
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
            nrows = (num_groups + ncol - 1) // ncol

            if nrows > 6:
                y_offset, fontsize = -0.30, 24
            elif nrows > 4:
                y_offset, fontsize = -0.27, 25
            elif nrows > 2:
                y_offset, fontsize = -0.24, 26
            else:
                y_offset, fontsize = -0.21, 27

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
        nrows = (num_groups + 1) // 2
        if nrows > 6:
            plt.tight_layout(rect=(0, 0.36, 1, 1))
        elif nrows > 4:
            plt.tight_layout(rect=(0, 0.33, 1, 1))
        elif nrows > 2:
            plt.tight_layout(rect=(0, 0.30, 1, 1))
        else:
            plt.tight_layout(rect=(0, 0.27, 1, 1))

    if external_ax:
        return fig if return_fig else None

    if fn is None:
        project_root = Path(__file__).parent.parent
        final_results_dir = project_root / "final_results"
        final_results_dir.mkdir(parents=True, exist_ok=True)
        fn = final_results_dir / "exploitability_groups.pdf"

    _save_fig(fig, fn)

    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_exploitability_multiple_versions(
    environment: str,
    versions_withhyper: list[str],
    outputs_dir: str | Path = "outputs",
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: str | Path | None = None,
    log_scale: bool = False,
    colors: ColorsConfig | None = None,
    color_list: list[str] | None = None,
    cmap: str | Any | None = None,
    legend_loc: str | None = None,
    show_legend: bool = True,
    label_format: str = "algorithm",
    best_version: str | None = None,
    best_model_yaml_path: str | Path | None = None,
    write_best_model_yaml: bool = True,
    plot_every_n: int = 1,
    marker: str | None = None,
    marker_list: list[str] | None = None,
) -> Figure | None:
    """Plot exploitabilities for multiple algorithm versions on the same axes.

    Each version is shown as mean ± std over all seeds.
    Writes results/{environment}/best/best_model.yaml with the version that achieved
    the lowest final mean exploitability.

    Args:
        label_format: "algorithm" for short names (use for comparison across algos),
            "hyperparameters" for param strings (use for sweep within one algo),
            "full" for raw version names.
    """
    if len(versions_withhyper) == 0:
        raise ValueError("versions_withhyper cannot be empty")

    all_groups = group_exploitabilities_by_seed(environment, outputs_dir)

    final_means: dict[str, float] = {}
    exploitabilities_groups: list[list[np.ndarray]] = []
    labels: list[str] = []

    for version_withhyper in versions_withhyper:
        if version_withhyper not in all_groups:
            available = list(all_groups.keys())
            raise ValueError(
                f"Version '{version_withhyper}' not found. Available versions: {available}"
            )

        version_data = all_groups[version_withhyper]
        seed_groups: list[list[np.ndarray]] = version_data["groups"]

        if len(seed_groups) == 0:
            print(
                f"Warning: No exploitability data found for version '{version_withhyper}', skipping."
            )
            continue

        combined: list[np.ndarray] = []
        for sg in seed_groups:
            combined.extend(sg)

        if len(combined) > 0:
            exp_arrays = [np.array(exp) for exp in combined]
            max_len = max(len(exp) for exp in exp_arrays)
            padded = np.full((len(exp_arrays), max_len), np.nan)
            for i, exp in enumerate(exp_arrays):
                padded[i, : len(exp)] = exp

            mean_exp = np.nanmean(padded, axis=0)
            final_means[version_withhyper] = mean_exp[-1]
            exploitabilities_groups.append(combined)

            if label_format == "algorithm":
                label = version_to_algorithm_name(version_withhyper)
            elif label_format == "hyperparameters":
                hp = extract_hyperparameters(version_withhyper)
                label = hp if hp else version_withhyper
            else:
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
                                math_parts.append(
                                    f"\\mathbf{{{part.replace(' ', chr(92) + ' ')}}}"
                                )
                    label = f"${''.join(math_parts)}$"
                else:
                    label = f"$\\mathbf{{{label}}}$"

            labels.append(label)

    if len(exploitabilities_groups) == 0:
        raise ValueError("No valid exploitability data found for any version")

    if final_means and write_best_model_yaml:
        sorted_versions = sorted(final_means.items(), key=lambda x: x[1])
        best_version_name = sorted_versions[0][0]
        best_final_mean = sorted_versions[0][1]
        print(
            f"\nBest model: {best_version_name} with final mean exploitability: {best_final_mean:.6f}"
        )

        if best_model_yaml_path is None:
            project_root = Path(__file__).parent.parent
            results_dir = project_root / "results" / environment / "best"
            results_dir.mkdir(parents=True, exist_ok=True)
            yaml_path = results_dir / "best_model.yaml"
        else:
            yaml_path = Path(best_model_yaml_path)
            yaml_path.parent.mkdir(parents=True, exist_ok=True)
        best_seed_records = all_groups[best_version_name].get("seed_records", [])
        final_values = [
            seed_record["final_exploitability"] for seed_record in best_seed_records
        ]
        yaml_data = {
            "environment": environment,
            "algorithm": all_groups[best_version_name].get("algorithm"),
            "selection_policy": "latest_run",
            "best_version": best_version_name,
            "num_seeds": len(best_seed_records),
            "final_mean_exploitability": float(best_final_mean),
            "final_std_exploitability": (
                float(np.std(final_values)) if final_values else 0.0
            ),
            "seeds": best_seed_records,
            "all_versions": [
                {"version": v, "final_mean_exploitability": float(e)}
                for v, e in sorted_versions
            ],
        }
        with open(yaml_path, "w") as f:
            yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)
        print(f"Best model saved to: {yaml_path}")

    if cmap is not None and color_list is None:
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        colormap = cm.get_cmap(cmap) if isinstance(cmap, str) else cmap
        n = len(exploitabilities_groups)
        color_list = [
            mcolors.to_hex(colormap(i / (n - 1) if n > 1 else 0)) for i in range(n)
        ]

    if fn is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        safe = "_".join(v.replace("/", "_") for v in versions_withhyper[:3])
        if len(versions_withhyper) > 3:
            safe += f"_and_{len(versions_withhyper) - 3}_more"
        fn = results_dir / f"multiple_versions_{safe}.pdf"

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


# ---------------------------------------------------------------------------
# Runtime plot functions
# ---------------------------------------------------------------------------


def plot_runtime_bar(
    data: list[list[float]],
    labels: list[str],
    ylabel: str = "Wall-clock runtime (s)",
    return_fig: bool = False,
    fn: str | Path | None = None,
    colors: ColorsConfig | None = None,
    color_list: list[str] | None = None,
    legend_loc: str | None = None,
    show_legend: bool = True,
) -> Figure | None:
    """Horizontal box plot of runtimes on a log x-axis, one row per algorithm."""
    if colors is None:
        colors = ColorsConfig()

    default_colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
        "#937860",
        "#DA8BC3",
        "#8C8C8C",
        "#CCB974",
        "#64B5CD",
    ]
    if color_list is None:
        color_list = [default_colors[i % len(default_colors)] for i in range(len(data))]

    n = len(labels)
    fig_h = max(2.5, n * 0.55 + 1.0)
    fig, ax = plt.subplots(figsize=(5, fig_h))
    fig.patch.set_facecolor(colors.figure_background)

    bp = ax.boxplot(
        data,
        vert=False,
        patch_artist=True,
        widths=0.5,
        flierprops={"marker": "o", "markersize": 3, "linestyle": "none"},
        medianprops={"color": "black", "linewidth": 1.5},
    )

    for patch, flier, color in zip(bp["boxes"], bp["fliers"], color_list, strict=False):
        patch.set_facecolor(color)
        patch.set_alpha(0.75)
        flier.set_markerfacecolor(color)
        flier.set_markeredgecolor(color)

    for i, color in enumerate(color_list):
        for w in bp["whiskers"][2 * i : 2 * i + 2]:
            w.set_color(color)
        for c in bp["caps"][2 * i : 2 * i + 2]:
            c.set_color(color)

    ax.set_xscale("log")
    ax.set_yticks(range(1, n + 1))
    ax.set_yticklabels(labels, fontsize=13)
    ax.set_xlabel("Wall-clock runtime (s)", fontsize=13)
    ax.grid(True, axis="x", linestyle="--", linewidth=0.4, alpha=0.5)
    ax.tick_params(axis="x", labelsize=11)
    ax.invert_yaxis()

    plt.tight_layout()
    _save_fig(fig, fn)

    if return_fig:
        return fig
    plt.close(fig)
    return None


def plot_runtime_multiple_versions(
    environment: str,
    versions_withhyper: list[str],
    outputs_dir: str | Path = "outputs",
    ylabel: str = "Wall-clock runtime (s)",
    return_fig: bool = False,
    fn: str | Path | None = None,
    colors: ColorsConfig | None = None,
    color_list: list[str] | None = None,
    label_format: str = "algorithm",
    legend_loc: str | None = None,
    show_legend: bool = False,
) -> Figure | None:
    """Box plot of wall-clock runtimes for multiple algorithm versions."""
    if len(versions_withhyper) == 0:
        raise ValueError("versions_withhyper cannot be empty")

    all_runtime_data = group_runtimes_by_seed(environment, outputs_dir)

    raw_data: list[list[float]] = []
    labels: list[str] = []

    for version_withhyper in versions_withhyper:
        if version_withhyper not in all_runtime_data:
            print(f"Warning: no runtime data for '{version_withhyper}', skipping.")
            continue
        runtimes = all_runtime_data[version_withhyper]["runtimes"]
        if len(runtimes) == 0:
            print(f"Warning: empty runtimes for '{version_withhyper}', skipping.")
            continue
        raw_data.append(list(runtimes))

        if label_format == "algorithm":
            label = version_to_algorithm_name(version_withhyper)
        elif label_format == "hyperparameters":
            hp = extract_hyperparameters(version_withhyper)
            label = hp if hp else version_withhyper
        else:
            label = version_withhyper
        labels.append(label)

    if len(raw_data) == 0:
        raise ValueError("No runtime data found for any of the specified versions.")

    if fn is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        fn = results_dir / "runtime_comparison.pdf"

    return plot_runtime_bar(
        data=raw_data,
        labels=labels,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn,
        colors=colors,
        color_list=color_list,
        legend_loc=legend_loc,
        show_legend=show_legend,
    )
