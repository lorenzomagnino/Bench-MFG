"""Plot utilities for MFG experiments."""

from typing import List, Optional

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from conf.visualization.visualization_schema import ColorsConfig


def plot_mean_field(
    mean_field,
    is_grid,
    grid_dim=None,
    walls=None,
    return_fig=False,
    fn=None,
    title=None,
    colors: Optional[ColorsConfig] = None,
    background_color: Optional[str] = None,
    bar_color: Optional[str] = None,
    grid_color: Optional[str] = None,
    cmap_2d: Optional[str] = None,
):
    """Plot the mean field.

    Args:
        mean_field: The mean field to plot.
        is_grid: Whether the environment is a grid.
        grid_dim: The dimension of the grid.
        walls: The walls of the grid.
        colors: Optional color configuration.
        background_color: Optional background color for the plot (e.g., "#D0D8E0").
        bar_color: Optional color for the bars (e.g., "#0F3E66").
        grid_color: Optional color for the grid lines (e.g., "gray", "#808080").
        cmap_2d: Optional colormap name for 2D grid plots (e.g., "viridis", "plasma").
    """
    if is_grid:
        if grid_dim is None:
            raise ValueError("grid_dim must be provided when is_grid=True")
        return plot_mean_field_evolution_2D(
            mean_field, grid_dim, walls, return_fig, fn, title, colors, cmap_2d
        )
    else:
        return plot_mean_field_evolution_1D(
            mean_field,
            return_fig,
            fn,
            title,
            colors,
            background_color,
            bar_color,
            grid_color,
        )


def plot_mean_field_evolution_1D(
    mean_field,
    return_fig=False,
    fn=None,
    title=None,
    colors: Optional[ColorsConfig] = None,
    background_color: Optional[str] = None,
    bar_color: Optional[str] = None,
    grid_color: Optional[str] = None,
):
    """
    Plots the mean field for 1D environments. Handles both time-dependent and stationary cases.

    Parameters:
    - mean_field: A numpy array of shape (N_states,) for stationary mean field
                  or (T, N_states) for time-dependent mean field evolution.
    - return_fig: If True, returns the figure object.
    - fn: If provided, saves the figure to the specified filename.
    - title: Optional title for the plot.
    - colors: Optional color configuration.
    - background_color: Optional background color for the plot (e.g., "#D0D8E0").
    - bar_color: Optional color for the bars (e.g., "#0F3E66").
    - grid_color: Optional color for the grid lines (e.g., "gray", "#808080").
    """
    mean_field = np.array(mean_field)
    if mean_field.ndim != 1:
        return f"mean_field must be 1D, got {mean_field.ndim}D array"

    if colors is None:
        from conf.visualization.visualization_schema import ColorsConfig

        colors = ColorsConfig()

    N_states = len(mean_field)
    states = np.arange(N_states)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(colors.figure_background)
    bg_color = background_color if background_color is not None else "#D0D8E0"
    ax.set_facecolor(bg_color)

    barwidth = 0.8
    bar_col = bar_color if bar_color is not None else "#0F3E66"
    ax.bar(
        states,
        mean_field,
        color=bar_col,
        width=barwidth,
        edgecolor=None,
        alpha=0.8,
    )
    ax.set_xlabel("States", fontsize=28)
    ax.set_ylabel("Probability Mass", fontsize=28)
    grid_col = grid_color if grid_color is not None else colors.mean_field_1d_grid
    ax.grid(
        True,
        which="both",
        linestyle="--",
        linewidth=0.5,
        color=grid_col,
        alpha=0.6,
    )
    y_max_data = np.max(mean_field) * 1.15
    y_max = min(y_max_data, 1.0)
    ax.set_ylim(0, y_max)

    x_ticks = states[::2]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i) for i in x_ticks])

    y_ticks = np.arange(0, min(y_max + 0.2, 1.2), 0.2)
    if y_max >= 0.9:
        y_ticks = np.arange(0, 1.1, 0.2)
    ax.set_yticks(y_ticks)
    ax.tick_params(axis="both", which="major", labelsize=22)

    plt.tight_layout()

    if fn is not None:
        if fn.lower().endswith(".pdf"):
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
        else:
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig


def plot_mean_field_evolution_2D(
    mean_field,
    grid_dim,
    walls=None,
    return_fig=False,
    fn=None,
    title=None,
    colors: Optional[ColorsConfig] = None,
    cmap_2d: Optional[str] = None,
):
    mean_field = np.array(mean_field)
    n_rows, n_cols = grid_dim
    N_flat = n_rows * n_cols

    assert len(mean_field) == N_flat
    if walls is not None:
        assert len(walls) == N_flat

    if colors is None:
        from conf.visualization.visualization_schema import ColorsConfig

        colors = ColorsConfig()

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))
    mean_field_2d = mean_field.reshape(grid_dim)

    if cmap_2d is not None:
        cmap_name = cmap_2d
    elif colors is not None and hasattr(colors, "mean_field_2d_cmap"):
        cmap_name = colors.mean_field_2d_cmap
    else:
        cmap_name = "viridis"
    im = ax.imshow(
        mean_field_2d,
        cmap=cmap_name,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )

    if walls is not None:
        walls_reshaped = np.array(walls).reshape(grid_dim)
        wall_image = np.ones((n_rows, n_cols, 4))  # RGBA
        wall_image[:, :, :3] = 0.3  # Dark grey RGB (0.3, 0.3, 0.3)
        wall_image[:, :, 3] = 1.0
        wall_mask = walls_reshaped == 0
        wall_image[~wall_mask, 3] = 0.0
        ax.imshow(
            wall_image,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
        )

    ax.set_xlabel("X-axis", fontsize=22)
    ax.set_ylabel("Y-axis", fontsize=22)

    if n_cols <= 10:
        x_step = 1
    elif n_cols <= 20:
        x_step = 2
    else:
        x_step = max(1, n_cols // 10)

    if n_rows <= 10:
        y_step = 1
    elif n_rows <= 20:
        y_step = 2
    else:
        y_step = max(1, n_rows // 10)

    x_ticks = np.arange(0, n_cols, x_step)
    if x_ticks[-1] != n_cols - 1:
        x_ticks = np.append(x_ticks, n_cols - 1)

    y_ticks = np.arange(0, n_rows, y_step)
    if y_ticks[-1] != n_rows - 1:
        y_ticks = np.append(y_ticks, n_rows - 1)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(i)) for i in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(i)) for i in y_ticks])
    ax.tick_params(axis="both", which="major", labelsize=20)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)

    ax.grid(
        which="minor", color=colors.mean_field_2d_grid, linestyle="-", linewidth=1.0
    )
    ax.tick_params(which="minor", size=0)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = fig.colorbar(im, cax=cax)
    cbar.ax.tick_params(labelsize=18)

    plt.tight_layout()

    if fn is not None:
        if fn.lower().endswith(".pdf"):
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
        else:
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig


def plot_mean_field_evolution_3D(
    mean_field,
    return_fig=False,
    fn=None,
    title=None,
    colors: Optional[ColorsConfig] = None,
):
    """
    Plots the mean field evolution for a 1D mean field over time steps in 3D.

    Parameters:
    - mean_field: A numpy array of shape (T, N_states) representing the mean field over time.
    - return_fig: If True, returns the figure object.
    - fn: If provided, saves the figure to the specified filename.
    - colors: Optional color configuration.
    """
    if colors is None:
        from conf.visualization.visualization_schema import ColorsConfig

        colors = ColorsConfig()

    T, N_states = mean_field.shape
    time = np.arange(T)
    states = np.arange(N_states)
    Time, States = np.meshgrid(time, states)
    Z = mean_field.T
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(Time, States, Z, cmap=colors.mean_field_3d_cmap)

    ax.set_xlabel("Time")
    ax.set_ylabel("States")
    ax.set_zlabel("Mean Field Value")
    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title("Mean Field Evolution")

    cbar = fig.colorbar(surf, shrink=0.5, aspect=8, pad=0.1)
    cbar.set_label("Mean Field Value", rotation=270, labelpad=15)

    if fn is not None:
        plt.savefig(fn, bbox_inches="tight", pad_inches=0.1)

    if return_fig:
        return fig

    plt.close(fig)


def plot_policy(
    policy_array,
    is_grid=False,
    grid_dim=None,
    walls=None,
    return_fig=False,
    fn=None,
    colors: Optional[ColorsConfig] = None,
    action_labels: Optional[List[str]] = None,
    action_cmaps: Optional[List[str]] = None,
    cmap: Optional[str] = None,
    cmap_2d: Optional[str] = None,
    show_interval_in_labels: bool = True,
    tick_step: Optional[int] = None,
):
    """Plot the policy.

    Args:
        policy_array: The policy array to plot.
            - For 1D: shape (N_steps, N_actions, N_states)
            - For 2D: shape (N_states, N_actions) where N_states = n_rows * n_cols
        grid_dim: Tuple (n_rows, n_cols) specifying the grid dimensions. If provided, plots as 2D grid.
        walls: Optional array for 2D plots indicating wall positions (0 = wall, 1 = free).
        return_fig: If True, returns the figure object.
        fn: If provided, saves the figure to the specified filename.
        colors: Optional color configuration.
        action_labels: Optional list of action labels for 1D plots.
        action_cmaps: Optional list of colormap names for 2D plots, one for each action.
        cmap: Optional colormap name for 1D policy plots (e.g., "viridis", "plasma").
        cmap_2d: Optional colormap name for 2D grid plots (currently not used, policy 2D uses action_cmaps).
        show_interval_in_labels: If True, show probability intervals in legend labels. Default True.
        tick_step: Step size for axis ticks (e.g., 2 for 0, 2, 4, ...). If None, auto-detects for 11x11 grids.
    """
    if grid_dim is not None:
        return plot_policy_2D(
            policy_array,
            grid_dim,
            walls,
            return_fig,
            fn,
            colors,
            action_cmaps,
            action_labels,
            cmap_2d,
            show_interval_in_labels,
            tick_step,
        )
    else:
        return plot_policy_1D(policy_array, return_fig, fn, colors, action_labels, cmap)


def plot_policy_1D(
    policy_array,
    return_fig=False,
    fn=None,
    colors: Optional[ColorsConfig] = None,
    action_labels: Optional[List[str]] = None,
    cmap: Optional[str] = None,
):
    """
    Plots a stationary policy array.

    Parameters:
    - policy_array: A numpy array of shape (N_states, N_actions) for stationary policy.
                    Each element corresponds to the probability of taking a specific action
                    at a specific state.
    - colors: Optional color configuration.
    - action_labels: Optional list of action labels.
    - cmap: Optional colormap name (e.g., "viridis", "plasma"). If None, uses default from colors.
    """
    if colors is None:
        from conf.visualization.visualization_schema import ColorsConfig

        colors = ColorsConfig()

    policy_array = np.array(policy_array)
    assert (
        policy_array.ndim == 2
    ), f"Policy array must be 2D, got shape {policy_array.shape}"

    N_states, N_actions = policy_array.shape

    # Ensure policy values are in [0, 1] range (normalize if needed)
    # Policy should be probabilities, but handle cases where it might be logits or unnormalized
    policy_min = policy_array.min()
    policy_max = policy_array.max()

    # If values are outside [0, 1], they might be logits - but for now, just clip/normalize
    # Check if values look like probabilities (should be in reasonable [0, 1] range)
    if policy_max > 1.0 or policy_min < 0.0:
        # Values might be unnormalized or in wrong format
        # For safety, normalize each row (state) to sum to 1
        policy_array = policy_array / (policy_array.sum(axis=1, keepdims=True) + 1e-10)

    # Ensure values are in [0, 1] for display
    policy_array = np.clip(policy_array, 0.0, 1.0)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    # Use provided cmap or default from colors
    policy_cmap = cmap if cmap is not None else colors.policy_cmap
    # Explicitly set vmin and vmax to [0, 1] to ensure colorbar shows correct range
    c = ax.imshow(
        policy_array.T,
        cmap=policy_cmap,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    ax.set_xticks(np.arange(N_states))
    ax.set_yticks(np.arange(N_actions))
    if action_labels is None:
        # Generate default action labels based on number of actions
        action_labels = [str(i) for i in range(N_actions)]
    elif len(action_labels) < N_actions:
        # If provided labels are insufficient, extend with default labels
        action_labels = list(action_labels) + [
            str(i) for i in range(len(action_labels), N_actions)
        ]
    ax.set_yticklabels([action_labels[i] for i in range(N_actions)])
    ax.set_xlabel("States", fontsize=28)
    ax.set_ylabel("Actions", fontsize=28)
    ax.set_xticks(np.arange(-0.5, N_states, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_actions, 1), minor=True)
    ax.grid(which="minor", color=colors.policy_grid, linestyle="--", linewidth=0.5)
    # Make tick numbers bigger
    ax.tick_params(axis="both", which="major", labelsize=22)

    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.tick_params(labelsize=18)
    # Format colorbar to show values between 0 and 1 with appropriate precision
    from matplotlib.ticker import MaxNLocator, ScalarFormatter

    # Set reasonable number of ticks (e.g., 5 ticks for 0 to 1 range)
    cbar.locator = MaxNLocator(nbins=5)
    cbar.update_ticks()
    # Format as decimals (e.g., 0.0, 0.25, 0.5, 0.75, 1.0) without scientific notation
    formatter = ScalarFormatter(useMathText=False)
    formatter.set_scientific(False)
    formatter.set_useOffset(False)
    cbar.ax.yaxis.set_major_formatter(formatter)

    plt.tight_layout()

    if fn is not None:
        if fn.lower().endswith(".pdf"):
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
        else:
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig

    plt.close(fig)


def plot_policy_2D(
    policy_array,
    grid_dim,
    walls=None,
    return_fig=False,
    fn=None,
    colors: Optional[ColorsConfig] = None,
    action_cmaps: Optional[List[str]] = None,
    action_labels: Optional[List[str]] = None,
    cmap_2d: Optional[str] = None,
    show_interval_in_labels: bool = True,
    tick_step: Optional[int] = None,
):
    """
    Plots the policy as a heatmap on a 2D grid space using different colormaps for each action.
    For each state, the colormap of the action with maximum probability is used.

    Parameters:
    - policy_array: A numpy array of shape (N_states, N_actions) where N_states = n_rows * n_cols.
                    Each element corresponds to the probability of taking a specific action at a specific state.
    - grid_dim: Tuple (n_rows, n_cols) specifying the grid dimensions.
    - walls: Optional array of shape (N_states,) indicating wall positions (0 = wall, 1 = free).
    - return_fig: If True, returns the figure object.
    - fn: If provided, saves the figure to the specified filename.
    - colors: Optional color configuration.
    - action_cmaps: Optional list of colormap names, one for each action.
                    If None, uses default colormaps: ['viridis', 'plasma', 'coolwarm', 'magma', 'inferno'].
    - action_labels: Optional list of action labels. If None, uses default labels based on action index.
                     For 2D environments: ['up', 'right', 'down', 'left', 'stay'].
    """
    policy_array = np.array(policy_array)
    n_rows, n_cols = grid_dim
    N_flat = n_rows * n_cols
    N_actions = policy_array.shape[1]

    assert (
        policy_array.shape[0] == N_flat
    ), f"Policy array first dimension ({policy_array.shape[0]}) must match grid size ({N_flat})"
    assert (
        policy_array.ndim == 2
    ), f"Policy array must be 2D (N_states, N_actions), got shape {policy_array.shape}"

    if walls is not None:
        assert len(walls) == N_flat

    if colors is None:
        from conf.visualization.visualization_schema import ColorsConfig

        colors = ColorsConfig()

    if n_rows == 11 and n_cols == 11:
        if tick_step is None:
            tick_step = 2  # Show ticks at 0, 2, 4, etc.
        show_interval_in_labels = False

    if action_cmaps is None:
        if (
            colors.policy2d_action_cmaps is not None
            and len(colors.policy2d_action_cmaps) == N_actions
        ):
            action_cmaps = colors.policy2d_action_cmaps
        elif (
            colors.policy2d_action_cmaps is not None
            and len(colors.policy2d_action_cmaps) >= N_actions
        ):
            action_cmaps = colors.policy2d_action_cmaps[:N_actions]
        else:
            default_cmaps = [
                "Greens",
                "Purples",
                "Oranges",
                "Blues",
                "Greys",
                "Reds",
                "YlGn",
            ]
            action_cmaps = [
                default_cmaps[i % len(default_cmaps)] for i in range(N_actions)
            ]
    else:
        assert (
            len(action_cmaps) == N_actions
        ), f"Number of colormaps ({len(action_cmaps)}) must match number of actions ({N_actions})"

    # Default action labels from config if available
    if action_labels is None:
        if (
            colors.policy2d_action_labels is not None
            and len(colors.policy2d_action_labels) == N_actions
        ):
            action_labels = colors.policy2d_action_labels
        elif (
            colors.policy2d_action_labels is not None
            and len(colors.policy2d_action_labels) >= N_actions
        ):
            action_labels = colors.policy2d_action_labels[:N_actions]
        else:
            action_labels = [f"Action {i}" for i in range(N_actions)]
    else:
        assert (
            len(action_labels) == N_actions
        ), f"Number of action labels ({len(action_labels)}) must match number of actions ({N_actions})"

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    best_actions = np.argmax(policy_array, axis=1, out=None)

    rgb_image = np.zeros((n_rows, n_cols, 3))

    action_info = []

    for action_idx in range(N_actions):
        action_probs = policy_array[:, action_idx]
        action_probs_2d = action_probs.reshape(grid_dim)

        prob_min = action_probs_2d.min()
        prob_max = action_probs_2d.max()

        cmap = plt.get_cmap(action_cmaps[action_idx])

        action_mask = best_actions == action_idx
        if np.any(action_mask):
            normalized_probs = np.clip(action_probs_2d, 0.0, 1.0)

            rgba = cmap(normalized_probs)
            mask_2d = action_mask.reshape(grid_dim)
            rgb_image[mask_2d, :] = rgba[mask_2d, :3]

        action_info.append(
            {
                "action_idx": action_idx,
                "action_label": action_labels[action_idx],
                "cmap": cmap,
                "prob_min": prob_min,
                "prob_max": prob_max,
            }
        )

    ax.imshow(
        rgb_image,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )

    # Overlay walls in dark grey if provided
    if walls is not None:
        walls_reshaped = np.array(walls).reshape(grid_dim)
        wall_image = np.ones((n_rows, n_cols, 4))  # RGBA
        wall_image[:, :, :3] = 0.3  # Dark grey RGB (0.3, 0.3, 0.3)
        wall_image[:, :, 3] = 1.0
        wall_mask = walls_reshaped == 0
        wall_image[~wall_mask, 3] = 0.0
        ax.imshow(
            wall_image,
            origin="lower",
            interpolation="nearest",
            aspect="equal",
        )

    ax.set_xlabel("X-axis", fontsize=28)
    ax.set_ylabel("Y-axis", fontsize=28)

    if tick_step is None:
        tick_step = 2 if n_cols == 11 and n_rows == 11 else 1

    x_ticks = np.arange(0, n_cols, tick_step)
    if x_ticks[-1] != n_cols - 1:
        x_ticks = np.append(x_ticks, n_cols - 1)
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(int(i)) for i in x_ticks])

    y_ticks = np.arange(0, n_rows, tick_step)
    if y_ticks[-1] != n_rows - 1:
        y_ticks = np.append(y_ticks, n_rows - 1)
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(int(i)) for i in y_ticks])

    ax.tick_params(axis="both", which="major", labelsize=22)

    ax.set_xticks(np.arange(-0.5, n_cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_rows, 1), minor=True)

    ax.grid(
        which="minor", color=colors.mean_field_2d_grid, linestyle="-", linewidth=1.0
    )
    ax.tick_params(which="minor", size=0)

    ax.set_xlim(-0.5, n_cols - 0.5)
    ax.set_ylim(-0.5, n_rows - 0.5)

    from matplotlib.patches import Rectangle

    legend_elements = []
    for info in action_info:
        cmap = info["cmap"]
        max_prob_clipped = np.clip(info["prob_max"], 0.0, 1.0)
        color = cmap(max_prob_clipped)

        if show_interval_in_labels:
            label = f"{info['action_label']}\n[{info['prob_min']:.2f}, {info['prob_max']:.3f}]"
        else:
            label = info["action_label"]

        patch = Rectangle(
            (0, 0), 1, 1, facecolor=color, edgecolor="black", linewidth=0.5
        )
        legend_elements.append((patch, label))

    patches = [elem[0] for elem in legend_elements]
    labels = [elem[1] for elem in legend_elements]

    ax.legend(
        patches,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=16,
        frameon=False,
    )

    plt.tight_layout()

    if fn is not None:
        if fn.lower().endswith(".pdf"):
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
        else:
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig
