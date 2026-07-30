"""Plot utilities for MFG experiments."""

from matplotlib.colors import to_rgba
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np

from benchmfg.conf.visualization.visualization_schema import ColorsConfig


def _get_color_config_value(colors: ColorsConfig, key: str):
    """Read a color setting with fallback to schema defaults for older configs."""
    default_colors = ColorsConfig()
    try:
        value = getattr(colors, key)
    except Exception:
        value = getattr(default_colors, key)
    return value


def plot_mean_field(
    mean_field,
    is_grid,
    grid_dim=None,
    walls=None,
    return_fig=False,
    fn=None,
    title=None,
    colors: ColorsConfig | None = None,
    background_color: str | None = None,
    bar_color: str | None = None,
    grid_color: str | None = None,
    cmap_2d: str | None = None,
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
    colors: ColorsConfig | None = None,
    background_color: str | None = None,
    bar_color: str | None = None,
    grid_color: str | None = None,
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
        from benchmfg.conf.visualization.visualization_schema import ColorsConfig

        colors = ColorsConfig()

    N_states = len(mean_field)
    states = np.arange(N_states)
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    fig.patch.set_facecolor(_get_color_config_value(colors, "figure_background"))
    bg_color = (
        background_color
        if background_color is not None
        else _get_color_config_value(colors, "mean_field_1d_background")
    )
    ax.set_facecolor(bg_color)

    barwidth = 0.8
    bar_col = (
        bar_color
        if bar_color is not None
        else _get_color_config_value(colors, "mean_field_1d_bar")
    )
    face_col = to_rgba(
        bar_col,
        alpha=_get_color_config_value(colors, "mean_field_1d_bar_alpha"),
    )
    ax.bar(
        states,
        mean_field,
        color=face_col,
        width=barwidth,
        edgecolor=bar_col,
        linewidth=_get_color_config_value(colors, "mean_field_1d_bar_linewidth"),
    )
    ax.set_xlabel("States", fontsize=28)
    ax.set_ylabel("Probability Mass", fontsize=28)
    grid_col = (
        grid_color
        if grid_color is not None
        else _get_color_config_value(colors, "mean_field_1d_grid")
    )
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
    ax.tick_params(axis="both", which="major", labelsize=26)

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
    colors: ColorsConfig | None = None,
    cmap_2d: str | None = None,
):
    mean_field = np.array(mean_field)
    n_rows, n_cols = grid_dim
    N_flat = n_rows * n_cols

    assert len(mean_field) == N_flat
    if walls is not None:
        assert len(walls) == N_flat

    if colors is None:
        from benchmfg.conf.visualization.visualization_schema import ColorsConfig

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

    x_ticks = [0, n_cols // 2, n_cols - 1]
    y_ticks = [0, n_rows // 2, n_rows - 1]

    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i) for i in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i) for i in y_ticks])
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
    from matplotlib.ticker import FixedFormatter, FixedLocator

    vmin = float(mean_field.min())
    vmax = float(mean_field.max())
    vmid = round((vmin + vmax) / 2, 3)
    cbar.ax.yaxis.set_major_locator(FixedLocator([vmin, vmid, vmax]))
    cbar.ax.yaxis.set_major_formatter(
        FixedFormatter([f"{vmin:.3f}", f"{vmid:.3f}", f"{vmax:.3f}"])
    )
    cbar.ax.tick_params(labelsize=22)

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
    colors: ColorsConfig | None = None,
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
        from benchmfg.conf.visualization.visualization_schema import ColorsConfig

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
    colors: ColorsConfig | None = None,
    action_labels: list[str] | None = None,
    action_cmaps: list[str] | None = None,
    cmap: str | None = None,
    cmap_2d: str | None = None,
    show_interval_in_labels: bool = True,
    tick_step: int | None = None,
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
    colors: ColorsConfig | None = None,
    action_labels: list[str] | None = None,
    cmap: str | None = None,
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
        from benchmfg.conf.visualization.visualization_schema import ColorsConfig

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
    policy_cmap = (
        cmap if cmap is not None else _get_color_config_value(colors, "policy_cmap")
    )
    # Explicitly set vmin and vmax to [0, 1] to ensure colorbar shows correct range
    c = ax.imshow(
        policy_array.T,
        cmap=policy_cmap,
        aspect="auto",
        origin="lower",
        vmin=0.0,
        vmax=1.0,
    )
    # 3 evenly-spaced x ticks (start, middle, end)
    x_ticks = [0, N_states // 2, N_states - 1]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i) for i in x_ticks])

    ax.set_yticks(np.arange(N_actions))
    if action_labels is None:
        action_labels = [str(i) for i in range(N_actions)]
    elif len(action_labels) < N_actions:
        action_labels = list(action_labels) + [
            str(i) for i in range(len(action_labels), N_actions)
        ]
    ax.set_yticklabels([action_labels[i] for i in range(N_actions)])
    ax.set_xlabel("States", fontsize=28)
    ax.set_ylabel("Actions", fontsize=28)
    ax.set_xticks(np.arange(-0.5, N_states, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, N_actions, 1), minor=True)
    ax.grid(which="minor", color=colors.policy_grid, linestyle="--", linewidth=0.5)
    ax.tick_params(axis="both", which="major", labelsize=26)

    from matplotlib.ticker import FixedFormatter, FixedLocator

    cbar = fig.colorbar(c, ax=ax)
    cbar.ax.yaxis.set_major_locator(FixedLocator([0.0, 0.5, 1.0]))
    cbar.ax.yaxis.set_major_formatter(FixedFormatter(["0.0", "0.5", "1.0"]))
    cbar.ax.tick_params(labelsize=26)

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
    colors: ColorsConfig | None = None,
    action_cmaps: list[str] | None = None,
    action_labels: list[str] | None = None,
    cmap_2d: str | None = None,
    show_interval_in_labels: bool = True,
    tick_step: int | None = None,
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
        from benchmfg.conf.visualization.visualization_schema import ColorsConfig

        colors = ColorsConfig()

    if n_rows == 11 and n_cols == 11 and tick_step is None:
        tick_step = 2  # Show ticks at 0, 2, 4, etc.

    if action_cmaps is None:
        cfg_cmaps = _get_color_config_value(colors, "policy2d_action_cmaps")
        if cfg_cmaps is not None and len(cfg_cmaps) >= N_actions:
            action_cmaps = cfg_cmaps[:N_actions]
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
        cfg_labels = _get_color_config_value(colors, "policy2d_action_labels")
        if cfg_labels is not None and len(cfg_labels) >= N_actions:
            action_labels = cfg_labels[:N_actions]
        else:
            action_labels = [f"Action {i}" for i in range(N_actions)]
    else:
        assert (
            len(action_labels) == N_actions
        ), f"Number of action labels ({len(action_labels)}) must match number of actions ({N_actions})"

    import matplotlib.colors as mcolors

    # Resolve solid colors for each action
    cfg_action_colors = _get_color_config_value(colors, "policy2d_action_colors")
    if cfg_action_colors is not None and len(cfg_action_colors) >= N_actions:
        action_colors_hex = cfg_action_colors[:N_actions]
    else:
        action_colors_hex = ["#7B2FBE", "#4A55A2", "#FA8072", "#2D6A4F", "#1B7A7A"][
            :N_actions
        ]

    action_colors_rgb = [mcolors.to_rgb(c) for c in action_colors_hex]

    fig, ax = plt.subplots(1, 1, figsize=(6, 5))

    best_actions = np.argmax(policy_array, axis=1)
    rgb_image = np.zeros((n_rows, n_cols, 3))

    action_info = []

    for action_idx in range(N_actions):
        action_mask = best_actions == action_idx
        if np.any(action_mask):
            mask_2d = action_mask.reshape(grid_dim)
            rgb_image[mask_2d, :] = action_colors_rgb[action_idx]

        action_info.append(
            {
                "action_idx": action_idx,
                "action_label": action_labels[action_idx],
                "color": action_colors_hex[action_idx],
            }
        )

    # Paint wall cells white before rendering
    wall_mask_2d = None
    if walls is not None:
        walls_reshaped = np.array(walls).reshape(grid_dim)
        wall_mask_2d = walls_reshaped == 0
        rgb_image[wall_mask_2d, :] = 1.0

    ax.imshow(
        rgb_image,
        origin="lower",
        interpolation="nearest",
        aspect="equal",
    )

    # Draw wall cells as white rectangles with black border
    if wall_mask_2d is not None:
        from matplotlib.patches import Rectangle as _Rect

        for r in range(n_rows):
            for c in range(n_cols):
                if wall_mask_2d[r, c]:
                    ax.add_patch(
                        _Rect(
                            (c - 0.5, r - 0.5),
                            1,
                            1,
                            linewidth=1.5,
                            edgecolor="black",
                            facecolor="white",
                        )
                    )

    ax.set_xlabel("X-axis", fontsize=28)
    ax.set_ylabel("Y-axis", fontsize=28)

    x_ticks = [0, n_cols // 2, n_cols - 1]
    y_ticks = [0, n_rows // 2, n_rows - 1]
    ax.set_xticks(x_ticks)
    ax.set_xticklabels([str(i) for i in x_ticks])
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([str(i) for i in y_ticks])

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

    patches, labels = [], []
    for info in action_info:
        patch = Rectangle(
            (0, 0),
            1,
            1,
            facecolor=info["color"],
            edgecolor="none",
            alpha=1.0,
        )
        patches.append(patch)
        labels.append(info["action_label"])

    ax.legend(
        patches,
        labels,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=20,
        frameon=False,
        handlelength=1.5,
        handleheight=1.5,
    )

    plt.tight_layout()

    if fn is not None:
        if fn.lower().endswith(".pdf"):
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, format="pdf")
        else:
            plt.savefig(fn, bbox_inches="tight", pad_inches=0.1, dpi=300)

    if return_fig:
        return fig
