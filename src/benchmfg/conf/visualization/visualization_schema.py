from dataclasses import dataclass, field


@dataclass
class ColorsConfig:
    """Color configuration for visualization."""

    mean_field_1d_bar: str = "#2B6C96"
    mean_field_1d_background: str = "white"
    mean_field_1d_bar_alpha: float = 0.45
    mean_field_1d_bar_linewidth: float = 4.0
    mean_field_1d_grid: str = "lightgray"

    mean_field_2d_cmap: str = "BuPu"
    mean_field_2d_walls_cmap: str = "Greys"
    mean_field_2d_grid: str = "lightgray"

    mean_field_3d_cmap: str = "viridis"

    policy_cmap: str = "RdPu"
    policy_grid: str = "gray"

    figure_background: str = "white"

    policy2d_action_cmaps: list[str] | None = field(
        default_factory=lambda: ["Greens", "Purples", "Oranges", "Blues", "Greys"]
    )
    policy2d_action_labels: list[str] | None = field(
        default_factory=lambda: ["up", "right", "down", "left", "stay"]
    )
    policy2d_action_colors: list[str] | None = field(
        default_factory=lambda: [
            "#7B2FBE",  # purple
            "#4A55A2",  # blue-purplish
            "#FA8072",  # salmon
            "#2D6A4F",  # dark cold green
            "#1B7A7A",  # teal
        ]
    )
