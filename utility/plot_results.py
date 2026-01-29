"""Plot experimental results."""

from pathlib import Path
from typing import Optional

from conf.config_schema import MFGConfig
from utility.MFGPlots import plot_mean_field, plot_policy
from utility.path_utils import get_output_directory


def plot_results(results, cfg: MFGConfig, run_id: Optional[str] = None):
    """Plot experimental results
    - plot the mean field
    - plot the policy
    Args:
        results: Tuple of optimal policy, mean field, and exploitabilities.
        cfg: MFGConfig
        run_id: Unique run identifier. If None, plots are saved to base directory.
    Returns:
        Figure object for the mean field plot
        Figure object for the policy plot
    """
    optimal_policy, mean_field, _ = results

    base_dir = Path(get_output_directory(cfg))
    output_dir = base_dir / run_id if run_id is not None else base_dir
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    mean_field_fig = None
    policy_fig = None

    if cfg.visualization.show_mean_field_evolution:
        plot_filename_mean_field = (
            f"{cfg.experiment.name}_{cfg.algorithm._target_.lower()}_mean_field.pdf"
        )
        plot_path_mean_field = plots_dir / plot_filename_mean_field
        title_mean_field = f"{cfg.environment.name} - {cfg.algorithm._target_} Algorithm\nFinal Mean Field Distribution"
        is_grid = cfg.environment.grid.is_grid
        grid_dim = cfg.environment.grid.dimension if is_grid else None
        walls = None
        mean_field_fig = plot_mean_field(
            mean_field=mean_field,
            is_grid=is_grid,
            grid_dim=grid_dim,
            walls=walls,
            title=title_mean_field,
            fn=str(plot_path_mean_field),
            return_fig=True,
            colors=cfg.visualization.colors,
        )
    if cfg.visualization.show_policy_evolution:
        plot_filename_policy = (
            f"{cfg.experiment.name}_{cfg.algorithm._target_.lower()}_policy.pdf"
        )
        plot_path_policy = plots_dir / plot_filename_policy
        is_grid = cfg.environment.grid.is_grid
        grid_dim = cfg.environment.grid.dimension if is_grid else None
        walls = None
        policy_fig = plot_policy(
            policy_array=optimal_policy,
            is_grid=is_grid,
            grid_dim=grid_dim,
            walls=walls,
            return_fig=True,
            fn=str(plot_path_policy),
            colors=cfg.visualization.colors,
        )
    return mean_field_fig, policy_fig
