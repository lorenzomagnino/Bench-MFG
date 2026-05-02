"""Plot exploitability sweep for all hyperparameter versions of one algorithm.

Usage:
    PYTHONPATH=src python -m utility.plot_sweep <environment> <algorithm> [options]
    PYTHONPATH=src python src/utility/plot_sweep.py <environment> <algorithm> [options]

Scans outputs/{environment}/{algorithm}/ for all hyperparameter versions, plots
mean ± std exploitability for each, and saves YAML artifacts for use by plot_comparison.

Example:
    PYTHONPATH=src python -m utility.plot_sweep LasryLionsChain PSO --log-scale
    PYTHONPATH=src python -m utility.plot_sweep LasryLionsChain OMD --legend-loc upper_right

Known algorithm directory names: PSO, OMD, DampedFP_damped, PI_smooth_policy_iteration,
PI_boltzmann_policy_iteration, DampedFP_fictitious_play, DampedFP_pure, PI_policy_iteration
"""

import argparse
from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from conf.visualization.visualization_schema import ColorsConfig  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
import numpy as np  # noqa: E402
from utility.plot_discovery import (  # noqa: E402
    ALGORITHMS,
    build_results_yaml_for_algorithm,
    group_exploitabilities_by_seed,
    write_best_model_yaml,
)
from utility.plot_primitives import (  # noqa: E402
    plot_exploitability_mean_variance,
    plot_exploitability_multiple_versions,
)


def plot_exploitability_by_version_and_seed(
    environment: str,
    version_withhyper: str,
    outputs_dir: str | Path = "outputs",
    xlabel: str = "Iteration",
    ylabel: str = "Exploitability",
    return_fig: bool = False,
    fn: str | Path | None = None,
    log_scale: bool = False,
    colors: ColorsConfig | None = None,
    color_list: list[str] | None = None,
) -> Figure | None:
    """Plot combined exploitabilities (all seeds) for a single hyperparameter version."""
    all_groups = group_exploitabilities_by_seed(environment, outputs_dir)

    if version_withhyper not in all_groups:
        available = list(all_groups.keys())
        raise ValueError(
            f"Version '{version_withhyper}' not found. Available: {available}"
        )

    version_data = all_groups[version_withhyper]
    seed_groups: list[list[np.ndarray]] = version_data["groups"]

    if len(seed_groups) == 0:
        raise ValueError(
            f"No exploitability data found for version '{version_withhyper}'"
        )

    combined: list[np.ndarray] = []
    for sg in seed_groups:
        combined.extend(sg)

    if len(combined) == 0:
        raise ValueError(
            f"No exploitability data found for version '{version_withhyper}'"
        )

    if fn is None:
        project_root = Path(__file__).parent.parent
        results_dir = project_root / "results" / environment
        results_dir.mkdir(parents=True, exist_ok=True)
        safe = version_withhyper.replace("/", "_").replace("\\", "_")
        fn = results_dir / f"{safe}_by_seed.pdf"

    return plot_exploitability_mean_variance(
        exploitabilities_list=combined,
        xlabel=xlabel,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn,
        log_scale=log_scale,
        colors=colors,
        label=None,
    )


def plot_exploitability_by_algorithm(
    environment: str,
    algorithm: str,
    outputs_dir: str | Path = "outputs",
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
) -> Figure | None:
    """Plot all hyperparameter versions for one algorithm.

    For each version, combines all seeds and plots mean ± std. All versions are
    shown on the same figure. Also writes results/{environment}/{algorithm}/results.yaml
    and results/{environment}/{algorithm}/best_model.yaml for downstream plotting.

    If there are more than 20 versions, splits into two figures (part1 / part2).
    """
    results_data = build_results_yaml_for_algorithm(environment, algorithm, outputs_dir)
    best_model_data = write_best_model_yaml(results_data, environment, algorithm)
    versions_withhyper = [
        configuration["version"]
        for configuration in results_data.get("configurations", [])
    ]

    if len(versions_withhyper) == 0:
        raise ValueError(
            f"No versions found for algorithm '{algorithm}' in environment '{environment}'"
        )

    print(
        f"Found {len(versions_withhyper)} versions for {algorithm}: {versions_withhyper}"
    )

    project_root = Path(__file__).parent.parent
    algo_results_dir = project_root / "results" / environment / algorithm
    algo_results_dir.mkdir(parents=True, exist_ok=True)

    final_means = {
        configuration["version"]: float(configuration["final_mean_exploitability"])
        for configuration in results_data.get("configurations", [])
    }

    if final_means:
        sorted_versions = sorted(final_means.items(), key=lambda x: x[1])
        best_version = sorted_versions[0][0]
        best_mean = sorted_versions[0][1]
        print(f"\nBest final mean exploitability: {best_version} = {best_mean:.6f}")
        print(f"\n{'Rank':<6} {'Version':<50} {'Final Exploitability':<20}")
        print("-" * 76)
        for rank, (v, e) in enumerate(sorted_versions, 1):
            print(f"{rank:<6} {v:<50} {e:.6f}")
        print(f"\nResults saved to: {algo_results_dir / 'results.yaml'}")
        print(f"Best model saved to: {algo_results_dir / 'best_model.yaml'}")

    best_version_for_legend = best_model_data["best_version"] if final_means else None

    if fn is None:
        fn = algo_results_dir / "exploitability.pdf"

    max_per_plot = 20
    if len(versions_withhyper) > max_per_plot:
        mid = len(versions_withhyper) // 2
        v_part1, v_part2 = versions_withhyper[:mid], versions_withhyper[mid:]

        bv_part1 = (
            best_version_for_legend if best_version_for_legend in v_part1 else None
        )
        bv_part2 = (
            best_version_for_legend if best_version_for_legend in v_part2 else None
        )

        print(
            f"\nSplitting {len(versions_withhyper)} versions into two figures: "
            f"Part 1 ({len(v_part1)}), Part 2 ({len(v_part2)})"
        )

        fn = Path(fn)
        fn_part1 = fn.parent / f"{fn.stem}_part1{fn.suffix}"
        fn_part2 = fn.parent / f"{fn.stem}_part2{fn.suffix}"

        plot_exploitability_multiple_versions(
            environment=environment,
            versions_withhyper=v_part1,
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
            best_version=bv_part1,
            best_model_yaml_path=algo_results_dir / "best_model.yaml",
            write_best_model_yaml=False,
            plot_every_n=plot_every_n,
            marker=marker,
        )
        print(f"Saved part 1 to: {fn_part1}")

        fig = plot_exploitability_multiple_versions(
            environment=environment,
            versions_withhyper=v_part2,
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
            best_version=bv_part2,
            best_model_yaml_path=algo_results_dir / "best_model.yaml",
            write_best_model_yaml=False,
            plot_every_n=plot_every_n,
            marker=marker,
        )
        print(f"Saved part 2 to: {fn_part2}")
        print(f"Sweep artifacts saved to: {algo_results_dir}")
        return fig
    else:
        fig = plot_exploitability_multiple_versions(
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
            best_model_yaml_path=algo_results_dir / "best_model.yaml",
            write_best_model_yaml=False,
            plot_every_n=plot_every_n,
            marker=marker,
        )
        print(f"Sweep artifacts saved to: {algo_results_dir}")
        return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Plot exploitability sweep for one algorithm across all hyperparameter versions."
    )
    parser.add_argument(
        "environment", type=str, help="Environment name (e.g. LasryLionsChain)"
    )
    parser.add_argument(
        "algorithm",
        type=str,
        help=f"Algorithm directory name. Known: {', '.join(ALGORITHMS)}",
    )
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--log-scale", action="store_true")
    parser.add_argument(
        "--legend-loc",
        type=str,
        default=None,
        help="Legend location, e.g. 'upper right'. Default: below plot.",
    )
    parser.add_argument(
        "--plot-every-n",
        type=int,
        default=10,
        help="Subsample: plot every N-th iteration (default: 10)",
    )
    args = parser.parse_args()

    legend_location = args.legend_loc.replace("_", " ") if args.legend_loc else None

    plot_exploitability_by_algorithm(
        environment=args.environment,
        algorithm=args.algorithm,
        outputs_dir=args.outputs_dir,
        log_scale=args.log_scale,
        legend_loc=legend_location,
        show_legend=True,
        plot_every_n=args.plot_every_n,
    )
