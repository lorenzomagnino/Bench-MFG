"""Compare best hyperparameter version of each algorithm for one environment.

Usage:
    PYTHONPATH=src python -m utility.plot_comparison <environment> [options]
    PYTHONPATH=src python src/utility/plot_comparison.py <environment> [options]

Prerequisite: run plot_sweep.py for each algorithm first — it writes
results/{environment}/{algorithm}/best_model.yaml which this script reads to
select the best hyperparameter version per algorithm.

Produces:
  - Exploitability comparison plot (mean ± std per algorithm)
  - Runtime box plot

Example:
    # Step 1: generate best_model.yaml for each algorithm
    for algo in PSO OMD DampedFP_damped PI_smooth_policy_iteration PI_boltzmann_policy_iteration; do
        PYTHONPATH=src python -m utility.plot_sweep MultipleEquilibriaGame $algo --log-scale
    done

    # Step 2: compare
    PYTHONPATH=src python -m utility.plot_comparison MultipleEquilibriaGame
"""

import argparse
from pathlib import Path
import sys

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from conf.visualization.visualization_schema import ColorsConfig  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from utility.plot_discovery import (  # noqa: E402
    DEFAULT_COMPARISON_ALGORITHMS,
    get_best_versions_by_algorithm,
)
from utility.plot_primitives import (  # noqa: E402
    plot_exploitability_multiple_versions,
    plot_runtime_multiple_versions,
)


def plot_runtime_for_env(
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
    """Plot runtime box chart for an environment using best hyperparameters per algorithm.

    Delegates to plot_runtime_multiple_versions using the resolved best versions.
    """
    if len(versions_withhyper) == 0:
        raise ValueError(
            f"No versions found for environment '{environment}'. "
            "Run plot_sweep.py for each algorithm first to generate best_model.yaml files."
        )

    if fn is None:
        project_root = Path(__file__).parent.parent
        env_results_dir = project_root / "results" / environment
        env_results_dir.mkdir(parents=True, exist_ok=True)
        fn = env_results_dir / "runtime_best_versions.pdf"

    return plot_runtime_multiple_versions(
        environment=environment,
        versions_withhyper=versions_withhyper,
        outputs_dir=outputs_dir,
        ylabel=ylabel,
        return_fig=return_fig,
        fn=fn,
        colors=colors,
        color_list=color_list,
        label_format=label_format,
        legend_loc=legend_loc,
        show_legend=show_legend,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare best version of each algorithm for one environment."
    )
    parser.add_argument(
        "environment", type=str, help="Environment name (e.g. MultipleEquilibriaGame)"
    )
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument(
        "--algorithms",
        type=str,
        nargs="+",
        default=None,
        help="Algorithms to compare. Default: all known algorithms.",
    )
    parser.add_argument(
        "--plot-every-n",
        type=int,
        default=10,
        help="Subsample: plot every N-th iteration (default: 10)",
    )
    parser.add_argument(
        "--legend-loc",
        type=str,
        default="upper right",
        help="Legend location (default: upper right)",
    )
    parser.add_argument(
        "--color-list",
        type=str,
        nargs="+",
        default=[
            "#F86262",
            "#F0816A",
            "#7F11F5",
            "#0936C8",
            "#63B0F8",
            "#FA8FBF",
            "#703F62",
            "#97B9C3",
        ],
    )
    parser.add_argument(
        "--marker-list",
        type=str,
        nargs="+",
        default=["o", "s", "D", "^", "v", "p", "h", "*"],
    )
    parser.add_argument(
        "--skip-runtime", action="store_true", help="Skip the runtime box plot"
    )
    args = parser.parse_args()

    legend_location = args.legend_loc.replace("_", " ") if args.legend_loc else None

    selected_algorithms = (
        args.algorithms
        if args.algorithms is not None
        else list(DEFAULT_COMPARISON_ALGORITHMS.keys())
    )
    best_versions_by_algorithm, missing_algorithms = get_best_versions_by_algorithm(
        environment=args.environment,
        algorithms=selected_algorithms,
    )
    ready_algorithms = [
        algorithm
        for algorithm in selected_algorithms
        if algorithm in best_versions_by_algorithm
    ]
    versions = [best_versions_by_algorithm[algorithm] for algorithm in ready_algorithms]

    if missing_algorithms:
        print(
            f"Missing sweep results for environment '{args.environment}': "
            f"{missing_algorithms}"
        )
        if ready_algorithms:
            ready_algorithms_str = " ".join(ready_algorithms)
            print(
                "\nReady algorithms found:"
                f" {ready_algorithms}\n"
                "To compare only the available ones, run:\n"
                f"  PYTHONPATH=src python -m utility.plot_comparison "
                f"{args.environment} --algorithms {ready_algorithms_str}"
            )
        else:
            print(
                "\nNo ready algorithms found. Run plot_sweep.py first for the "
                "algorithms you want to compare."
            )
        sys.exit(1)

    if not versions:
        print(
            f"No versions found for '{args.environment}'. "
            "Run plot_sweep.py for each algorithm first."
        )
        sys.exit(1)

    print(f"Comparing best versions for algorithms: {ready_algorithms}")
    print(f"Resolved versions: {versions}")

    project_root = Path(__file__).parent.parent
    comparison_dir = project_root / "results" / args.environment / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    exploitability_path = comparison_dir / "exploitability_best_versions.pdf"
    exploitability_log_path = comparison_dir / "exploitability_best_versions_log.pdf"
    runtime_path = comparison_dir / "runtime_best_versions.pdf"

    plot_exploitability_multiple_versions(
        environment=args.environment,
        versions_withhyper=versions,
        outputs_dir=args.outputs_dir,
        fn=exploitability_path,
        log_scale=False,
        color_list=args.color_list,
        legend_loc=legend_location,
        plot_every_n=args.plot_every_n,
        marker_list=args.marker_list,
        label_format="algorithm",
        write_best_model_yaml=False,
    )
    plot_exploitability_multiple_versions(
        environment=args.environment,
        versions_withhyper=versions,
        outputs_dir=args.outputs_dir,
        fn=exploitability_log_path,
        log_scale=True,
        color_list=args.color_list,
        legend_loc=legend_location,
        plot_every_n=args.plot_every_n,
        marker_list=args.marker_list,
        label_format="algorithm",
        write_best_model_yaml=False,
    )
    print(f"Exploitability plot saved to: {exploitability_path}")
    print(f"Log exploitability plot saved to: {exploitability_log_path}")

    if not args.skip_runtime:
        plot_runtime_for_env(
            environment=args.environment,
            versions_withhyper=versions,
            outputs_dir=args.outputs_dir,
            fn=runtime_path,
            label_format="algorithm",
        )
        print(f"Runtime plot saved to: {runtime_path}")
