"""Command line interface for BenchMFG."""

from __future__ import annotations

import argparse
from importlib.resources import files
from pathlib import Path
import runpy
import sys

from benchmfg.utility.plot_discovery import ALGORITHMS


def _config_options(group: str) -> list[str]:
    """Return packaged Hydra config names for a config group."""
    group_files = files(f"benchmfg.config.{group}")
    return sorted(
        path.name.removesuffix(".yaml")
        for path in group_files.iterdir()
        if path.name.endswith(".yaml")
    )


def _print_values(values: list[str]) -> None:
    for value in values:
        print(value)


def _run_hydra(args: list[str]) -> None:
    old_argv = sys.argv[:]
    sys.argv = ["benchmfg train", *args]
    try:
        from benchmfg.train import main as train_main

        train_main()
    finally:
        sys.argv = old_argv


def _run_module(module: str, args: list[str]) -> None:
    old_argv = sys.argv[:]
    sys.argv = [module, *args]
    try:
        runpy.run_module(module, run_name="__main__")
    finally:
        sys.argv = old_argv


def _plot(args: list[str]) -> None:
    if not args:
        raise SystemExit("Usage: benchmfg plot {single-run,sweep,compare} ...")

    command, rest = args[0], args[1:]
    modules = {
        "single-run": "benchmfg.utility.plot_single_run",
        "sweep": "benchmfg.utility.plot_sweep",
        "compare": "benchmfg.utility.plot_comparison",
    }
    try:
        module = modules[command]
    except KeyError as exc:
        raise SystemExit(f"Unknown plot command: {command}") from exc
    _run_module(module, rest)


def _garnet(args: list[str]) -> None:
    if not args:
        from benchmfg import banners

        banners.garnet()
        return

    command, rest = args[0], args[1:]
    modules = {
        "scaling": "benchmfg.garnet.matrix",
        "aggregate": "benchmfg.garnet.aggregate",
        "plot-scaling": "benchmfg.garnet.plot",
    }
    try:
        module = modules[command]
    except KeyError as exc:
        raise SystemExit(
            "Unknown garnet command: "
            f"{command}. Use one of: scaling, aggregate, plot-scaling"
        ) from exc
    _run_module(module, rest)


def _list(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="benchmfg list")
    parser.add_argument(
        "kind",
        choices=["envs", "environments", "algos", "algorithms", "plot-algorithms"],
    )
    parsed = parser.parse_args(args)
    if parsed.kind in {"envs", "environments"}:
        _print_values(_config_options("environment"))
    elif parsed.kind in {"algos", "algorithms"}:
        _print_values(_config_options("algorithm"))
    else:
        _print_values(ALGORITHMS)


def _env(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="benchmfg env")
    parser.add_argument("command", choices=["list"])
    parser.parse_args(args)
    _print_values(_config_options("environment"))


def _algo(args: list[str]) -> None:
    parser = argparse.ArgumentParser(prog="benchmfg algo")
    parser.add_argument("command", choices=["list"])
    parser.parse_args(args)
    _print_values(_config_options("algorithm"))


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="benchmfg",
        description="Run and analyze BenchMFG experiments.",
    )
    parser.add_argument(
        "command",
        nargs="?",
        help="Command: train, sweep, plot, list, or a Hydra override for train.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """Route BenchMFG commands."""
    args = list(sys.argv[1:] if argv is None else argv)
    if not args or args[0] in {"-h", "--help"}:
        _parser().print_help()
        print()
        print("Commands:")
        print("  hello")
        print("  garnet")
        print("  garnet {scaling,aggregate,plot-scaling} ...")
        print("  mfpso")
        print("  train [HYDRA_OVERRIDES...]")
        print("  sweep [HYDRA_OVERRIDES...]")
        print("  plot {single-run,sweep,compare} ...")
        print("  env list")
        print("  algo list")
        print("  list {envs,algorithms,plot-algorithms}")
        return

    command, rest = args[0], args[1:]
    if command in {"hello", "mfpso"}:
        from benchmfg import banners

        getattr(banners, command)()
    elif command == "garnet":
        _garnet(rest)
    elif command == "train":
        _run_hydra(rest)
    elif command == "sweep":
        _run_hydra(["-m", *rest])
    elif command == "plot":
        _plot(rest)
    elif command in {"env", "environment"}:
        _env(rest)
    elif command in {"algo", "algorithm"}:
        _algo(rest)
    elif command == "list":
        _list(rest)
    elif command.endswith(".py") and Path(command).name == "main.py":
        _run_hydra(rest)
    else:
        _run_hydra(args)


if __name__ == "__main__":
    main()
