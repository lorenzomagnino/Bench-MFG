"""Collect best-model NPZ artifacts into results/ for self-contained plot regeneration.

For each environment × algorithm, reads the per-algorithm best_model.yaml, picks the
seed with the lowest final exploitability, and copies:
  - final_mean_field.npz
  - final_policy.npz
into results/{environment}/{algorithm}/.

Also copies those files for the overall comparison winner into
results/{environment}/best/  (used by plot_mean_field_from_npz / plot_policy_from_npz).

After running this script every plot can be regenerated purely from src/results/
without touching the outputs/ directory.

Usage:
    benchmfg collect_best_artifacts
    benchmfg collect_best_artifacts --outputs-dir /path/to/outputs
"""

import argparse
from pathlib import Path
import shutil
import sys

import yaml

_src = Path(__file__).resolve().parent.parent
if str(_src) not in sys.path:
    sys.path.insert(0, str(_src))

from benchmfg.utility.plot_discovery import version_to_algorithm_dir  # noqa: E402

_NPZ_FILES = ["final_mean_field.npz", "final_policy.npz"]


def _best_seed_run_dir(
    environment: str,
    best_version: str,
    seeds: list[dict],
    outputs_dir: Path,
) -> Path | None:
    """Return the run directory for the seed with the lowest final exploitability."""
    algo_dir = version_to_algorithm_dir(best_version)

    valid = [s for s in seeds if "final_exploitability" in s] or seeds
    best_seed = min(valid, key=lambda s: s.get("final_exploitability", float("inf")))

    seed_name = best_seed["seed"]
    run_id = best_seed.get("run_id")

    version_dir = outputs_dir / environment / algo_dir / seed_name / best_version
    if not version_dir.exists():
        return None

    if run_id:
        candidate = version_dir / run_id
        if candidate.exists():
            return candidate

    subdirs = [d for d in version_dir.iterdir() if d.is_dir()]
    return max(subdirs, key=lambda d: d.stat().st_mtime) if subdirs else None


def _copy_npz(run_dir: Path, dest_dir: Path, label: str) -> None:
    copied = []
    for name in _NPZ_FILES:
        src = run_dir / name
        if src.exists():
            shutil.copy2(src, dest_dir / name)
            copied.append(name)
    status = ", ".join(copied) if copied else "nothing to copy"
    print(f"  [{label}] {status}")


def collect_for_environment(
    environment: str, results_dir: Path, outputs_dir: Path
) -> None:
    env_dir = results_dir / environment

    # Per-algorithm yamls
    for algo_path in sorted(env_dir.iterdir()):
        if not algo_path.is_dir() or algo_path.name == "best":
            continue
        yaml_path = algo_path / "best_model.yaml"
        if not yaml_path.exists():
            continue

        with open(yaml_path) as f:
            data = yaml.safe_load(f)

        best_version = data.get("best_version")
        seeds = data.get("seeds", [])
        if not best_version or not seeds:
            print(
                f"  [{environment}/{algo_path.name}] skipped: missing best_version/seeds"
            )
            continue

        run_dir = _best_seed_run_dir(environment, best_version, seeds, outputs_dir)
        if run_dir is None:
            print(
                f"  [{environment}/{algo_path.name}] run dir not found in outputs, skipping"
            )
            continue

        _copy_npz(run_dir, algo_path, f"{environment}/{algo_path.name}")

    # Overall comparison best
    overall_yaml = env_dir / "best" / "best_model.yaml"
    if overall_yaml.exists():
        with open(overall_yaml) as f:
            data = yaml.safe_load(f)

        best_version = data.get("best_version")
        seeds = data.get("seeds", [])
        if best_version and seeds:
            run_dir = _best_seed_run_dir(environment, best_version, seeds, outputs_dir)
            if run_dir is not None:
                _copy_npz(run_dir, env_dir / "best", f"{environment}/best")
            else:
                print(f"  [{environment}/best] run dir not found in outputs, skipping")


def collect_all(outputs_dir: Path) -> None:
    project_root = Path.cwd()
    results_dir = project_root / "results"

    environments = sorted(d.name for d in results_dir.iterdir() if d.is_dir())
    if not environments:
        print("No environments found in results/")
        return

    for env in environments:
        print(f"\n{env}")
        collect_for_environment(env, results_dir, outputs_dir)

    print("\nDone.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Copy best-model NPZ files into results/ for self-contained plotting."
    )
    parser.add_argument(
        "--outputs-dir",
        type=str,
        default="outputs",
        help="Root outputs directory (default: outputs)",
    )
    args = parser.parse_args()

    outputs_path = Path(args.outputs_dir).resolve()
    if not outputs_path.exists():
        print(f"Outputs directory not found: {outputs_path}")
        sys.exit(1)

    collect_all(outputs_path)
