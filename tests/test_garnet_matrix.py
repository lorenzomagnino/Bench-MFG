from pathlib import Path

from benchmfg.envs.mfg_model_class_jit import get_jax_device
from benchmfg.garnet import aggregate as _AGG, matrix as _MODULE
import numpy as np
import pytest
import yaml


def _write_run(
    root: Path,
    states: int,
    algorithm: str,
    garnet_seed: int,
    random_seed: int,
    *,
    variant: str | None = None,
    runtime_s: float = 1.0,
    final: float = 0.5,
    dynamics: str = "additive",
    reward: str = "multiplicative",
) -> Path:
    """Create a run directory shaped like the ones save_results writes."""
    run_dir = (
        root / f"{algorithm}_{states}_{garnet_seed}_{random_seed}_{dynamics}_{reward}"
    )
    run_dir.mkdir(parents=True)
    algorithm_cfg: dict = {"_target_": algorithm}
    if algorithm == "DampedFP":
        algorithm_cfg["dampedfp"] = {"lambda_schedule": variant}
    if algorithm == "PI":
        algorithm_cfg["pi"] = {"variant": variant}
    config = {
        "environment": {
            "num_states": states,
            "num_actions": 6,
            "reward": {
                "mfgarnet": {
                    "branching_factor": 6,
                    "seed": garnet_seed,
                    "dynamics_structure": dynamics,
                    "reward_structure": reward,
                }
            },
        },
        "experiment": {"random_seed": random_seed},
        "algorithm": algorithm_cfg,
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    np.savez(run_dir / "metrics.npz", runtime_s=runtime_s)
    np.savez(run_dir / "exploitabilities.npz", exploitabilities=np.array([1.0, final]))
    return run_dir


def test_matrix_job_count_and_seed_pairing():
    args = _MODULE._parser().parse_args(
        ["--states", "20", "80", "130", "400", "--num-seeds", "3"]
    )
    jobs = _MODULE.build_jobs(args)

    assert len(jobs) == 96
    assert jobs[0].algorithm == "pso"
    assert jobs[0].garnet_seed == 0
    assert jobs[0].algorithm_seed == 42
    assert jobs[2].garnet_seed == 2
    assert jobs[2].algorithm_seed == 111
    assert jobs[-1].states == 400


def test_default_is_two_seeds():
    args = _MODULE._parser().parse_args(["--states", "20", "400"])
    jobs = _MODULE.build_jobs(args)

    assert args.num_seeds == 2
    # 2 state counts x 8 algorithms x 2 seeds
    assert len(jobs) == 32
    assert {job.algorithm_seed for job in jobs} == {42, 10}


def test_gpu_id_requires_gpu_device():
    args = _MODULE._parser().parse_args(["--gpu-id", "0"])
    job = _MODULE.build_jobs(args)[0]

    with pytest.raises(ValueError, match="requires"):
        _MODULE._run(job, args, 1)


def test_cpu_device_selection():
    assert get_jax_device("cpu").platform == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device"):
        get_jax_device("cuda:gpu0")


def test_cell_key_matches_saved_run(tmp_path):
    """A finished run must produce exactly the key the launcher checks for resume."""
    _write_run(tmp_path, 20, "PSO", garnet_seed=0, random_seed=42)
    args = _MODULE._parser().parse_args(
        ["--states", "20", "--algorithms", "pso", "--num-seeds", "1"]
    )
    job = _MODULE.build_jobs(args)[0]

    assert _MODULE.cell_key(job, args) in _AGG.completed_cells(tmp_path)


def test_cell_key_matches_variant_algorithms(tmp_path):
    _write_run(
        tmp_path, 20, "PI", garnet_seed=0, random_seed=42, variant="policy_iteration"
    )
    _write_run(
        tmp_path, 20, "DampedFP", garnet_seed=0, random_seed=42, variant="damped"
    )
    args = _MODULE._parser().parse_args(
        [
            "--states",
            "20",
            "--algorithms",
            "pi:policy_iteration",
            "dampedfp:damped",
            "--num-seeds",
            "1",
        ]
    )
    done = _AGG.completed_cells(tmp_path)

    for job in _MODULE.build_jobs(args):
        assert _MODULE.cell_key(job, args) in done


def test_other_modality_does_not_satisfy_a_cell(tmp_path):
    """An A/M run must not make the launcher skip the matching M/A job."""
    _write_run(
        tmp_path,
        20,
        "PSO",
        garnet_seed=0,
        random_seed=42,
        dynamics="additive",
        reward="multiplicative",
    )
    ma_args = _MODULE._parser().parse_args(
        [
            "--states",
            "20",
            "--algorithms",
            "pso",
            "--num-seeds",
            "1",
            "--dynamics-structure",
            "multiplicative",
            "--reward-structure",
            "additive",
        ]
    )
    am_args = _MODULE._parser().parse_args(
        ["--states", "20", "--algorithms", "pso", "--num-seeds", "1"]
    )
    done = _AGG.completed_cells(tmp_path)

    assert _MODULE.cell_key(_MODULE.build_jobs(am_args)[0], am_args) in done
    assert _MODULE.cell_key(_MODULE.build_jobs(ma_args)[0], ma_args) not in done


def test_modalities_are_aggregated_separately(tmp_path):
    """A/M and M/A results at the same size must not be averaged together."""
    _write_run(
        tmp_path,
        20,
        "PSO",
        0,
        42,
        final=0.2,
        dynamics="additive",
        reward="multiplicative",
    )
    _write_run(
        tmp_path,
        20,
        "PSO",
        0,
        42,
        final=0.8,
        dynamics="multiplicative",
        reward="additive",
    )
    rows = _AGG.collect(tmp_path)

    assert len(rows) == 2
    by_modality = {row["modality"]: row for row in rows}
    assert set(by_modality) == {"A/M", "M/A"}
    assert by_modality["A/M"]["final_exploitability_mean"] == pytest.approx(0.2)
    assert by_modality["M/A"]["final_exploitability_mean"] == pytest.approx(0.8)
    assert all(row["runs"] == 1 for row in rows)


def test_float32_floor_is_reported_as_a_bound(tmp_path):
    """Sub-ULP exploitability means "solved", so print a floor, not fake digits."""
    # 5.96e-08 is 2^-24, one float32 ULP near 1.0.
    _write_run(tmp_path, 20, "PSO", 0, 42, final=5.96e-08)
    _write_run(tmp_path, 20, "PSO", 1, 10, final=1.19e-07)
    rows = _AGG.collect(tmp_path)

    assert rows[0]["seeds_at_floor"] == 2
    assert (
        _AGG.format_exploitability(
            rows[0]["final_exploitability_mean"], rows[0]["final_exploitability_std"]
        )
        == _AGG.FLOOR_LABEL
    )

    table = tmp_path / "table.md"
    _AGG.write_markdown(rows, table)
    text = table.read_text()
    assert _AGG.FLOOR_LABEL in text
    # The meaningless digits must not appear as a value.
    assert "5.96e-08 ±" not in text


def test_real_values_are_not_floored(tmp_path):
    _write_run(tmp_path, 20, "PSO", 0, 42, final=0.002)
    _write_run(tmp_path, 20, "PSO", 1, 10, final=0.004)
    rows = _AGG.collect(tmp_path)

    assert rows[0]["seeds_at_floor"] == 0
    assert _AGG.FLOOR_LABEL not in _AGG.format_exploitability(
        rows[0]["final_exploitability_mean"], rows[0]["final_exploitability_std"]
    )


def test_partially_floored_cell_reports_the_split(tmp_path):
    """One solved instance and one unsolved must not hide behind a mean."""
    _write_run(tmp_path, 20, "PSO", 0, 42, final=0.0)
    _write_run(tmp_path, 20, "PSO", 1, 10, final=0.0038)
    rows = _AGG.collect(tmp_path)

    assert rows[0]["seeds_at_floor"] == 1
    assert rows[0]["runs"] == 2
    table = tmp_path / "table.md"
    _AGG.write_markdown(rows, table)

    assert "[1/2]" in table.read_text()


def test_markdown_separates_modality_sections(tmp_path):
    _write_run(
        tmp_path,
        20,
        "PSO",
        0,
        42,
        final=0.2,
        dynamics="additive",
        reward="multiplicative",
    )
    _write_run(
        tmp_path,
        20,
        "PSO",
        0,
        42,
        final=0.8,
        dynamics="multiplicative",
        reward="additive",
    )
    table = tmp_path / "table.md"

    _AGG.write_markdown(_AGG.collect(tmp_path), table)
    text = table.read_text()

    assert "## Modality A/M" in text
    assert "## Modality M/A" in text


def test_non_garnet_runs_are_ignored(tmp_path):
    """outputs/ holds other environments; those runs are not Garnet table cells."""
    run_dir = tmp_path / "KineticCongestion_PSO"
    run_dir.mkdir(parents=True)
    config = {
        "environment": {"num_states": 20, "num_actions": 6, "reward": {"kinetic": {}}},
        "experiment": {"random_seed": 42},
        "algorithm": {"_target_": "PSO"},
    }
    (run_dir / "config.yaml").write_text(yaml.safe_dump(config))
    np.savez(run_dir / "metrics.npz", runtime_s=1.0)
    np.savez(run_dir / "exploitabilities.npz", exploitabilities=np.array([0.5]))
    _write_run(tmp_path, 20, "PSO", garnet_seed=0, random_seed=42)

    records = _AGG.scan_runs(tmp_path)

    assert len(records) == 1
    assert len(_AGG.completed_cells(tmp_path)) == 1


def test_partial_run_is_not_a_completed_cell(tmp_path):
    run_dir = _write_run(tmp_path, 20, "PSO", garnet_seed=0, random_seed=42)
    (run_dir / "exploitabilities.npz").unlink()

    assert _AGG.completed_cells(tmp_path) == set()


def test_aggregate_averages_over_seeds(tmp_path):
    _write_run(tmp_path, 20, "PSO", 0, 42, runtime_s=10.0, final=0.2)
    _write_run(tmp_path, 20, "PSO", 1, 10, runtime_s=20.0, final=0.4)
    rows = _AGG.collect(tmp_path)

    assert len(rows) == 1
    assert rows[0]["runs"] == 2
    assert rows[0]["runtime_mean_s"] == pytest.approx(15.0)
    assert rows[0]["final_exploitability_mean"] == pytest.approx(0.3)


def test_combined_pivot_has_exploitability_and_runtime_per_cell(tmp_path):
    _write_run(tmp_path, 20, "PSO", 0, 42, runtime_s=6.0, final=0.2)
    _write_run(tmp_path, 400, "PSO", 0, 42, runtime_s=435.0, final=0.9)
    table = tmp_path / "table.md"

    _AGG.write_markdown(_AGG.collect(tmp_path), table)
    lines = table.read_text().splitlines()
    pso = next(line for line in lines if line.startswith("| PSO "))

    # Each cell carries both metrics: "<exploitability> ± <std> / <seconds>s".
    assert "0.2" in pso and "6s" in pso
    assert "0.9" in pso and "435s" in pso
    assert "Exploitability and wall-clock time" in table.read_text()


def test_markdown_has_algorithm_by_states_pivot(tmp_path):
    _write_run(tmp_path, 20, "PSO", 0, 42, final=0.2)
    _write_run(tmp_path, 400, "PSO", 0, 42, final=0.4)
    _write_run(tmp_path, 20, "OMD", 0, 42, final=0.1)
    table = tmp_path / "table.md"

    _AGG.write_markdown(_AGG.collect(tmp_path), table)
    text = table.read_text()

    assert "| Algorithm | S=20 | S=400 |" in text
    assert "Final exploitability" in text
    assert "Runtime in seconds" in text
    # OMD has no S=400 cell yet, so that entry stays empty.
    omd_line = next(line for line in text.splitlines() if line.startswith("| OMD "))
    assert omd_line.strip().endswith("| - |")


def test_hung_cell_times_out_and_frees_its_gpu(tmp_path, monkeypatch):
    """A deadlocked cell must not block the sweep: kill it, fail it, release the GPU."""
    import queue
    import subprocess

    args = _MODULE._parser().parse_args(
        [
            "--states",
            "20",
            "--algorithms",
            "pso",
            "--num-seeds",
            "1",
            "--job-timeout",
            "0.2",
            "--output-root",
            str(tmp_path / "out"),
            "--log-dir",
            str(tmp_path / "logs"),
        ]
    )
    job = _MODULE.build_jobs(args)[0]

    def fake_run(*a, timeout=None, **kw):
        raise subprocess.TimeoutExpired(cmd="x", timeout=timeout)

    monkeypatch.setattr(_MODULE.subprocess, "run", fake_run)
    monkeypatch.setattr(_MODULE, "_GPU_POOL", queue.Queue())
    _MODULE._GPU_POOL.put(7)

    with pytest.raises(RuntimeError, match="timed out"):
        _MODULE._run(job, args, 1)

    # The GPU must go back to the pool or later cells would starve.
    assert _MODULE._GPU_POOL.get_nowait() == 7
