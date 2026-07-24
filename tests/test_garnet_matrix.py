import importlib.util
from pathlib import Path
import sys

from benchmfg.envs.mfg_model_class_jit import get_jax_device
import pytest

_SCRIPT = Path(__file__).parents[1] / "scripts/garnet/run_garnet_matrix.py"
_SPEC = importlib.util.spec_from_file_location("run_garnet_matrix", _SCRIPT)
_MODULE = importlib.util.module_from_spec(_SPEC)
assert _SPEC.loader is not None
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


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


def test_gpu_id_requires_gpu_device():
    args = _MODULE._parser().parse_args(["--gpu-id", "0"])
    job = _MODULE.build_jobs(args)[0]

    with pytest.raises(ValueError, match="requires"):
        _MODULE._run(job, args)


def test_cpu_device_selection():
    assert get_jax_device("cpu").platform == "cpu"


def test_unknown_device_is_rejected():
    with pytest.raises(ValueError, match="Unsupported device"):
        get_jax_device("cuda:gpu0")
