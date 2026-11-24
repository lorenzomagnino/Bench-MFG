"""Tests for the notebook-facing API: History, plot_gif, print_config_*."""

import benchmfg as bmfg
import numpy as np
import pytest

OVERRIDES = [
    "environment=lasry_lions_chain",
    "algorithm=omd",
    "algorithm.omd.num_iterations=3",
    "device=cpu",
]


@pytest.mark.parametrize("use_python", [False, True])
def test_history_records_every_iteration(use_python):
    cfg = bmfg.load_config(OVERRIDES + [f"algorithm.omd.use_python={use_python}"])
    env, pi0 = bmfg.make_environment(cfg)
    solver = bmfg.make_solver(cfg, environment=env, initial_policy=pi0)
    history = bmfg.History()
    policy, mean_field, exploitabilities = solver.eval(logger=history)

    assert history.exploitabilities == pytest.approx(exploitabilities)
    assert len(history.mean_fields) == len(history.policies) == 4
    np.testing.assert_allclose(history.policies[-1], policy, atol=1e-6)
    np.testing.assert_allclose(history.mean_fields[-1], mean_field, atol=1e-6)


@pytest.mark.parametrize(
    "kind, frames",
    [
        ("mean_field", [np.full(5, 0.2), np.eye(5)[2]]),
        ("policy", [np.full((5, 3), 1 / 3), np.eye(3)[np.zeros(5, int)]]),
        ("exploitability", [10.0, 1.0, 0.1]),
    ],
)
def test_plot_gif_writes_file(tmp_path, kind, frames):
    out = bmfg.plot_gif(frames, tmp_path / "sub" / "x.gif", kind=kind)
    assert out.read_bytes()[:6] == b"GIF89a"


@pytest.mark.parametrize(
    "kind, frames",
    [
        ("mean_field", [np.full(4, 0.25), np.eye(4)[3]]),
        ("policy", [np.full((4, 5), 0.2), np.eye(5)[np.full(4, 4)]]),
    ],
)
def test_plot_gif_grid(tmp_path, kind, frames):
    out = bmfg.plot_gif(frames, tmp_path / "x.gif", kind=kind, grid_dim=(2, 2))
    assert out.read_bytes()[:6] == b"GIF89a"


def test_plot_gif_rejects_bad_kind(tmp_path):
    with pytest.raises(ValueError):
        bmfg.plot_gif([np.ones(3)], tmp_path / "x.gif", kind="nope")


def test_print_config(capsys):
    bmfg.print_config_env(OVERRIDES)
    bmfg.print_config_algo(OVERRIDES)
    out = capsys.readouterr().out
    assert "LasryLionsChain" in out
    assert "learning_rate" in out


def test_make_environment_is_seeded():
    _, pi_a = bmfg.make_environment(OVERRIDES)
    _, pi_b = bmfg.make_environment(OVERRIDES)
    np.testing.assert_array_equal(pi_a, pi_b)


def test_show_gifs_embeds_all(tmp_path):
    pytest.importorskip("IPython")  # notebook-only helper; IPython is optional
    paths = [
        bmfg.plot_gif([1.0, 0.5], tmp_path / f"{i}.gif", kind="exploitability")
        for i in range(2)
    ]
    assert bmfg.show_gifs(*paths).data.count("data:image/gif;base64,") == 2
