from benchmfg.cli import main


def test_algo_parameters_lists_every_algorithm(capsys):
    main(["algo-parameters"])
    out = capsys.readouterr().out
    for algo in ("pso", "omd", "damped_fixed_point", "pi"):
        assert f"algorithm={algo}" in out
    assert "num_particles" in out
    assert "benchmfg sweep" in out
