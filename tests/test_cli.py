from benchmfg.cli import main


def test_hello_mentions_package_guides(capsys):
    main(["hello"])
    out = capsys.readouterr().out
    assert "benchmfg garnet" in out
    assert "benchmfg mfpso" in out
    assert "benchmfg algo-parameters" in out


def test_algo_parameters_lists_every_algorithm(capsys):
    main(["algo-parameters"])
    out = capsys.readouterr().out
    for algo in ("pso", "omd", "damped_fixed_point", "pi"):
        assert f"algorithm={algo}" in out
    assert "num_particles" in out
    assert "benchmfg sweep" in out
