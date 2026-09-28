# BenchMFG

A benchmark suite for **Mean Field Game algorithms**, with JAX solvers,
configurable environments, and reproducible experiment workflows.

[Paper](https://arxiv.org/abs/2602.12517) ·
[Code on GitHub](https://github.com/lorenzomagnino/Bench-MFG) ·
[Package on PyPI](https://pypi.org/project/bench-mfg-suite/)

## Start with one experiment

Install the package and run a small CPU experiment:

```bash
pip install bench-mfg-suite
benchmfg train algorithm=omd environment=four_rooms_obstacles device=cpu algorithm.omd.num_iterations=20
```

BenchMFG provides a shared workflow for selecting environments and algorithms,
sweeping parameters, and plotting saved results. The MF-Garnet generator lets
you compare solvers across reproducible random games.

```{toctree}
:maxdepth: 1
:caption: Documentation

getting-started
MFG_GARNET
EXTENDING
```

## Paper and citation

[**Bench-MFG: A Benchmark Suite for Learning in Stationary Mean Field Games**](https://arxiv.org/abs/2602.12517)

Lorenzo Magnino, Jiacheng Shen, Matthieu Geist, Olivier Pietquin, and Mathieu Laurière.

arXiv preprint, 2026. [Read the PDF](https://arxiv.org/pdf/2602.12517).

The paper introduces a taxonomy of stationary MFG problems, the MF-Garnet random
game generator, and MF-PSO, a black-box solver for exploitability minimization.
It compares learning algorithms and proposes guidelines for experimental evaluation.

If you use BenchMFG in your research, please cite:

```bibtex
@misc{magnino2026benchmfg,
  title = {{Bench-MFG}: A Benchmark Suite for Learning in Stationary Mean Field Games},
  author = {Magnino, Lorenzo and Shen, Jiacheng and Geist, Matthieu and Pietquin, Olivier and Lauri\`ere, Mathieu},
  year = {2026},
  eprint = {2602.12517},
  archivePrefix = {arXiv},
  primaryClass = {cs.LG},
  url = {https://arxiv.org/abs/2602.12517}
}
```

## Contributing

To add an environment or solver, follow the [extension guide](EXTENDING.md).
For questions and bug reports, [open an issue](https://github.com/lorenzomagnino/Bench-MFG/issues).
