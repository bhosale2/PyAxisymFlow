<div align='center'>
<h1> PyAxisymFlow </h1>

[![CI][badge-CI]][link-CI] [![DOI][badge-doi]][link-doi]
</div>

Python implementation of an axisymmetric elastohydrodynamic solver, for resolving flow-structure interaction of 3D axisymmetric
mixed soft/rigid bodies in viscous flows.

## Installation

Below are steps of how to install `pyaxisymflow`. We mainly use `poetry` to manage
the project, although most of the important commands will be provided in `Makefile`.

1. Clone!

First **create the fork repository and clone** to your local machine.

2. Virtual python workspace: `conda`, `pyenv`, or `venv`.

We recommend using python version above 3.8.0.

```bash
conda create --name pyaxisymflow-env
conda activate pyaxisymflow-env
conda install python==3.10
```

3. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
make pre-commit-install
```

## Contribution

If you would like to participate, please read our [contribution guideline](CONTRIBUTING.md)

## Citation

We ask that any publications which use SophT cite as following:

```
@software{yashraj_bhosale_2023_7658925,
  author       = {Yashraj Bhosale and
                  Gaurav Upadhyay and
                  Songyuan Cui and
                  Fan Kiat Chan and
                  Mattia Gazzola},
  title        = {PyAxisymFlow},
  month        = feb,
  year         = 2023,
  publisher    = {Zenodo},
  version      = {0.0.1},
  doi          = {10.5281/zenodo.7658925},
  url          = {https://doi.org/10.5281/zenodo.7658925}
}
```

## Senior Developers ✨
_Names arranged alphabetically_
- [Fan Kiat Chan](https://github.com/fankiat)
- Gaurav Upadhyay
- Songyuan Cui
- [Yashraj Bhosale](https://github.com/bhosale2)


[badge-doi]: https://zenodo.org/badge/DOI/10.5281/zenodo.7658925.svg
[badge-CI]: https://github.com/bhosale2/PyAxisymFlow/workflows/CI/badge.svg
[link-doi]: https://doi.org/10.5281/zenodo.7658925
[link-CI]: https://github.com/bhosale2/PyAxisymFlow/actions
