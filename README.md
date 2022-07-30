# PyAxisymFlow

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
conda install pip
```

3. Setup [`poetry`](https://python-poetry.org) and `dependencies`!

```bash
make poetry-download
make install
make pre-commit-install
```
