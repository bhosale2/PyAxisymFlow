#* Variables
PYTHON := python3
PYTHONPATH := `pwd`

#* Poetry
.PHONY: poetry-download
poetry-download:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) -

.PHONY: poetry-remove
poetry-remove:
	curl -sSL https://install.python-poetry.org/ | $(PYTHON) - --uninstall

#* Installation
.PHONY: install
install:
	poetry install
	# install pyaxisymflow.core deps
	cd pyaxisymflow/core/src && make clean && make bind

.PHONY: install_with_new_dependency
install_with_new_dependency:
	poetry lock
	poetry install

.PHONY: pre-commit-install
pre-commit-install:
	poetry run pre-commit install

#* Formatters
.PHONY: black
black:
	poetry run black --version
	poetry run black --config pyproject.toml pyaxisymflow examples

.PHONY: black-check
black-check:
	poetry run black --version
	poetry run black --diff --check --config pyproject.toml pyaxisymflow examples

.PHONY: flake8
flake8:
	poetry run flake8 --version
	poetry run flake8 pyaxisymflow

.PHONY: autoflake8-check
autoflake8-check:
	poetry run autoflake8 --version
	poetry run autoflake8 -r --exclude '__init__.py' pyaxisymflow examples
	poetry run autoflake8 --check -r --exclude '__init__.py' pyaxisymflow examples

.PHONY: format-codestyle
format-codestyle: black flake8

.PHONY: check-codestyle
check-codestyle: black-check flake8 autoflake8-check

.PHONY: formatting
formatting: format-codestyle

.PHONY: test
test:
	poetry run pytest

.PHONY: test_coverage
test_coverage:
	NUMBA_DISABLE_JIT=1 poetry run pytest --cov=pyaxisymflow

.PHONY: test_coverage_xml
test_coverage_xml:
	NUMBA_DISABLE_JIT=1 poetry run pytest --cov=pyaxisymflow --cov-report=xml

.PHONY: update-dev-deps
update-dev-deps:
	poetry add -D pytest@latest coverage@latest pytest-html@latest pytest-cov@latest black@latest

#* Cleaning
.PHONY: pycache-remove
pycache-remove:
	find . | grep -E "(__pycache__|\.pyc|\.pyo$$)" | xargs rm -rf

.PHONY: dsstore-remove
dsstore-remove:
	find . | grep -E ".DS_Store" | xargs rm -rf

.PHONY: ipynbcheckpoints-remove
ipynbcheckpoints-remove:
	find . | grep -E ".ipynb_checkpoints" | xargs rm -rf

.PHONY: pytestcache-remove
pytestcache-remove:
	find . | grep -E ".pytest_cache" | xargs rm -rf

.PHONY: build-remove
build-remove:
	rm -rf build/

.PHONY: cleanup
cleanup: pycache-remove dsstore-remove ipynbcheckpoints-remove pytestcache-remove

all: format-codestyle cleanup

ci: check-codestyle
