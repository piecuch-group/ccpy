.DEFAULT_GOAL := all
src := ccpy
isort = isort $(src)
black = black $(src)
autoflake = autoflake -ir --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables $(src)
mypy = mypy ccpy
pylint = pylint ccpy

.PHONY: all
all:
	cd ccpy/utilities/updates && $(MAKE) $@

.PHONY: install
install:
	$(CONDA_PREFIX)/bin/pip install -e .

.PHONY: all-quality
all-quality: format-check lint test

.PHONY: format
format:
	$(autoflake)
	$(isort)
	$(black)

.PHONY: format-check
format-check:
	$(isort) --check-only
	$(black) --check

.PHONY: lint
lint:
	$(pylint)
	$(mypy)

.PHONY: check-dist
check-dist:
	python setup.py check -ms
	python setup.py sdist
	twine check dist/*

.PHONY: mypy
mypy:
	$(mypy)

.PHONY: test
test:
	pytest

.PHONY: docs
docs:
	black -l 80 docs/examples
	mkdocs build

.PHONY: bootstrap
bootstrap:
	$(CONDA_PREFIX)/bin/pip install -r requirements-dev.txt
	$(CONDA_PREFIX)/bin/pip install -r requirements.txt
