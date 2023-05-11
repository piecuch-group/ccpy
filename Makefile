## Notes on this file. The `help` target parses the file and can generate
## headings and docstrings for targets, in the order they are in the file. To
## create a heading make a comment like `##@ Heading Content`. To document
## targets make a comment on the same line as the target name with two `##
## docstring explanation...`. To leave a target undocumented simply provide no
## docstring.

SRC := ccpy
ISORT := isort $(SRC)
BLACK := black $(SRC)
AUTOFLAKE := autoflake -ir --remove-all-unused-imports --ignore-init-module-imports --remove-unused-variables $(SRC)
MYPY := mypy ccpy
PYLINT := pylint ccpy


##@ Getting Started

bootstrap: ## Bootstrap project development
	pip install -r requirements-dev.txt
	pip install -e .
	pre-commit install
.PHONY: bootstrap


##@ QA

qa: format-check lint test  ## Run all QA rules
.PHONY: qa

format: ## Run all formatters
	pre-commit run --all-files
.PHONY: format

format-check: ## Check format
	$(ISORT) --check-only
	$(BLACK) --check
.PHONY: format-check

lint: ## Run linters
	$(PYLINT)
	$(MYPY)
.PHONY: lint

validate: ## Run typecheck
	$(MYPY)
.PHONY: validate

test: ## Run tests
	pytest
.PHONY: test


##@ Development

all: ## Compile all fortran modules
	cd ccpy/utilities/updates && $(MAKE) $@
.PHONY: all

check-dist: ## Check distribution
	python setup.py check -ms
	python setup.py sdist
	twine check dist/*
.PHONY: check-dist

docs: ## Build documentation
	$(BLACK) -l 80 docs/examples
	mkdocs build
.PHONY: docs


##@ Help

# An automatic help command: https://www.padok.fr/en/blog/beautiful-makefile-awk
.DEFAULT_GOAL := help

help: ## (DEFAULT) This command, show the help message
	@echo "See CONTRIBUTING.md for dependencies and more detailed instructions, then run this:"
	@echo "  > make setup"
	@echo ""
	@echo "The QA commands should all run without any further setup at this point."
	@echo "  > make format validate typecheck test-unit"
	@echo ""
	@echo "You can also set up a live, hot-reloading, code mirroring DevEnv with services in the DevEnv section."
	@echo "  > make up"
	@echo ""
	@echo "Or just run the integration tests against the DevEnv"
	@echo "  > make test-integ"
	@echo ""
	@echo "Then shut it down when you don't need it:"
	@echo "  > make down"
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_0-9-]+:.*?##/ { printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) } ' $(MAKEFILE_LIST)
.PHONY: help
