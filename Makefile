.DEFAULT_GOAL := help

src_dirs := compressai_trainer tests


.PHONY: help
help: ## Show this message
	@echo "Usage: make COMMAND\n\nCommands:"
	@grep '\s##\s' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' | cat


.PHONY: install
install: ## Install via poetry
	curl -sSL https://install.python-poetry.org | python3 -
	poetry install --with=dev,docs,tests
	@echo "Virtual environment created in $(poetry env list --full-path)"
	@echo ""
	@echo "\033[1;34mIMPORTANT!\033[0mPlease run:"
	@echo "poetry run pip install --editable /path/to/compressai"
	@echo "Then, to activate the virtual environment, please run:"
	@echo "poetry shell"


.PHONY: static-analysis
static-analysis: check-black check-isort check-ruff ## Run all static checks


.PHONY: check-black
check-black: ## Run black
	@echo "--> Running black"
	black --check --diff $(src_dirs)


.PHONY: check-isort
check-isort: ## Run isort
	@echo "--> Running isort"
	isort --check-only $(src_dirs)


.PHONY: check-ruff
check-ruff: ## Run ruff
	@echo "--> Running ruff"
	ruff check $(src_dirs)
	ruff format --check --diff $(src_dirs)


.PHONY: tests
tests: ## Run tests
	@echo "--> Running Python tests"
	pytest tests/


.PHONY: docs docs-serve
docs: ## Build documentation
	@echo "--> Building docs"
	@cd docs && SPHINXOPTS="-W" make html


docs-serve: docs ## Serve documentation
	@echo "--> Serving docs"
	@cd docs && sphinx-serve --build _build
