UV := uv
PACKAGE := fse
SRC := src/$(PACKAGE)

.DEFAULT_GOAL := help

.PHONY: help env install init sync rebuild-ext format format-check lint lint-fix typecheck test test-network test-cov check build clean pre-commit

##@ Setup
help: ## Show this help
	@awk 'BEGIN {FS = ":.*##"; printf "\nUsage:\n  make \033[36m<target>\033[0m\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2 } /^##@/ { printf "\n\033[1m%s\033[0m\n", substr($$0, 5) }' $(MAKEFILE_LIST)

env: ## Create virtual environment
	$(UV) venv

install: ## Install package with all dependency groups
	$(UV) sync --all-groups

init: env install pre-commit ## Full setup: env + install + pre-commit

sync: ## Sync all dependencies from lockfile
	$(UV) sync --all-groups

rebuild-ext: ## Rebuild the Cython extension after editing .pyx or .pxd files
	$(UV) sync --all-groups --reinstall-package $(PACKAGE)

##@ Quality
format: ## Format code with ruff
	$(UV) run ruff format $(SRC) tests
	$(UV) run ruff check --fix --fix-only $(SRC) tests

format-check: ## Check formatting (CI mode)
	$(UV) run ruff format --check $(SRC) tests

lint: ## Lint code with ruff
	$(UV) run ruff check $(SRC) tests

lint-fix: ## Lint and auto-fix
	$(UV) run ruff check --fix $(SRC) tests

typecheck: ## Type check with mypy
	$(UV) run mypy

##@ Testing
test: ## Run tests without network access
	$(UV) run pytest tests -m "not network"

test-network: ## Run the tests that download from the Hugging Face Hub
	$(UV) run pytest tests -m network

test-cov: ## Run tests with coverage
	$(UV) run pytest tests -m "not network" --cov=$(PACKAGE) --cov-report=term-missing

##@ Combined
check: format-check lint typecheck test ## Run all checks: format + lint + typecheck + test

##@ Build
build: clean ## Build sdist and wheel
	$(UV) build
	$(UV) run --with twine twine check dist/*

clean: ## Remove build artifacts and caches
	rm -rf build/ dist/ wheelhouse/ *.egg-info src/*.egg-info
	rm -rf .pytest_cache .ruff_cache htmlcov .coverage .coverage.*
	rm -f $(SRC)/*.c
	find . -type d -name __pycache__ -not -path "./.venv/*" -exec rm -rf {} + 2>/dev/null || true

##@ Hooks
pre-commit: ## Install pre-commit hooks
	$(UV) run pre-commit install
