UV := uv
PACKAGE := fse
SRC := src/$(PACKAGE)
LINT_PATHS := $(SRC) tests tools benchmarks

# The worker processes of pytest-xdist for the targets that run the whole suite: `auto` is
# one per logical CPU, `make test PYTEST_N=0` runs serially in one process (to debug a test,
# or as the baseline of a measurement). PYTEST_DIST is the xdist distribution mode.
PYTEST_N ?= auto
PYTEST_DIST ?= loadfile
XDIST = -n $(PYTEST_N) --dist $(PYTEST_DIST)

.DEFAULT_GOAL := help

.PHONY: help env install init sync rebuild-ext format format-check lint lint-fix guardrails typecheck test test-network test-st test-cov check build clean pre-commit profile-cpu profile-mem profile-rss

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

rebuild-ext: ## Rebuild the Cython extension after editing .pyx, .pxi, .pxd or .h files
	$(UV) sync --all-groups --reinstall-package $(PACKAGE)

##@ Quality
format: ## Format code with ruff
	$(UV) run ruff format $(LINT_PATHS)
	$(UV) run ruff check --fix --fix-only $(LINT_PATHS)

format-check: ## Check formatting (CI mode)
	$(UV) run ruff format --check $(LINT_PATHS)

lint: ## Lint with ruff, the import contracts, the guardrails, unused code and dependencies
	$(UV) run ruff check $(LINT_PATHS)
	$(UV) run lint-imports --no-cache
	$(UV) run python tools/check_guardrails.py
	$(UV) run vulture
	$(UV) run deptry src

lint-fix: ## Lint and auto-fix
	$(UV) run ruff check --fix $(LINT_PATHS)

guardrails: ## File length, complexity, native imports and test files against the ratchet
	$(UV) run python tools/check_guardrails.py

typecheck: ## Type check with mypy
	$(UV) run mypy

##@ Profiling
# make profile-cpu RUN="benchmarks/bench_store.py write --workers 10"; the outputs go to
# PROFILE_DIR (one file per kind, overwritten).
PROFILE_DIR ?= profiles
RUN ?=

profile-cpu: ## CPU profile of RUN with native frames: samply on macOS, py-spy --native on Linux
	@mkdir -p $(PROFILE_DIR)
	@if [ "$$(uname)" = "Darwin" ]; then \
		command -v samply >/dev/null || { echo "samply is missing: cargo install --locked samply"; exit 1; }; \
		samply record --save-only --unstable-presymbolicate -o $(PROFILE_DIR)/cpu.json -- $(UV) run python $(RUN); \
		$(UV) run python tools/profile_top.py $(PROFILE_DIR)/cpu.json; \
		echo "the flame graph: samply load $(PROFILE_DIR)/cpu.json"; \
	else \
		$(UV) run py-spy record --native --threads --subprocesses -f speedscope -o $(PROFILE_DIR)/cpu.speedscope.json -- python $(RUN); \
		echo "open it at https://www.speedscope.app"; \
	fi

profile-mem: ## Memory profile of RUN with native allocations (memray), as a flame graph over time
	@mkdir -p $(PROFILE_DIR)
	rm -f $(PROFILE_DIR)/mem.bin
	$(UV) run memray run --native --follow-fork -o $(PROFILE_DIR)/mem.bin $(RUN)
	$(UV) run memray flamegraph --temporal -f -o $(PROFILE_DIR)/mem.html $(PROFILE_DIR)/mem.bin
	$(UV) run memray stats $(PROFILE_DIR)/mem.bin | head -40

profile-rss: ## Resident memory of RUN and its child processes over time
	@mkdir -p $(PROFILE_DIR)
	$(UV) run python tools/rss.py --csv $(PROFILE_DIR)/rss.csv -- $(UV) run python $(RUN)

##@ Testing
test: ## Run tests without network access
	$(UV) run pytest tests -m "not network" $(XDIST)

test-network: ## Run the tests that download from the Hugging Face Hub
	$(UV) run pytest tests -m network

test-st: ## Run the comparisons with sentence-transformers (group st), required to run
	FSE_REQUIRE_ST=1 $(UV) run --group st pytest tests/test_encoder_sentence_transformers_parity.py tests/test_encoder_sentence_transformers.py -m "not network"

test-cov: ## Run tests with coverage
	$(UV) run pytest tests -m "not network" $(XDIST) --cov=$(PACKAGE) --cov-report=term-missing --cov-fail-under=0

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
