PYTHON_VERSION := 3.12.13
VENV := .venv
PYTHON := $(VENV)/bin/python

.PHONY: venv
venv:
	@if [ -d "$(VENV)" ]; then \
		echo "$(VENV) already exists. Skipping virtual environment creation."; \
	else \
		uv python pin ${PYTHON_VERSION}; \
		uv venv $(VENV) --prompt ${VENV} --python ${PYTHON_VERSION}; \
	fi

.PHONY: sync
sync:
	uv sync --dev

.PHONY: clean
clean:
	@if [ -d "$(VENV)" ]; then \
		rm -rf $(VENV); \
		echo "Removed virtual environment $(VENV)"; \
	else \
		echo "Virtual environment $(VENV) does not exist. Nothing to clean."; \
	fi

.PHONY: test
test:
	uv run pytest


.PHONY: check
check:
	uv run ruff check

.PHONY: format
format:
	uv run ruff format --check
