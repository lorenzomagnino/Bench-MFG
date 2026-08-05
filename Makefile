.PHONY: install run profile test lint format clean help hello garnet mfpso algo-parameters

PROFILE_OUT ?= profiling_output/profile_$(shell date +%Y%m%d_%H%M%S).html

help:
	@echo "Available commands:"
	@echo "  make hello     - Print a BenchMFG quick-start guide"
	@echo "  make garnet    - Print MF-Garnet usage notes"
	@echo "  make mfpso     - Print Mean Field PSO usage notes"
	@echo "  make algo-parameters - Print algorithm hyperparameters and sweep ranges"
	@echo "  make install   - Install dependencies with uv"
	@echo "  make run       - Run main.py"
	@echo "  make profile   - Profile main.py and open flamegraph HTML"
	@echo "  make test      - Run tests"
	@echo "  make lint      - Run ruff linter"
	@echo "  make format    - Run black formatter"
	@echo "  make clean     - Clean cache files"

install:
	uv pip install pyinstrument
	uv sync

run:
	uv run python main.py

profile:
	@mkdir -p profiling_output
	uv run python -m pyinstrument --renderer html --outfile $(PROFILE_OUT) main.py
	uv run python scripts/patch_flamegraph_colors.py $(PROFILE_OUT)
	@echo "Profile saved → $(PROFILE_OUT)"
	open $(PROFILE_OUT)

test:
	uv run pytest tests/

lint:
	uv run ruff check .

format:
	uv run black .

clean:
	find . -type d -name "__pycache__" -exec rm -r {} + 2>/dev/null || true

hello:
	@uv run benchmfg hello

garnet:
	@uv run benchmfg garnet

mfpso:
	@uv run benchmfg mfpso

algo-parameters:
	@uv run benchmfg algo-parameters
