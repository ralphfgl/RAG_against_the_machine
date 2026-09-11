SRCS_DIR = src
UV_PY = uv run



install:
	uv sync --cache-dir ~/goinfre/.uv_cache sync

install-dataset:
	@mkdir -p data/raw
	@unzip vllm-0.10.1.zip -d data/raw
	@echo "vllm-0.10.1.zip extracted to data/raw/vllm-0.10.1"

run:
	@uv run -m src $(ARGS)

debug:
	@uv run -m src -v $(ARGS)

clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name ".mypy_cache" -exec rm -rf {} +
	rm -rf __pycache__ .mypy_cache .pytest_cache

fclean: clean
	rm -rf .venv

lint:
	uv run flake8 src
	uv run mypy src --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

lint-strict:
	uv run flake8 src
	uv run mypy src --strict --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs

.PHONY: install run debug clean fclean lint lint-strict install-dataset
