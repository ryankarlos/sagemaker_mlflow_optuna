# Makefile for SageMaker MLflow Optuna Pipeline

.PHONY: setup test lint format clean help

# Install dependencies
setup:
	uv sync

# Run tests
test:
	uv run pytest tests/ -v --cov=. --cov-report=term-missing

# Run linting
lint:
	uv run ruff check pipelines/ steps/ utils/ data/ tests/
	uv run black --check pipelines/ steps/ utils/ data/ tests/

# Format code
format:
	uv run black pipelines/ steps/ utils/ data/ tests/
	uv run ruff check --fix pipelines/ steps/ utils/ data/ tests/

# Generate sample gambling data
generate-data:
	uv run python -m data.simulate_gambling_data

# Run FM Optuna pipeline (local mode)
pipeline-fm:
	uv run python -m pipelines.fm_optuna_train --local --n_users 1000 --n_games 50 --max_trials 10

# Run FM Optuna pipeline (full)
pipeline-fm-full:
	uv run python -m pipelines.fm_optuna_train --local --n_users 5000 --n_games 100 --max_trials 30

# Start MLflow UI
mlflow-ui:
	uv run mlflow ui --port 5000

# Clean artifacts
clean:
	rm -rf mlruns/
	rm -rf results/
	rm -rf .pytest_cache
	rm -rf htmlcov
	rm -f .coverage
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true

# Build documentation
docs-build:
	uv run mkdocs build

# Serve documentation locally
docs-serve:
	uv run mkdocs serve

# Full CI/CD simulation
all: setup lint test
	@echo "CI/CD simulation complete!"

# Help
help:
	@echo "Available targets:"
	@echo "  setup          - Install dependencies"
	@echo "  test           - Run tests with coverage"
	@echo "  lint           - Run linting"
	@echo "  format         - Format code"
	@echo "  generate-data  - Generate sample gambling data"
	@echo "  pipeline-fm    - Run FM Optuna pipeline (quick local test)"
	@echo "  pipeline-fm-full - Run FM Optuna pipeline (full local)"
	@echo "  mlflow-ui      - Start MLflow UI"
	@echo "  docs-build     - Build documentation"
	@echo "  docs-serve     - Serve documentation locally"
	@echo "  clean          - Clean artifacts"
	@echo "  all            - Full CI/CD simulation"
